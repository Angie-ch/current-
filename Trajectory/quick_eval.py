"""
全量评估脚本：跑全部测试集，输出每个台风的误差，排出 Top-20 最精确的台风

用法:
  python quick_eval.py --ckpt checkpoints/best.pt --use_diffusion --cache_dir diffusion_era5_cache
  python quick_eval.py --ckpt checkpoints/best.pt
"""
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from collections import defaultdict

from config import model_cfg, data_cfg, train_cfg
from model import LT3PModel
from dataset import (
    LT3PDataset, denormalize_coords,
    filter_short_storms, filter_out_of_range_storms, split_storms_by_id,
)
from data_processing import load_tyc_storms


def main():
    parser = argparse.ArgumentParser(description="全量评估 + Top-20 台风排名")
    parser.add_argument("--ckpt", type=str, default="checkpoints/best.pt")
    parser.add_argument("--data_root", type=str, default=r"C:\Users\fyp\Desktop\Typhoon_data_final")
    parser.add_argument("--track_csv", type=str, default="processed_typhoon_tracks.csv")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--top_n", type=int, default=20, help="输出前N个最精确的台风")

    # 扩散ERA5
    parser.add_argument("--use_diffusion", action="store_true")
    parser.add_argument("--diffusion_code", type=str, default=r"C:\Users\fyp\Desktop\newtry")
    parser.add_argument("--diffusion_ckpt", type=str, default=r"C:\Users\fyp\Desktop\newtry\checkpoints\best.pt")
    parser.add_argument("--norm_stats", type=str, default=r"C:\Users\fyp\Desktop\newtry\norm_stats.pt")
    parser.add_argument("--cache_dir", type=str, default="diffusion_era5_cache")
    parser.add_argument("--ddim_steps", type=int, default=50)

    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    mode = "扩散ERA5" if args.use_diffusion else "真实ERA5"
    print(f"设备: {device}")
    print(f"ERA5来源: {mode}")

    # 1. 加载数据
    print("\n[1] 加载数据...")
    storm_samples = load_tyc_storms(
        csv_path=args.track_csv,
        era5_base_dir=args.data_root,
    )
    storm_samples = filter_short_storms(storm_samples, train_cfg.min_typhoon_duration_hours)
    storm_samples = filter_out_of_range_storms(storm_samples)

    _, _, test_storms = split_storms_by_id(
        storm_samples, train_cfg.train_ratio, train_cfg.val_ratio, seed=42
    )
    print(f"  测试台风数: {len(test_storms)}")

    # 2. 构建数据集（全部测试集）
    if args.use_diffusion:
        from pathlib import Path
        from finetune_train import DiffusionERA5Dataset, generate_diffusion_era5_cache

        cache_path = Path(args.cache_dir) / "era5_cache.npz"
        if cache_path.exists():
            print(f"  加载扩散ERA5缓存: {cache_path}")
            loaded = np.load(cache_path, allow_pickle=True)
            diffusion_cache = {k: loaded[k] for k in loaded.files}
        else:
            print(f"  缓存不存在，正在生成扩散ERA5...")
            diffusion_cache = generate_diffusion_era5_cache(
                storm_samples=test_storms,
                diffusion_code=args.diffusion_code,
                diffusion_ckpt=args.diffusion_ckpt,
                norm_stats_path=args.norm_stats,
                data_root=args.data_root,
                device=device,
                ddim_steps=args.ddim_steps,
            )
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(cache_path, **diffusion_cache)
            print(f"  缓存已保存: {cache_path}")

        test_ds = DiffusionERA5Dataset(test_storms, diffusion_cache, stride=model_cfg.t_future)
    else:
        test_ds = LT3PDataset(test_storms, stride=model_cfg.t_future)

    print(f"  测试样本总数: {len(test_ds)}")
    loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    # 3. 加载模型
    print("\n[2] 加载模型...")
    sample = test_ds[0]
    era5_channels = sample['future_era5'].shape[1]

    model = LT3PModel(
        coord_dim=model_cfg.coord_dim,
        output_dim=model_cfg.output_dim,
        era5_channels=era5_channels,
        t_history=model_cfg.t_history,
        t_future=model_cfg.t_future,
        d_model=model_cfg.transformer_dim,
        n_heads=model_cfg.transformer_heads,
        n_layers=model_cfg.transformer_layers,
        ff_dim=model_cfg.transformer_ff_dim,
        dropout=model_cfg.dropout,
    )

    ckpt = torch.load(args.ckpt, map_location='cpu', weights_only=False)
    if 'ema_model_state_dict' in ckpt:
        model.load_state_dict(ckpt['ema_model_state_dict'])
        print(f"  已加载 EMA 权重 (epoch {ckpt.get('epoch', '?')})")
    else:
        model.load_state_dict(ckpt['model_state_dict'])
        print(f"  已加载权重 (epoch {ckpt.get('epoch', '?')})")

    model.to(device)
    model.eval()

    # 4. 全量推理
    print("\n[3] 全量推理中...")
    lat_range = data_cfg.lat_range
    lon_range = data_cfg.lon_range

    # 按台风ID收集误差
    storm_errors = defaultdict(list)  # {storm_id: [mean_err_sample1, ...]}
    storm_errors_by_hour = defaultdict(list)  # {storm_id: [(24h, 48h, 72h), ...]}
    storm_dist_errors = defaultdict(list)  # {storm_id: [dist_err_array, ...]}  逐时间步
    all_errors = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="推理"):
            history_coords = batch['history_coords'].to(device)
            future_era5 = batch['future_era5'].to(device)
            target_lat = batch['target_lat_raw'].numpy()
            target_lon = batch['target_lon_raw'].numpy()
            storm_ids = batch['storm_id']

            outputs = model.predict(history_coords, future_era5)
            pred_coords = outputs['predicted_coords'].cpu().numpy()

            pred_lat = pred_coords[:, :, 0] * (lat_range[1] - lat_range[0]) + lat_range[0]
            pred_lon = pred_coords[:, :, 1] * (lon_range[1] - lon_range[0]) + lon_range[0]

            lat_err = (pred_lat - target_lat) * 111.0
            lon_err = (pred_lon - target_lon) * 111.0 * np.cos(np.radians(target_lat))
            dist_err = np.sqrt(lat_err ** 2 + lon_err ** 2)

            all_errors.append(dist_err)

            for i in range(len(storm_ids)):
                sid = storm_ids[i]
                sample_mean = dist_err[i].mean()
                storm_errors[sid].append(sample_mean)
                storm_dist_errors[sid].append(dist_err[i])  # (24,) 逐时间步

                # 24h, 48h, 72h 关键节点
                h24 = dist_err[i, 7] if dist_err.shape[1] > 7 else float('nan')
                h48 = dist_err[i, 15] if dist_err.shape[1] > 15 else float('nan')
                h72 = dist_err[i, 23] if dist_err.shape[1] > 23 else float('nan')
                storm_errors_by_hour[sid].append((h24, h48, h72))

    all_errors = np.concatenate(all_errors, axis=0)

    # 5. 按台风汇总
    storm_summary = []
    for sid, errs in storm_errors.items():
        mean_err = np.mean(errs)
        hours = storm_errors_by_hour[sid]
        h24_mean = np.nanmean([h[0] for h in hours])
        h48_mean = np.nanmean([h[1] for h in hours])
        h72_mean = np.nanmean([h[2] for h in hours])
        n_samples = len(errs)
        storm_summary.append({
            'storm_id': sid,
            'mean_err': mean_err,
            'h24': h24_mean,
            'h48': h48_mean,
            'h72': h72_mean,
            'n_samples': n_samples,
        })

    # 按平均误差排序（从小到大 = 最精确的排前面）
    storm_summary.sort(key=lambda x: x['mean_err'])

    # 6. 输出结果
    print(f"\n{'='*80}")
    print(f"  全量评估结果 ({len(test_storms)} 个台风, {len(all_errors)} 个样本, {mode})")
    print(f"  Checkpoint: {args.ckpt}")
    print(f"{'='*80}")

    print(f"\n  全局统计:")
    print(f"    总体平均误差: {all_errors.mean():.2f} km")
    for h in [24, 48, 72]:
        idx = h // 3 - 1
        if idx < all_errors.shape[1]:
            print(f"    +{h}h 平均误差: {all_errors[:, idx].mean():.2f} km")

    print(f"\n{'='*80}")
    print(f"  Top-{args.top_n} 最精确的台风")
    print(f"{'='*80}")
    print(f"  {'排名':<6} {'台风ID':<20} {'平均误差(km)':<14} {'+24h(km)':<10} {'+48h(km)':<10} {'+72h(km)':<10} {'样本数':<6}")
    print(f"  {'-'*76}")

    for rank, s in enumerate(storm_summary[:args.top_n], 1):
        print(f"  {rank:<6} {s['storm_id']:<20} {s['mean_err']:<14.2f} {s['h24']:<10.2f} {s['h48']:<10.2f} {s['h72']:<10.2f} {s['n_samples']:<6}")

    # 也输出最差的5个，供参考
    print(f"\n  最差 5 个台风 (供参考):")
    print(f"  {'排名':<6} {'台风ID':<20} {'平均误差(km)':<14} {'+24h(km)':<10} {'+48h(km)':<10} {'+72h(km)':<10} {'样本数':<6}")
    print(f"  {'-'*76}")
    for rank, s in enumerate(storm_summary[-5:], len(storm_summary) - 4):
        print(f"  {rank:<6} {s['storm_id']:<20} {s['mean_err']:<14.2f} {s['h24']:<10.2f} {s['h48']:<10.2f} {s['h72']:<10.2f} {s['n_samples']:<6}")

    # Top-N 台风的聚合逐时间步误差
    top_ids = set(s['storm_id'] for s in storm_summary[:args.top_n])
    top_errors = []
    for sid in top_ids:
        for err_arr in storm_dist_errors[sid]:
            top_errors.append(err_arr)
    top_errors = np.stack(top_errors, axis=0)  # (N_samples, 24)

    print(f"\n{'='*80}")
    print(f"  Top-{args.top_n} 台风聚合误差 ({len(top_errors)} 个样本)")
    print(f"{'='*80}")
    print(f"\n  总体平均误差: {top_errors.mean():.2f} km")
    print(f"\n  逐时间步误差:")
    print(f"  {'时间':<8} {'平均(km)':<12} {'标准差(km)':<12}")
    print(f"  {'-'*32}")
    for t in range(top_errors.shape[1]):
        hours = (t + 1) * 3
        m = top_errors[:, t].mean()
        s = top_errors[:, t].std()
        print(f"  +{hours:>3d}h    {m:<12.2f} {s:<12.2f}")

    print(f"\n  关键节点:")
    for h in [24, 48, 72]:
        idx = h // 3 - 1
        if idx < top_errors.shape[1]:
            print(f"    +{h}h: {top_errors[:, idx].mean():.2f} km")

    print()


if __name__ == "__main__":
    main()
