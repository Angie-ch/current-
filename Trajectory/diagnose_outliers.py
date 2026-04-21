"""
诊断 LT3P 测试集中的离群样本
找出误差极大的样本，分析其原因
"""
import torch
import numpy as np
from collections import Counter
from tqdm import tqdm

from config import model_cfg, data_cfg
from model import LT3PModel
from dataset import LT3PDataset, split_storms_by_id, filter_short_storms, filter_out_of_range_storms, normalize_coords
from data_processing import load_tyc_storms


def main():
    print("=" * 70)
    print("LT3P 离群样本诊断")
    print("=" * 70)

    # === 1. 加载数据 ===
    print("\n[1] Loading data...")
    storm_samples = load_tyc_storms(
        csv_path=data_cfg.csv_path,
        era5_base_dir=data_cfg.era5_dir
    )
    storm_samples = filter_short_storms(storm_samples, 120, 3)
    storm_samples = filter_out_of_range_storms(storm_samples)
    train_s, val_s, test_s = split_storms_by_id(storm_samples, 0.7, 0.15, seed=42)
    test_ds = LT3PDataset(test_s, stride=model_cfg.t_future)
    print(f"Test samples: {len(test_ds)}")

    # === 2. 先检查数据本身的问题（不需要模型）===
    print("\n[2] 检查数据坐标范围...")
    lat_range = data_cfg.lat_range
    lon_range = data_cfg.lon_range

    out_of_range_samples = []
    era5_zero_samples = []

    for idx in range(len(test_ds)):
        storm_idx, start_idx = test_ds.samples_index[idx]
        sample = test_ds.storm_samples[storm_idx]

        history_start = start_idx
        history_end = start_idx + test_ds.t_history
        future_start = history_end
        future_end = history_end + test_ds.t_future

        # 检查历史坐标范围
        h_lat = sample.track_lat[history_start:history_end]
        h_lon = sample.track_lon[history_start:history_end]
        f_lat = sample.track_lat[future_start:future_end]
        f_lon = sample.track_lon[future_start:future_end]

        all_lat = np.concatenate([h_lat, f_lat])
        all_lon = np.concatenate([h_lon, f_lon])

        # 检查是否越界
        lat_oob = (all_lat < lat_range[0]).any() or (all_lat > lat_range[1]).any()
        lon_oob = (all_lon < lon_range[0]).any() or (all_lon > lon_range[1]).any()

        if lat_oob or lon_oob:
            out_of_range_samples.append({
                'idx': idx,
                'storm_id': sample.storm_id,
                'lat_min': all_lat.min(),
                'lat_max': all_lat.max(),
                'lon_min': all_lon.min(),
                'lon_max': all_lon.max(),
                'lat_oob': lat_oob,
                'lon_oob': lon_oob,
            })

        # 检查 ERA5 是否全零
        era5 = test_ds._get_era5_video(sample, future_start, future_end)
        if np.abs(era5).sum() == 0:
            era5_zero_samples.append({
                'idx': idx,
                'storm_id': sample.storm_id,
            })

    print(f"\n  坐标越界样本数: {len(out_of_range_samples)} / {len(test_ds)}")
    if out_of_range_samples:
        print(f"  {'idx':>5} | {'storm_id':>15} | {'lat范围':>18} | {'lon范围':>18} | 越界类型")
        print(f"  {'-'*5}-+-{'-'*15}-+-{'-'*18}-+-{'-'*18}-+-{'-'*10}")
        for s in out_of_range_samples[:20]:
            oob_type = []
            if s['lat_oob']:
                oob_type.append('LAT')
            if s['lon_oob']:
                oob_type.append('LON')
            print(f"  {s['idx']:5d} | {s['storm_id']:>15} | "
                  f"[{s['lat_min']:6.1f}, {s['lat_max']:5.1f}] | "
                  f"[{s['lon_min']:6.1f}, {s['lon_max']:5.1f}] | {'+'.join(oob_type)}")
        if len(out_of_range_samples) > 20:
            print(f"  ... 还有 {len(out_of_range_samples) - 20} 个")

    # 统计越界台风
    oob_storms = Counter(s['storm_id'] for s in out_of_range_samples)
    if oob_storms:
        print(f"\n  越界台风统计 (共 {len(oob_storms)} 个台风):")
        for sid, count in oob_storms.most_common(10):
            print(f"    {sid}: {count} 个样本")

    print(f"\n  ERA5全零样本数: {len(era5_zero_samples)} / {len(test_ds)}")

    # === 3. 加载模型做推理诊断 ===
    print("\n[3] 加载模型并预测...")
    checkpoint_path = 'checkpoints/best.pt'
    checkpoint = torch.load(checkpoint_path, map_location='cuda', weights_only=False)

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
    ).cuda()
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # 逐样本预测
    errors_per_sample = []
    pred_ranges = []

    with torch.no_grad():
        for idx in tqdm(range(len(test_ds)), desc="诊断预测"):
            batch = test_ds[idx]
            history_coords = batch['history_coords'].unsqueeze(0).cuda()
            future_era5 = batch['future_era5'].unsqueeze(0).cuda()
            target_lat = batch['target_lat_raw'].numpy()
            target_lon = batch['target_lon_raw'].numpy()
            storm_id = batch['storm_id']

            outputs = model.predict(history_coords, future_era5)
            pred_coords = outputs['predicted_coords'].cpu().numpy()[0]

            # 预测的归一化坐标范围
            pred_lat_norm = pred_coords[:, 0]
            pred_lon_norm = pred_coords[:, 1]

            # 反归一化
            pred_lat = pred_lat_norm * (lat_range[1] - lat_range[0]) + lat_range[0]
            pred_lon = pred_lon_norm * (lon_range[1] - lon_range[0]) + lon_range[0]

            # 计算误差
            lat_err = (pred_lat - target_lat) * 111
            lon_err = (pred_lon - target_lon) * 111 * np.cos(np.radians(target_lat))
            dist_err = np.sqrt(lat_err ** 2 + lon_err ** 2)

            errors_per_sample.append({
                'idx': idx,
                'storm_id': storm_id,
                'mean_error_km': dist_err.mean(),
                'max_error_km': dist_err.max(),
                'err_3h': dist_err[0],
                'err_72h': dist_err[-1],
                'pred_lat_range': (pred_lat.min(), pred_lat.max()),
                'pred_lon_range': (pred_lon.min(), pred_lon.max()),
                'pred_lat_norm_range': (pred_lat_norm.min(), pred_lat_norm.max()),
                'pred_lon_norm_range': (pred_lon_norm.min(), pred_lon_norm.max()),
                'true_lat_range': (target_lat.min(), target_lat.max()),
                'true_lon_range': (target_lon.min(), target_lon.max()),
                'is_oob': any(s['idx'] == idx for s in out_of_range_samples),
                'is_era5_zero': any(s['idx'] == idx for s in era5_zero_samples),
            })

    # === 4. 分析离群样本 ===
    errors_sorted = sorted(errors_per_sample, key=lambda x: x['mean_error_km'], reverse=True)
    threshold = np.percentile([e['mean_error_km'] for e in errors_per_sample], 95)

    outliers = [e for e in errors_sorted if e['mean_error_km'] > threshold]

    print(f"\n{'=' * 70}")
    print(f"[4] 离群样本详细分析 (阈值: {threshold:.1f} km, 共 {len(outliers)} 个)")
    print(f"{'=' * 70}")

    # 分类统计
    n_oob_in_outliers = sum(1 for e in outliers if e['is_oob'])
    n_era5_zero_in_outliers = sum(1 for e in outliers if e['is_era5_zero'])
    n_pred_exploded = sum(1 for e in outliers
                          if e['pred_lat_norm_range'][0] < -0.5 or e['pred_lat_norm_range'][1] > 1.5
                          or e['pred_lon_norm_range'][0] < -0.5 or e['pred_lon_norm_range'][1] > 1.5)

    print(f"\n  离群原因统计:")
    print(f"    坐标越界 (数据本身超出 lat[5,45] lon[100,180]): {n_oob_in_outliers}/{len(outliers)}")
    print(f"    ERA5 全零:                                       {n_era5_zero_in_outliers}/{len(outliers)}")
    print(f"    预测值爆炸 (归一化值超出 [-0.5, 1.5]):           {n_pred_exploded}/{len(outliers)}")

    print(f"\n  Top 15 离群样本:")
    print(f"  {'排名':>4} | {'storm_id':>12} | {'均误差km':>9} | {'3h误差':>8} | {'72h误差':>9} | "
          f"{'预测lat范围':>18} | {'真实lat范围':>18} | {'越界':>4} | {'ERA5零':>6}")
    print(f"  {'-' * 120}")

    for rank, e in enumerate(outliers[:15], 1):
        print(f"  {rank:4d} | {e['storm_id']:>12} | {e['mean_error_km']:9.1f} | "
              f"{e['err_3h']:8.1f} | {e['err_72h']:9.1f} | "
              f"[{e['pred_lat_range'][0]:6.1f},{e['pred_lat_range'][1]:5.1f}] | "
              f"[{e['true_lat_range'][0]:6.1f},{e['true_lat_range'][1]:5.1f}] | "
              f"{'是' if e['is_oob'] else '否':>4} | {'是' if e['is_era5_zero'] else '否':>6}")

    # === 5. 总结 ===
    print(f"\n{'=' * 70}")
    print("[5] 诊断总结")
    print(f"{'=' * 70}")

    # 检查非离群和离群的越界比例
    non_outliers = [e for e in errors_per_sample if e['mean_error_km'] <= threshold]
    n_oob_normal = sum(1 for e in non_outliers if e['is_oob'])

    print(f"\n  正常样本中坐标越界比例: {n_oob_normal}/{len(non_outliers)} ({100*n_oob_normal/max(len(non_outliers),1):.1f}%)")
    print(f"  离群样本中坐标越界比例: {n_oob_in_outliers}/{len(outliers)} ({100*n_oob_in_outliers/max(len(outliers),1):.1f}%)")

    if n_oob_in_outliers / max(len(outliers), 1) > 0.5:
        print(f"\n  ⚠️ 结论: 大部分离群是因为坐标越界!")
        print(f"     建议: 扩大归一化范围 lat_range 和 lon_range，或在推理时 clamp 输出")
    elif n_era5_zero_in_outliers / max(len(outliers), 1) > 0.3:
        print(f"\n  ⚠️ 结论: 大量离群因为 ERA5 数据缺失（全零）!")
        print(f"     建议: 在数据加载时过滤掉 ERA5 缺失的样本")
    elif n_pred_exploded / max(len(outliers), 1) > 0.5:
        print(f"\n  ⚠️ 结论: 模型预测值发散（归一化坐标超出合理范围）!")
        print(f"     建议: 在模型输出层添加 sigmoid 或 clamp 限制输出范围")
    else:
        print(f"\n  需要进一步分析具体样本")


if __name__ == '__main__':
    main()
