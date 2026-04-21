"""
z 场基线对比测试: 扩散模型 vs 持续性 vs 线性外推
不需要 GPU，不需要模型推理，纯数据对比，几分钟出结果

用法:
  python z_baseline_compare.py \
      --data_root ~/autodl-tmp/Typhoon_data_final \
      --preprocess_dir /root/autodl-tmp/preprocessed_npy_9ch \
      --diffusion_cache eval_results/per_typhoon_rmse.npz
"""
import os
import argparse
import numpy as np
import torch
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from configs import get_config
from data.dataset import ERA5TyphoonDataset, split_typhoon_ids


def denorm(data, mean, std):
    std = np.where(std < 1e-8, 1.0, std)
    return data * std.reshape(-1, 1, 1) + mean.reshape(-1, 1, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--preprocess_dir", type=str, default=None)
    parser.add_argument("--diffusion_cache", type=str, default=None,
                        help="evaluate_multi.py 保存的 per_typhoon_rmse.npz")
    parser.add_argument("--num_typhoons", type=int, default=9999)
    parser.add_argument("--ar_steps", type=int, default=24)
    args = parser.parse_args()

    data_cfg, _, _, _ = get_config(data_root=args.data_root)
    C = data_cfg.num_channels
    z_idx = [6, 7, 8]  # z_850, z_500, z_250
    z_names = ['z_850', 'z_500', 'z_250']
    ar_steps = args.ar_steps

    # 加载归一化统计
    norm_path = "norm_stats.pt"
    stats = torch.load(norm_path, weights_only=True)
    norm_mean = stats["mean"].numpy()[:C]
    norm_std = stats["std"].numpy()[:C]

    print(f"z 通道归一化统计:")
    for i, name in enumerate(z_names):
        print(f"  {name}: mean={norm_mean[z_idx[i]]:.2f}, std={norm_std[z_idx[i]]:.2f}")

    # 构建测试集
    _, _, test_ids = split_typhoon_ids(data_cfg.data_root, seed=42)
    n_typhoons = min(args.num_typhoons, len(test_ids))

    test_dataset = ERA5TyphoonDataset(
        typhoon_ids=test_ids[:n_typhoons],
        data_root=data_cfg.data_root,
        pl_vars=data_cfg.pressure_level_vars,
        sfc_vars=data_cfg.surface_vars,
        pressure_levels=data_cfg.pressure_levels,
        history_steps=data_cfg.history_steps,
        forecast_steps=data_cfg.forecast_steps,
        norm_mean=norm_mean,
        norm_std=norm_std,
        preprocessed_dir=args.preprocess_dir,
    )

    # 按台风分组
    tid_to_indices = {}
    for i in range(len(test_dataset)):
        sample = test_dataset[i]
        tid = sample["typhoon_id"]
        if tid not in tid_to_indices:
            tid_to_indices[tid] = []
        tid_to_indices[tid].append(i)

    print(f"\n测试台风数: {len(tid_to_indices)}")

    # 逐台风计算基线 RMSE
    all_persistence_rmse = []  # (N, T, 3) z channels only
    all_linear_rmse = []
    all_tids = []

    for tid, indices in tid_to_indices.items():
        start_idx = indices[len(indices) // 2]
        sample = test_dataset[start_idx]
        cond = sample["condition"]  # (C*T_hist, H, W)

        T_hist = data_cfg.history_steps
        cond_5d = cond.reshape(T_hist, C, 40, 40).numpy()

        # 条件窗口最后一帧的 z (归一化空间)
        z_last_norm = cond_5d[-1, z_idx]  # (3, H, W)
        # 条件窗口的 z 变化趋势 (最后一帧 - 倒数第二帧)
        z_trend_norm = cond_5d[-1, z_idx] - cond_5d[-2, z_idx]  # (3, H, W)

        # 反归一化
        z_last_phys = denorm(z_last_norm, norm_mean[z_idx], norm_std[z_idx])
        z_trend_phys = z_trend_norm * norm_std[z_idx].reshape(-1, 1, 1)

        persist_rmse = np.full((ar_steps, 3), np.nan)
        linear_rmse = np.full((ar_steps, 3), np.nan)

        for t in range(ar_steps):
            gt_idx = start_idx + t
            if gt_idx >= len(test_dataset):
                break
            gt_sample = test_dataset[gt_idx]
            if gt_sample["typhoon_id"] != tid:
                break

            gt_z_norm = gt_sample["target"][z_idx].numpy()  # (3, H, W)
            gt_z_phys = denorm(gt_z_norm, norm_mean[z_idx], norm_std[z_idx])

            # 持续性基线: z 不变
            for v in range(3):
                persist_rmse[t, v] = np.sqrt(np.mean((z_last_phys[v] - gt_z_phys[v]) ** 2))

            # 线性外推基线: z = z_last + (t+1) * trend
            z_linear_phys = z_last_phys + (t + 1) * z_trend_phys
            for v in range(3):
                linear_rmse[t, v] = np.sqrt(np.mean((z_linear_phys[v] - gt_z_phys[v]) ** 2))

        all_persistence_rmse.append(persist_rmse)
        all_linear_rmse.append(linear_rmse)
        all_tids.append(tid)

    persist_stacked = np.stack(all_persistence_rmse, axis=0)  # (N, T, 3)
    linear_stacked = np.stack(all_linear_rmse, axis=0)

    # 加载扩散模型的 RMSE (如果有)
    diff_stacked = None
    if args.diffusion_cache and os.path.exists(args.diffusion_cache):
        cache = np.load(args.diffusion_cache, allow_pickle=True)
        diff_full = cache["stacked"]  # (N, T, C)
        diff_tids = list(cache["tids"])
        # 提取 z 通道, 按 tid 对齐
        diff_z = {}
        for i, tid in enumerate(diff_tids):
            diff_z[tid] = diff_full[i, :, z_idx]  # (T, 3)

        diff_stacked = np.full_like(persist_stacked, np.nan)
        for i, tid in enumerate(all_tids):
            if tid in diff_z:
                z_data = diff_z[tid]
                if z_data.shape == (3, ar_steps):
                    z_data = z_data.T
                diff_stacked[i] = z_data

    # ---- 打印对比表格 ----
    persist_mean = np.nanmean(persist_stacked, axis=0)
    linear_mean = np.nanmean(linear_stacked, axis=0)
    persist_median = np.nanmedian(persist_stacked, axis=0)
    linear_median = np.nanmedian(linear_stacked, axis=0)

    print(f"\n{'='*100}")
    print(f"  [Mean RMSE 对比] {len(all_tids)} typhoons — z 通道")
    print(f"{'='*100}")

    if diff_stacked is not None:
        diff_mean = np.nanmean(diff_stacked, axis=0)
        diff_median = np.nanmedian(diff_stacked, axis=0)
        print(f"{'时效':>6}  {'--- z_500 ---':>40}")
        print(f"{'':>6}  {'扩散模型':>12}  {'持续性':>10}  {'线性外推':>10}  {'扩散胜?':>8}")
        print(f"  {'-'*55}")
        for t in range(ar_steps):
            d = diff_mean[t, 1]
            p = persist_mean[t, 1]
            l = linear_mean[t, 1]
            win = "是" if d < min(p, l) else "否"
            print(f"  +{(t+1)*3:>3}h  {d:>12.1f}  {p:>10.1f}  {l:>10.1f}  {win:>8}")

        print(f"\n  {'平均':>4}  {np.nanmean(diff_mean[:,1]):>12.1f}  "
              f"{np.nanmean(persist_mean[:,1]):>10.1f}  {np.nanmean(linear_mean[:,1]):>10.1f}")
    else:
        print(f"{'时效':>6}  {'z_500持续性':>12}  {'z_500线性':>10}")
        for t in range(ar_steps):
            print(f"  +{(t+1)*3:>3}h  {persist_mean[t,1]:>12.1f}  {linear_mean[t,1]:>10.1f}")

    # ---- 逐台风对比: 扩散 vs 持续性 ----
    if diff_stacked is not None:
        print(f"\n{'='*100}")
        print(f"  [逐台风] z_500 平均 RMSE: 扩散模型 vs 持续性基线")
        print(f"{'='*100}")

        diff_wins = 0
        persist_wins = 0
        results = []

        for i, tid in enumerate(all_tids):
            d_avg = np.nanmean(diff_stacked[i, :, 1])
            p_avg = np.nanmean(persist_stacked[i, :, 1])
            if np.isnan(d_avg) or np.isnan(p_avg):
                continue
            winner = "扩散" if d_avg < p_avg else "持续性"
            if d_avg < p_avg:
                diff_wins += 1
            else:
                persist_wins += 1
            results.append((tid, d_avg, p_avg, winner))

        # 按扩散模型 RMSE 排序
        results.sort(key=lambda x: x[1], reverse=True)

        print(f"\n  扩散模型胜: {diff_wins} 个台风")
        print(f"  持续性胜:   {persist_wins} 个台风")
        print(f"  持续性胜出比例: {persist_wins/(diff_wins+persist_wins)*100:.1f}%")

        # 打印最差的 20 个
        print(f"\n  扩散模型最差的 20 个台风:")
        print(f"  {'台风ID':<20}  {'扩散z500':>10}  {'持续性z500':>12}  {'差距':>10}  {'胜者':<6}")
        print(f"  {'-'*65}")
        for tid, d, p, w in results[:20]:
            print(f"  {tid:<20}  {d:>10.1f}  {p:>12.1f}  {d-p:>+10.1f}  {w}")

        # 打印最好的 10 个
        print(f"\n  扩散模型最好的 10 个台风:")
        for tid, d, p, w in results[-10:]:
            print(f"  {tid:<20}  {d:>10.1f}  {p:>12.1f}  {d-p:>+10.1f}  {w}")

    print(f"\n对比完成!")


if __name__ == "__main__":
    main()
