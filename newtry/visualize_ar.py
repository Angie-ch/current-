"""
自回归预测可视化 — 对比真值 vs 预测（逐时效）

从数据集加载连续时间步的真值，与自回归预测逐步对比
"""
import os
import sys
import argparse
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ---- 需要从项目导入数据集 ----
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from configs import DataConfig
from data.dataset import ERA5TyphoonDataset


VAR_NAMES = [
    'u_850', 'u_500', 'u_250',
    'v_850', 'v_500', 'v_250',
    'z_850', 'z_500', 'z_250',
]
VAR_UNITS = {
    'u_850': 'm/s', 'u_500': 'm/s', 'u_250': 'm/s',
    'v_850': 'm/s', 'v_500': 'm/s', 'v_250': 'm/s',
    'z_850': 'm²/s²', 'z_500': 'm²/s²', 'z_250': 'm²/s²',
}


def denormalize(data_norm, mean, std):
    """反归一化: (C, H, W) numpy"""
    std = np.where(std < 1e-8, 1.0, std)
    return data_norm * std.reshape(-1, 1, 1) + mean.reshape(-1, 1, 1)


def plot_ar_comparison(preds, gts, typhoon_id, time_steps, save_path, var_idx=0):
    """
    绘制某个变量在多个时效的 GT vs Pred 对比

    preds: list of (C, H, W) numpy, 预测
    gts:   list of (C, H, W) numpy, 真值
    time_steps: list of int, 对应的预测时效 (小时)
    var_idx: 要画的变量索引
    """
    n_steps = len(time_steps)
    fig, axes = plt.subplots(3, n_steps, figsize=(4 * n_steps, 10))
    fig.suptitle(f'Typhoon {typhoon_id} — {VAR_NAMES[var_idx]} autoregressive forecast',
                 fontsize=14, fontweight='bold')

    for col, (t_hr, pred, gt) in enumerate(zip(time_steps, preds, gts)):
        p = pred[var_idx]
        g = gt[var_idx]
        diff = p - g
        rmse = np.sqrt(np.mean(diff ** 2))

        vmin = min(g.min(), p.min())
        vmax = max(g.max(), p.max())

        # GT
        ax0 = axes[0, col]
        im0 = ax0.imshow(g, cmap='RdYlBu_r', vmin=vmin, vmax=vmax)
        ax0.set_title(f'GT +{t_hr}h', fontsize=10)
        ax0.set_xticks([]); ax0.set_yticks([])
        plt.colorbar(im0, ax=ax0, fraction=0.046)

        # Pred
        ax1 = axes[1, col]
        im1 = ax1.imshow(p, cmap='RdYlBu_r', vmin=vmin, vmax=vmax)
        ax1.set_title(f'Pred +{t_hr}h', fontsize=10)
        ax1.set_xticks([]); ax1.set_yticks([])
        plt.colorbar(im1, ax=ax1, fraction=0.046)

        # Error
        ax2 = axes[2, col]
        abs_max = max(abs(diff.min()), abs(diff.max()), 1e-8)
        im2 = ax2.imshow(diff, cmap='bwr', vmin=-abs_max, vmax=abs_max)
        ax2.set_title(f'Err RMSE={rmse:.1f}', fontsize=10)
        ax2.set_xticks([]); ax2.set_yticks([])
        plt.colorbar(im2, ax=ax2, fraction=0.046)

    # 行标签
    axes[0, 0].set_ylabel('Ground Truth', fontsize=11)
    axes[1, 0].set_ylabel('Prediction', fontsize=11)
    axes[2, 0].set_ylabel('Error', fontsize=11)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_rmse_vs_leadtime(all_rmse, time_hours, typhoon_id, save_path):
    """
    绘制 RMSE 随时效增长的曲线

    all_rmse: (T, C) numpy
    time_hours: list of int
    """
    fig, ax = plt.subplots(figsize=(10, 5))

    for v_idx, vname in enumerate(VAR_NAMES):
        ax.plot(time_hours, all_rmse[:, v_idx], marker='o', markersize=3,
                label=f'{vname}', linewidth=1.5)

    ax.set_xlabel('Lead time (hours)', fontsize=12)
    ax.set_ylabel('RMSE (physical units)', fontsize=12)
    ax.set_title(f'Typhoon {typhoon_id} — RMSE vs Lead Time', fontsize=14)
    ax.legend(fontsize=9, ncol=2)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(time_hours[::2])  # 每隔一个时效标注

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Autoregressive prediction visualization")
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--sample_id", type=int, default=0)
    parser.add_argument("--save_dir", type=str, default="visualizations")
    parser.add_argument("--data_root", type=str, default=None,
                        help="ERA5数据根目录 (默认使用 DataConfig 中的路径)")
    parser.add_argument("--work_dir", type=str, default=".",
                        help="norm_stats.pt 所在目录")
    parser.add_argument("--preprocess_dir", type=str, default=None,
                        help="预处理NPY目录 (推荐，速度快50-100倍)")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    # ---- 加载自回归预测 ----
    pt_path = os.path.join(args.output_dir, f"ar_pred_{args.sample_id}.pt")
    if not os.path.exists(pt_path):
        print(f"File not found: {pt_path}")
        sys.exit(1)

    data = torch.load(pt_path, map_location="cpu", weights_only=False)
    preds = data["predictions"]  # (1, T, C, H, W) — 归一化空间
    tid = data["typhoon_id"]
    T_steps = preds.shape[1]     # 24 步
    C = preds.shape[2]           # 9 通道

    print(f"Typhoon: {tid}")
    print(f"Predictions shape: {preds.shape}")
    print(f"Autoregressive steps: {T_steps} -> {T_steps * 3}h")

    # ---- 加载归一化统计 ----
    norm_stats_path = os.path.join(args.work_dir, "norm_stats.pt")
    assert os.path.exists(norm_stats_path), f"找不到: {norm_stats_path}"
    stats = torch.load(norm_stats_path, weights_only=True)
    norm_mean = stats["mean"].numpy()[:C]   # 只取前 C 个通道
    norm_std = stats["std"].numpy()[:C]

    # ---- 构建数据集，获取真值 ----
    data_cfg = DataConfig()
    if args.data_root:
        data_cfg.data_root = args.data_root
    data_cfg.__post_init__()

    print(f"\n加载真值数据 (台风: {tid})...")
    print(f"  data_root: {data_cfg.data_root}")

    # 直接加载目标台风（而非盲目取 test_ids[:10]）
    # 先确认该台风目录存在
    tid_dir = os.path.join(data_cfg.data_root, tid)
    if not os.path.isdir(tid_dir) and args.preprocess_dir is None:
        print(f"  警告: 台风目录不存在: {tid_dir}")
        print(f"  请检查 --data_root 是否正确，或使用 --preprocess_dir")
        gt_targets = []
    else:
        try:
            test_dataset = ERA5TyphoonDataset(
                typhoon_ids=[tid],  # 只加载目标台风
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
            print(f"  数据集样本数: {len(test_dataset)}")

            if len(test_dataset) == 0:
                print(f"  警告: 台风 {tid} 无可用样本")
                gt_targets = []
            else:
                # 从 sample_id 对应的位置开始收集连续真值
                start = min(args.sample_id, len(test_dataset) - 1)
                gt_targets = []
                for idx in range(start, min(start + T_steps, len(test_dataset))):
                    sample = test_dataset[idx]
                    if sample["typhoon_id"] == tid:
                        gt_targets.append(sample["target"])  # (C, H, W) 归一化
                    else:
                        break
        except Exception as e:
            print(f"  加载数据集失败: {e}")
            gt_targets = []

    n_gt = len(gt_targets)
    print(f"  找到 {n_gt}/{T_steps} 步真值")

    # ---- 反归一化 ----
    n_compare = min(n_gt, T_steps)

    # 预测始终反归一化（无论有无真值）
    all_pred_phys = []
    for t in range(T_steps):
        pred_norm = preds[0, t].numpy()
        all_pred_phys.append(denormalize(pred_norm, norm_mean, norm_std))

    pred_phys_list = []
    gt_phys_list = []
    all_rmse = np.zeros((n_compare, len(VAR_NAMES)))

    for t in range(n_compare):
        pred_phys = all_pred_phys[t]
        gt_norm = gt_targets[t].numpy()
        gt_phys = denormalize(gt_norm, norm_mean, norm_std)

        pred_phys_list.append(pred_phys)
        gt_phys_list.append(gt_phys)

        for v in range(len(VAR_NAMES)):
            all_rmse[t, v] = np.sqrt(np.mean((pred_phys[v] - gt_phys[v]) ** 2))

    # ---- 无真值时：仅绘制预测演变图 ----
    if n_compare == 0:
        print("\n  无真值，绘制预测场演变图...")
        # 选关键时效: +3h, +12h, +24h, +48h, +72h
        key_indices = [0, 3, 7, 15, min(23, T_steps - 1)]
        key_indices = sorted(set(i for i in key_indices if i < T_steps))
        key_hours = [(i + 1) * 3 for i in key_indices]

        for v_idx in range(len(VAR_NAMES)):
            n_cols = len(key_indices)
            fig, axes = plt.subplots(1, n_cols, figsize=(4 * n_cols, 4))
            if n_cols == 1:
                axes = [axes]
            fig.suptitle(f'Typhoon {tid} — {VAR_NAMES[v_idx]} prediction evolution',
                         fontsize=14, fontweight='bold')

            for col, (ki, kh) in enumerate(zip(key_indices, key_hours)):
                field = all_pred_phys[ki][v_idx]
                im = axes[col].imshow(field, cmap='RdYlBu_r')
                axes[col].set_title(f'+{kh}h', fontsize=11)
                axes[col].set_xticks([]); axes[col].set_yticks([])
                plt.colorbar(im, ax=axes[col], fraction=0.046)

            plt.tight_layout()
            save_p = os.path.join(args.save_dir,
                                  f"ar_pred_only_{VAR_NAMES[v_idx]}_{args.sample_id}.png")
            plt.savefig(save_p, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"  Saved: {save_p}")

        print("\n可视化完成! (仅预测，无GT对比)")
        return

    # ---- 有真值：打印 RMSE 表格 ----
    time_hours = [(t + 1) * 3 for t in range(n_compare)]

    print(f"\n{'时效':<8}", end="")
    for vn in VAR_NAMES:
        print(f" {vn:>8}", end="")
    print()
    print("=" * (8 + 9 * len(VAR_NAMES)))

    def fmt_val(val, v_idx):
        return f" {val:>8.2f}"

    for t in range(n_compare):
        print(f"+{time_hours[t]:>3}h    ", end="")
        for v in range(len(VAR_NAMES)):
            print(fmt_val(all_rmse[t, v], v), end="")
        print()

    print(f"\n{'平均':>7} ", end="")
    for v in range(len(VAR_NAMES)):
        print(fmt_val(all_rmse[:, v].mean(), v), end="")
    print()

    # ---- 绘图 ----
    # 1. RMSE vs Lead Time 曲线
    if n_compare > 1:
        plot_rmse_vs_leadtime(
            all_rmse, time_hours, tid,
            os.path.join(args.save_dir, f"ar_rmse_curve_{args.sample_id}.png"),
        )

    # 2. 选取关键时效绘制 GT vs Pred 对比 (每个变量一张图)
    if n_compare >= 6:
        key_step_indices = [0, 3, 7, 11, 15, min(23, n_compare - 1)]
        key_step_indices = [i for i in key_step_indices if i < n_compare]
    elif n_compare >= 3:
        key_step_indices = list(range(0, n_compare, max(1, n_compare // 5)))[:6]
    else:
        key_step_indices = list(range(n_compare))

    key_preds = [pred_phys_list[i] for i in key_step_indices]
    key_gts = [gt_phys_list[i] for i in key_step_indices]
    key_hours = [time_hours[i] for i in key_step_indices]

    for v_idx in range(len(VAR_NAMES)):
        plot_ar_comparison(
            key_preds, key_gts, tid, key_hours,
            os.path.join(args.save_dir, f"ar_{VAR_NAMES[v_idx]}_{args.sample_id}.png"),
            var_idx=v_idx,
        )

    # 3. 综合全变量对比 (选 +3h 和最远时效)
    if n_compare >= 2:
        fig, axes = plt.subplots(len(VAR_NAMES), 4, figsize=(16, len(VAR_NAMES) * 3))
        fig.suptitle(f'Typhoon {tid} — +3h vs +{time_hours[n_compare-1]}h',
                     fontsize=14, fontweight='bold')

        for v in range(len(VAR_NAMES)):
            g_early = gt_phys_list[0][v]
            p_early = pred_phys_list[0][v]
            vmin_e = min(g_early.min(), p_early.min())
            vmax_e = max(g_early.max(), p_early.max())
            rmse_e = all_rmse[0, v]

            ax = axes[v, 0]
            ax.imshow(g_early, cmap='RdYlBu_r', vmin=vmin_e, vmax=vmax_e)
            ax.set_title(f'{VAR_NAMES[v]} GT +3h', fontsize=9)
            ax.set_xticks([]); ax.set_yticks([])

            ax = axes[v, 1]
            ax.imshow(p_early, cmap='RdYlBu_r', vmin=vmin_e, vmax=vmax_e)
            ax.set_title(f'Pred +3h (RMSE={rmse_e:.1f})', fontsize=9)
            ax.set_xticks([]); ax.set_yticks([])

            t_last = n_compare - 1
            g_late = gt_phys_list[t_last][v]
            p_late = pred_phys_list[t_last][v]
            vmin_l = min(g_late.min(), p_late.min())
            vmax_l = max(g_late.max(), p_late.max())
            rmse_l = all_rmse[t_last, v]

            ax = axes[v, 2]
            ax.imshow(g_late, cmap='RdYlBu_r', vmin=vmin_l, vmax=vmax_l)
            ax.set_title(f'GT +{time_hours[t_last]}h', fontsize=9)
            ax.set_xticks([]); ax.set_yticks([])

            ax = axes[v, 3]
            ax.imshow(p_late, cmap='RdYlBu_r', vmin=vmin_l, vmax=vmax_l)
            ax.set_title(f'Pred +{time_hours[t_last]}h (RMSE={rmse_l:.1f})', fontsize=9)
            ax.set_xticks([]); ax.set_yticks([])

        plt.tight_layout()
        save_p = os.path.join(args.save_dir, f"ar_summary_{args.sample_id}.png")
        plt.savefig(save_p, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: {save_p}")

    print("\n可视化完成!")


if __name__ == "__main__":
    main()
