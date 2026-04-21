"""
自回归预测 GIF 动画生成

生成 72h 预测过程的 GIF 动画（GT vs Pred 逐帧对比）
"""
import os
import sys
import argparse
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
import io

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from configs import DataConfig, get_config
from data.dataset import ERA5TyphoonDataset, split_typhoon_ids


VAR_NAMES = ['u_850', 'u_500', 'v_850', 'v_500', 'u10m', 'v10m']


def denormalize(data_norm, mean, std):
    """反归一化: (C, H, W) numpy"""
    std = np.where(std < 1e-8, 1.0, std)
    return data_norm * std.reshape(-1, 1, 1) + mean.reshape(-1, 1, 1)


def fig_to_image(fig):
    """将 matplotlib figure 转为 PIL Image"""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=120, bbox_inches='tight')
    buf.seek(0)
    img = Image.open(buf).copy()
    buf.close()
    return img


def make_gif_per_variable(pred_list, gt_list, typhoon_id, var_idx, save_path, fps=3):
    """
    为单个变量生成 GT vs Pred 并排对比 GIF

    pred_list: list of (C, H, W) numpy, 长度=T
    gt_list:   list of (C, H, W) numpy, 长度=T
    """
    # 计算全局 colorbar 范围（所有时间步统一颜色）
    all_vals = []
    for t in range(len(pred_list)):
        all_vals.append(pred_list[t][var_idx])
        all_vals.append(gt_list[t][var_idx])
    global_min = min(v.min() for v in all_vals)
    global_max = max(v.max() for v in all_vals)

    frames = []
    for t in range(len(pred_list)):
        gt_field = gt_list[t][var_idx]
        pred_field = pred_list[t][var_idx]
        rmse = np.sqrt(np.mean((pred_field - gt_field) ** 2))
        hour = (t + 1) * 3

        fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(8, 3.5))
        fig.suptitle(f'{VAR_NAMES[var_idx]}  |  Typhoon {typhoon_id}  |  +{hour}h',
                     fontsize=13, fontweight='bold')

        im0 = ax0.imshow(gt_field, cmap='RdYlBu_r', vmin=global_min, vmax=global_max, aspect='equal')
        ax0.set_title('Ground Truth', fontsize=11)
        ax0.set_xticks([]); ax0.set_yticks([])
        plt.colorbar(im0, ax=ax0, fraction=0.046, pad=0.04)

        im1 = ax1.imshow(pred_field, cmap='RdYlBu_r', vmin=global_min, vmax=global_max, aspect='equal')
        ax1.set_title(f'Prediction (RMSE={rmse:.1f} m/s)', fontsize=11)
        ax1.set_xticks([]); ax1.set_yticks([])
        plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

        plt.tight_layout()
        frames.append(fig_to_image(fig))
        plt.close()

    # 保存 GIF
    duration = int(1000 / fps)
    frames[0].save(
        save_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration,
        loop=0,
    )
    print(f"Saved: {save_path}  ({len(frames)} frames, {fps} fps)")


def make_gif_all_variables(pred_list, gt_list, typhoon_id, save_path, fps=3):
    """
    生成所有变量合在一起的 GIF（每帧 6 行 × 2 列）
    """
    n_vars = len(VAR_NAMES)

    # 全局 colorbar 范围（每个变量独立）
    global_ranges = []
    for v in range(n_vars):
        all_vals = [pred_list[t][v] for t in range(len(pred_list))] + \
                   [gt_list[t][v] for t in range(len(gt_list))]
        global_ranges.append((min(x.min() for x in all_vals), max(x.max() for x in all_vals)))

    frames = []
    for t in range(len(pred_list)):
        hour = (t + 1) * 3
        fig, axes = plt.subplots(n_vars, 2, figsize=(9, n_vars * 2.8))
        fig.suptitle(f'Typhoon {typhoon_id}  |  +{hour}h', fontsize=14, fontweight='bold')

        for v in range(n_vars):
            gt_field = gt_list[t][v]
            pred_field = pred_list[t][v]
            rmse = np.sqrt(np.mean((pred_field - gt_field) ** 2))
            vmin, vmax = global_ranges[v]

            ax0 = axes[v, 0]
            im0 = ax0.imshow(gt_field, cmap='RdYlBu_r', vmin=vmin, vmax=vmax, aspect='equal')
            ax0.set_title(f'{VAR_NAMES[v]} - GT', fontsize=9)
            ax0.set_xticks([]); ax0.set_yticks([])
            plt.colorbar(im0, ax=ax0, fraction=0.046, pad=0.04)

            ax1 = axes[v, 1]
            im1 = ax1.imshow(pred_field, cmap='RdYlBu_r', vmin=vmin, vmax=vmax, aspect='equal')
            ax1.set_title(f'{VAR_NAMES[v]} - Pred (RMSE={rmse:.1f})', fontsize=9)
            ax1.set_xticks([]); ax1.set_yticks([])
            plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

        plt.tight_layout()
        frames.append(fig_to_image(fig))
        plt.close()

    duration = int(1000 / fps)
    frames[0].save(
        save_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration,
        loop=0,
    )
    print(f"Saved: {save_path}  ({len(frames)} frames, {fps} fps)")


def main():
    parser = argparse.ArgumentParser(description="Generate GIF for autoregressive predictions")
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--sample_id", type=int, default=0)
    parser.add_argument("--save_dir", type=str, default="visualizations")
    parser.add_argument("--data_root", type=str, default="/root/autodl-tmp/Typhoon_data_final")
    parser.add_argument("--work_dir", type=str, default=".")
    parser.add_argument("--preprocess_dir", type=str, default=None)
    parser.add_argument("--fps", type=int, default=3, help="GIF 帧率")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    # ---- 加载自回归预测 ----
    pt_path = os.path.join(args.output_dir, f"ar_pred_{args.sample_id}.pt")
    if not os.path.exists(pt_path):
        print(f"File not found: {pt_path}")
        sys.exit(1)

    data = torch.load(pt_path, map_location="cpu", weights_only=False)
    preds = data["predictions"]  # (1, T, C, H, W)
    tid = data["typhoon_id"]
    T_steps = preds.shape[1]

    print(f"Typhoon: {tid}")
    print(f"Predictions: {T_steps} steps -> {T_steps * 3}h")

    # ---- 加载归一化统计 ----
    norm_stats_path = os.path.join(args.work_dir, "norm_stats.pt")
    assert os.path.exists(norm_stats_path), f"找不到: {norm_stats_path}"
    stats = torch.load(norm_stats_path, weights_only=True)
    norm_mean = stats["mean"].numpy()
    norm_std = stats["std"].numpy()

    # ---- 加载真值 ----
    data_cfg = DataConfig()
    data_cfg.data_root = args.data_root
    _, _, test_ids = split_typhoon_ids(data_cfg.data_root, seed=42)

    print("加载测试集获取真值...")
    test_dataset = ERA5TyphoonDataset(
        typhoon_ids=test_ids[:10],
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

    base_sample = test_dataset[args.sample_id]
    base_tid = base_sample["typhoon_id"]

    gt_targets = []
    idx = args.sample_id
    collected = 0
    while collected < T_steps and idx < len(test_dataset):
        sample = test_dataset[idx]
        if sample["typhoon_id"] == base_tid:
            gt_targets.append(sample["target"])
            collected += 1
        else:
            break
        idx += 1

    n_compare = min(len(gt_targets), T_steps)
    print(f"找到 {n_compare}/{T_steps} 步真值")

    if n_compare == 0:
        print("无法获取真值，退出")
        sys.exit(1)

    # ---- 反归一化 ----
    pred_phys_list = []
    gt_phys_list = []
    for t in range(n_compare):
        pred_phys = denormalize(preds[0, t].numpy(), norm_mean, norm_std)
        gt_phys = denormalize(gt_targets[t].numpy(), norm_mean, norm_std)
        pred_phys_list.append(pred_phys)
        gt_phys_list.append(gt_phys)

    # ---- 生成 GIF ----
    # 1. 每个变量单独的 GIF
    for v_idx in range(len(VAR_NAMES)):
        gif_path = os.path.join(args.save_dir, f"gif_{VAR_NAMES[v_idx]}_{args.sample_id}.gif")
        make_gif_per_variable(
            pred_phys_list, gt_phys_list, tid, v_idx, gif_path, fps=args.fps
        )

    # 2. 所有变量合在一起的 GIF
    gif_all_path = os.path.join(args.save_dir, f"gif_all_vars_{args.sample_id}.gif")
    make_gif_all_variables(
        pred_phys_list, gt_phys_list, tid, gif_all_path, fps=args.fps
    )

    print(f"\n完成! 共生成 {len(VAR_NAMES) + 1} 个 GIF 文件")


if __name__ == "__main__":
    main()
