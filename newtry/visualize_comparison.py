"""
扩散模型预测 vs 真实ERA5 对比可视化

从 ar_pred_*.pt 加载预测，从数据集加载真值，逐变量逐时效对比。

用法:
  # 单样本
  python visualize_comparison.py --data_root <ERA5数据目录> --sample_id 0
  # 全部样本汇总
  python visualize_comparison.py --data_root <ERA5数据目录> --all
"""
import os
import sys
import glob
import argparse
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from configs import DataConfig
from data.dataset import ERA5TyphoonDataset, split_typhoon_ids

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


def build_dataset(args):
    """构建测试数据集（只调用一次）"""
    # 加载第一个预测文件获取通道数
    pt_files = sorted(glob.glob(os.path.join(args.output_dir, "ar_pred_*.pt")))
    if not pt_files:
        print(f"在 {args.output_dir} 中找不到 ar_pred_*.pt 文件")
        sys.exit(1)

    sample0 = torch.load(pt_files[0], map_location="cpu", weights_only=False)
    C = sample0["predictions"].shape[2]
    T_steps = sample0["predictions"].shape[1]

    # 归一化统计
    norm_path = os.path.join(args.work_dir, "norm_stats.pt")
    stats = torch.load(norm_path, weights_only=True)
    norm_mean = stats["mean"].numpy()[:C]
    norm_std = stats["std"].numpy()[:C]

    # 数据集
    data_cfg = DataConfig()
    if args.data_root:
        data_cfg.data_root = args.data_root
    data_cfg.__post_init__()

    print(f"加载数据集 (data_root: {data_cfg.data_root})...")
    _, _, test_ids = split_typhoon_ids(data_cfg.data_root, seed=42)

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
    print(f"数据集样本数: {len(test_dataset)}")
    return test_dataset, norm_mean, norm_std, C, T_steps


def compute_rmse_for_sample(sample_id, output_dir, test_dataset, norm_mean, norm_std, C, T_steps):
    """计算单个样本的逐时效 RMSE，返回 (T_steps, C) 数组，无效位置为 nan"""
    pt_path = os.path.join(output_dir, f"ar_pred_{sample_id}.pt")
    if not os.path.exists(pt_path):
        return None, None

    data = torch.load(pt_path, map_location="cpu", weights_only=False)
    preds = data["predictions"]  # (1, T, C, H, W)
    tid = data["typhoon_id"]

    rmse = np.full((T_steps, C), np.nan)
    n_valid = 0

    for t in range(T_steps):
        gt_idx = sample_id + t
        if gt_idx >= len(test_dataset):
            break
        gt_sample = test_dataset[gt_idx]
        if gt_sample["typhoon_id"] != tid:
            break

        pred_phys = denormalize(preds[0, t].numpy(), norm_mean, norm_std)
        gt_phys = denormalize(gt_sample["target"][:C].numpy(), norm_mean, norm_std)

        for v in range(C):
            rmse[t, v] = np.sqrt(np.mean((pred_phys[v] - gt_phys[v]) ** 2))
        n_valid += 1

    return rmse, tid


def run_all(args):
    """批量模式：汇总所有 ar_pred_*.pt 的 RMSE"""
    test_dataset, norm_mean, norm_std, C, T_steps = build_dataset(args)
    os.makedirs(args.save_dir, exist_ok=True)

    # 找到所有预测文件
    pt_files = sorted(glob.glob(os.path.join(args.output_dir, "ar_pred_*.pt")))
    sample_ids = []
    for f in pt_files:
        base = os.path.basename(f)
        sid = int(base.replace("ar_pred_", "").replace(".pt", ""))
        sample_ids.append(sid)
    sample_ids.sort()

    print(f"\n共 {len(sample_ids)} 个样本: {sample_ids}")

    # 收集所有样本的 RMSE
    all_rmse = []       # list of (T, C) arrays
    all_tids = []
    all_valid = []

    for sid in sample_ids:
        rmse, tid = compute_rmse_for_sample(
            sid, args.output_dir, test_dataset, norm_mean, norm_std, C, T_steps
        )
        if rmse is None:
            continue
        n_valid = np.sum(~np.isnan(rmse[:, 0]))
        all_rmse.append(rmse)
        all_tids.append(tid)
        all_valid.append(n_valid)
        print(f"  样本 {sid} ({tid}): {int(n_valid)}/{T_steps} 步有效")

    if not all_rmse:
        print("没有有效样本!")
        return

    # ---- 汇总表格: 逐时效平均 RMSE ----
    stacked = np.stack(all_rmse, axis=0)  # (N, T, C)
    mean_rmse = np.nanmean(stacked, axis=0)  # (T, C)
    std_rmse = np.nanstd(stacked, axis=0)    # (T, C)
    count_per_step = np.sum(~np.isnan(stacked[:, :, 0]), axis=0)  # (T,)

    print(f"\n{'='*100}")
    print(f"汇总: {len(all_rmse)} 个样本的平均 RMSE")
    print(f"{'='*100}")
    print(f"{'时效':<8} {'N':>3}", end="")
    for vn in VAR_NAMES[:C]:
        print(f" {vn:>8}", end="")
    print()
    print("-" * (12 + 9 * C))

    for t in range(T_steps):
        if count_per_step[t] == 0:
            continue
        print(f"+{(t+1)*3:>3}h    {int(count_per_step[t]):>3}", end="")
        for v in range(C):
            print(f" {mean_rmse[t, v]:>8.2f}", end="")
        print()

    print(f"\n{'平均':>7}    ", end="")
    for v in range(C):
        print(f" {np.nanmean(mean_rmse[:, v]):>8.2f}", end="")
    print()

    # ---- 逐样本汇总 ----
    print(f"\n{'='*80}")
    print(f"逐样本平均 RMSE (所有时效)")
    print(f"{'='*80}")
    print(f"{'样本':>6} {'台风ID':<20} {'有效步':>6} {'u_avg':>8} {'v_avg':>8} {'z_avg':>10}")
    print("-" * 70)
    for i, (sid, tid, rmse) in enumerate(zip(sample_ids, all_tids, all_rmse)):
        u_mean = np.nanmean(rmse[:, :3])
        v_mean = np.nanmean(rmse[:, 3:6])
        z_mean = np.nanmean(rmse[:, 6:9])
        print(f"{sid:>6} {tid:<20} {int(all_valid[i]):>6} {u_mean:>8.2f} {v_mean:>8.2f} {z_mean:>10.2f}")

    # ---- 图1: 平均 RMSE vs Lead Time ----
    valid_mask = count_per_step > 0
    hours = np.array([(t + 1) * 3 for t in range(T_steps)])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f'Mean RMSE vs Lead Time ({len(all_rmse)} samples)', fontsize=14, fontweight='bold')

    # 风场
    for v in range(min(6, C)):
        y = mean_rmse[valid_mask, v]
        yerr = std_rmse[valid_mask, v]
        ax1.plot(hours[valid_mask], y, marker='o', markersize=3, label=VAR_NAMES[v], linewidth=1.5)
        ax1.fill_between(hours[valid_mask], y - yerr, y + yerr, alpha=0.1)
    ax1.set_xlabel('Lead time (hours)')
    ax1.set_ylabel('RMSE (m/s)')
    ax1.set_title('Wind (u, v)')
    ax1.legend(fontsize=8, ncol=2)
    ax1.grid(True, alpha=0.3)

    # 位势高度
    for v in range(6, C):
        y = mean_rmse[valid_mask, v]
        yerr = std_rmse[valid_mask, v]
        ax2.plot(hours[valid_mask], y, marker='s', markersize=3, label=VAR_NAMES[v], linewidth=1.5)
        ax2.fill_between(hours[valid_mask], y - yerr, y + yerr, alpha=0.1)
    ax2.set_xlabel('Lead time (hours)')
    ax2.set_ylabel('RMSE (m²/s²)')
    ax2.set_title('Geopotential (z)')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    save_p = os.path.join(args.save_dir, "all_rmse_curve.png")
    plt.savefig(save_p, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {save_p}")

    # ---- 图2: 热力图 (时效 x 变量) ----
    fig, ax = plt.subplots(figsize=(12, 6))
    # 归一化 RMSE (除以 norm_std)
    norm_rmse = mean_rmse.copy()
    for v in range(C):
        norm_rmse[:, v] /= (norm_std[v] + 1e-8)

    im = ax.imshow(norm_rmse[valid_mask].T, aspect='auto', cmap='YlOrRd',
                   vmin=0, vmax=min(2.0, np.nanmax(norm_rmse[valid_mask])))
    ax.set_yticks(range(C))
    ax.set_yticklabels(VAR_NAMES[:C])
    xtick_idx = list(range(0, int(valid_mask.sum()), max(1, int(valid_mask.sum()) // 8)))
    ax.set_xticks(xtick_idx)
    ax.set_xticklabels([f"+{hours[valid_mask][i]}h" for i in xtick_idx])
    ax.set_xlabel('Lead time')
    ax.set_title(f'Normalized RMSE (RMSE/std) — {len(all_rmse)} samples', fontsize=13)
    plt.colorbar(im, ax=ax, label='RMSE / std')

    # 标注数值
    for row in range(C):
        for col_i, col in enumerate(np.where(valid_mask)[0]):
            val = norm_rmse[col, row]
            if not np.isnan(val):
                color = 'white' if val > 1.0 else 'black'
                ax.text(col_i, row, f'{val:.2f}', ha='center', va='center',
                        fontsize=6, color=color)

    plt.tight_layout()
    save_p = os.path.join(args.save_dir, "all_rmse_heatmap.png")
    plt.savefig(save_p, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_p}")

    # ---- 图3: 箱线图 (每个变量在 24h/48h/72h 的分布) ----
    key_steps = {'24h': 7, '48h': 15, '72h': 23}
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f'RMSE Distribution at Key Lead Times ({len(all_rmse)} samples)',
                 fontsize=13, fontweight='bold')

    for ax_i, (label, t_idx) in enumerate(key_steps.items()):
        if t_idx >= T_steps:
            continue
        data_box = []
        labels_box = []
        for v in range(C):
            vals = stacked[:, t_idx, v]
            vals = vals[~np.isnan(vals)]
            if len(vals) > 0:
                data_box.append(vals)
                labels_box.append(VAR_NAMES[v])

        if data_box:
            bp = axes[ax_i].boxplot(data_box, labels=labels_box, patch_artist=True)
            colors = ['#4C72B0'] * 3 + ['#55A868'] * 3 + ['#C44E52'] * 3
            for patch, color in zip(bp['boxes'], colors[:len(data_box)]):
                patch.set_facecolor(color)
                patch.set_alpha(0.6)
            axes[ax_i].set_title(f'+{label}')
            axes[ax_i].tick_params(axis='x', rotation=45)
            axes[ax_i].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    save_p = os.path.join(args.save_dir, "all_rmse_boxplot.png")
    plt.savefig(save_p, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_p}")

    print(f"\n汇总完成! 图片保存在: {args.save_dir}/")


def run_single(args):
    """单样本模式"""
    test_dataset, norm_mean, norm_std, C, T_steps = build_dataset(args)
    os.makedirs(args.save_dir, exist_ok=True)

    # 加载预测
    pt_path = os.path.join(args.output_dir, f"ar_pred_{args.sample_id}.pt")
    if not os.path.exists(pt_path):
        print(f"文件不存在: {pt_path}")
        sys.exit(1)

    data = torch.load(pt_path, map_location="cpu", weights_only=False)
    preds = data["predictions"]
    tid = data["typhoon_id"]
    print(f"台风: {tid}, 预测: {preds.shape}")

    # 收集真值
    gt_list = []
    for t in range(T_steps):
        gt_idx = args.sample_id + t
        if gt_idx < len(test_dataset):
            gt_sample = test_dataset[gt_idx]
            if gt_sample["typhoon_id"] == tid:
                gt_list.append(gt_sample["target"][:C].numpy())
            else:
                gt_list.append(None)
        else:
            gt_list.append(None)

    n_valid = sum(1 for g in gt_list if g is not None)
    print(f"真值: {n_valid}/{T_steps} 步")
    if n_valid == 0:
        print("错误: 无法获取真值!")
        sys.exit(1)

    # 反归一化 + RMSE
    pred_phys = []
    gt_phys = []
    rmse_table = np.full((T_steps, C), np.nan)

    for t in range(T_steps):
        p = denormalize(preds[0, t].numpy(), norm_mean, norm_std)
        pred_phys.append(p)
        if gt_list[t] is not None:
            g = denormalize(gt_list[t], norm_mean, norm_std)
            gt_phys.append(g)
            for v in range(C):
                rmse_table[t, v] = np.sqrt(np.mean((p[v] - g[v]) ** 2))
        else:
            gt_phys.append(None)

    # 打印表格
    print(f"\n{'时效':<8}", end="")
    for vn in VAR_NAMES[:C]:
        print(f" {vn:>8}", end="")
    print()
    print("-" * (8 + 9 * C))
    for t in range(T_steps):
        if np.isnan(rmse_table[t, 0]):
            continue
        print(f"+{(t+1)*3:>3}h    ", end="")
        for v in range(C):
            print(f" {rmse_table[t, v]:>8.2f}", end="")
        print()

    # 选关键时效绘图
    valid_steps = [t for t in range(T_steps) if gt_phys[t] is not None]
    if len(valid_steps) >= 6:
        pick = [valid_steps[0]]
        for target in [3, 7, 11, 15, 23]:
            closest = min(valid_steps, key=lambda x: abs(x - target))
            if closest not in pick:
                pick.append(closest)
        pick = sorted(pick)[:6]
    else:
        pick = valid_steps[:6]

    n_cols = len(pick)
    if n_cols == 0:
        return

    for v_idx in range(C):
        fig, axes = plt.subplots(3, n_cols, figsize=(4 * n_cols, 10))
        if n_cols == 1:
            axes = axes.reshape(3, 1)
        fig.suptitle(f'{tid} — {VAR_NAMES[v_idx]} ({VAR_UNITS.get(VAR_NAMES[v_idx], "")})',
                     fontsize=14, fontweight='bold')
        for col, t in enumerate(pick):
            p = pred_phys[t][v_idx]
            g = gt_phys[t][v_idx]
            diff = p - g
            rmse = np.sqrt(np.mean(diff ** 2))
            vmin = min(g.min(), p.min())
            vmax = max(g.max(), p.max())

            im0 = axes[0, col].imshow(g, cmap='RdYlBu_r', vmin=vmin, vmax=vmax)
            axes[0, col].set_title(f'GT +{(t+1)*3}h', fontsize=10)
            axes[0, col].set_xticks([]); axes[0, col].set_yticks([])
            plt.colorbar(im0, ax=axes[0, col], fraction=0.046)

            im1 = axes[1, col].imshow(p, cmap='RdYlBu_r', vmin=vmin, vmax=vmax)
            axes[1, col].set_title(f'Pred +{(t+1)*3}h', fontsize=10)
            axes[1, col].set_xticks([]); axes[1, col].set_yticks([])
            plt.colorbar(im1, ax=axes[1, col], fraction=0.046)

            abs_max = max(abs(diff.min()), abs(diff.max()), 1e-8)
            im2 = axes[2, col].imshow(diff, cmap='bwr', vmin=-abs_max, vmax=abs_max)
            axes[2, col].set_title(f'Err RMSE={rmse:.1f}', fontsize=10)
            axes[2, col].set_xticks([]); axes[2, col].set_yticks([])
            plt.colorbar(im2, ax=axes[2, col], fraction=0.046)

        axes[0, 0].set_ylabel('Ground Truth', fontsize=11)
        axes[1, 0].set_ylabel('Prediction', fontsize=11)
        axes[2, 0].set_ylabel('Error', fontsize=11)
        plt.tight_layout()
        plt.savefig(os.path.join(args.save_dir, f"cmp_{VAR_NAMES[v_idx]}_{args.sample_id}.png"),
                    dpi=150, bbox_inches='tight')
        plt.close()

    print(f"\n图片保存在: {args.save_dir}/")


def main():
    parser = argparse.ArgumentParser(description="扩散模型预测 vs 真实ERA5 对比可视化")
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--sample_id", type=int, default=0)
    parser.add_argument("--save_dir", type=str, default="visualizations")
    parser.add_argument("--data_root", type=str, default=None)
    parser.add_argument("--preprocess_dir", type=str, default=None)
    parser.add_argument("--work_dir", type=str, default=".")
    parser.add_argument("--all", action="store_true", help="汇总所有 ar_pred_*.pt 的 RMSE")
    args = parser.parse_args()

    if args.all:
        run_all(args)
    else:
        run_single(args)


if __name__ == "__main__":
    main()
