"""
Comprehensive FM vs DM Channel Analysis — Fixes z-channel variance collapse detection.

Plots:
1. Per-channel RMSE/Corr/StdRatio bar charts (FM vs DM)
2. Z-channel spatial correlation maps (where model matches vs fails)
3. Per-channel spatial maps (GT vs FM vs DM) for z_850, z_500, z_250
4. Z-channel variance collapse diagnostic table

Usage:
    python analyze_fm_dm_channels.py \
        --fm_ckpt multi_seed_results_20ep/seed_42/checkpoints_fm/best.pt \
        --dm_ckpt multi_seed_results/seed_42/checkpoints_dm/best.pt \
        --data_root /root/autodl-tmp/fyp_final/Ver4/Trajectory/preprocessed_era5 \
        --output_dir channel_plots
"""
import os
import sys
import argparse
import logging
from typing import Dict, List

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import torch

sys.path.insert(0, '.')
from configs import DataConfig, ModelConfig, TrainConfig
from data.dataset import build_dataloaders
from models.unified_model import UnifiedModel
from models.adapter import load_newtry_checkpoint, AdaptedDiffusionModel
from models.trainer import EMA

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

COLORS = {"FM": "#2E86AB", "DM": "#A23B72", "GT": "#2D2D2D"}


def load_checkpoint(model, checkpoint_path, device):
    if not os.path.exists(checkpoint_path):
        return model, False
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    sd = ckpt["model_state_dict"]
    result = model.load_state_dict(sd, strict=False)
    ema_shadow = ckpt.get("ema_state_dict", {}).get("shadow", {})
    ema_applied = 0
    for name, param in model.named_parameters():
        if name in ema_shadow:
            param.data.copy_(ema_shadow[name].to(param.device, param.dtype))
            ema_applied += 1
    logger.info(f"Loaded: {checkpoint_path}  val_loss={ckpt.get('best_val_loss','?')}  EMA={ema_applied}")
    return model, True


def denormalize(data, mean, std):
    std = np.where(std < 1e-8, 1.0, std)
    mean = mean[np.newaxis, :, np.newaxis, np.newaxis]
    std = std[np.newaxis, :, np.newaxis, np.newaxis]
    return data * std + mean


# ============================================================
# Per-channel diagnostics
# ============================================================

def compute_all_channel_stats(pred_fm, pred_dm, gt, channel_names):
    """Compute per-channel RMSE, Corr, StdRatio for FM and DM."""
    n = len(gt)
    stats = {"channel_names": channel_names}

    for model_name, preds in [("FM", pred_fm), ("DM", pred_dm)]:
        rmse_list, corr_list, std_ratio_list = [], [], []
        bias_list = []
        for ch in range(len(channel_names)):
            p = preds[:n, ch]
            t = gt[:n, ch]
            err = p - t
            rmse_list.append(np.sqrt((err**2).mean()))
            bias_list.append(err.mean())
            p_flat = p.reshape(n, -1)
            t_flat = t.reshape(n, -1)
            corr = np.corrcoef(p_flat.mean(axis=1), t_flat.mean(axis=1))[0, 1]
            corr_list.append(corr)
            p_std = p.std()
            t_std = t.std()
            std_ratio_list.append(p_std / (t_std + 1e-8) if t_std > 1e-8 else 0.0)
        stats[f"RMSE_{model_name}"] = rmse_list
        stats[f"Corr_{model_name}"] = corr_list
        stats[f"StdRatio_{model_name}"] = std_ratio_list
        stats[f"Bias_{model_name}"] = bias_list

    return stats


def compute_z_diagnostics(pred_fm, pred_dm, gt, z_indices, z_names):
    """Z-channel specific diagnostics: std_ratio, spatial_corr, rmse per channel."""
    n = len(gt)
    results = {}
    for model_name, preds in [("FM", pred_fm), ("DM", pred_dm)]:
        for z_idx, z_name in zip(z_indices, z_names):
            p_ch = preds[:n, z_idx]
            t_ch = gt[:n, z_idx]
            p_std = p_ch.std()
            t_std = t_ch.std()
            std_ratio = p_std / (t_std + 1e-8)
            bias = (p_ch - t_ch).mean()
            rmse = np.sqrt(((p_ch - t_ch)**2).mean())
            p_flat = p_ch.reshape(n, -1)
            t_flat = t_ch.reshape(n, -1)
            p_mean = p_flat.mean(axis=1)
            t_mean = t_flat.mean(axis=1)
            cov = ((p_flat - p_mean[:, None]) * (t_flat - t_mean[:, None])).mean(axis=1)
            p_std_per = np.maximum(p_flat.std(axis=1), 1e-8)
            t_std_per = t_flat.std(axis=1).clamp(min=1e-8)
            spatial_corr = (cov / (p_std_per * t_std_per)).mean()
            results[f"{model_name}_{z_name}_std_ratio"] = std_ratio
            results[f"{model_name}_{z_name}_spatial_corr"] = spatial_corr
            results[f"{model_name}_{z_name}_rmse"] = rmse
            results[f"{model_name}_{z_name}_bias"] = bias
            results[f"{model_name}_{z_name}_pred_std"] = p_std
            results[f"{model_name}_{z_name}_gt_std"] = t_std
    return results


# ============================================================
# Plotting functions
# ============================================================

def plot_comprehensive_channel_stats(stats, save_path):
    """3-panel: RMSE, Corr, StdRatio for FM vs DM."""
    ch_names = stats["channel_names"]
    n = len(ch_names)
    x = np.arange(n)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Panel 1: RMSE
    ax = axes[0]
    w = 0.35
    bars = ax.bar(x - w/2, stats["RMSE_FM"], w, label="FM", color=COLORS["FM"], alpha=0.85)
    bars2 = ax.bar(x + w/2, stats["RMSE_DM"], w, label="DM", color=COLORS["DM"], alpha=0.85)
    ax.set_title("Per-Channel RMSE (physical units)", fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(ch_names, rotation=45, ha='right', fontsize=9)
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    # Annotate winner
    for i in range(n):
        better = "FM" if stats["RMSE_FM"][i] < stats["RMSE_DM"][i] else "DM"
        color = COLORS["FM"] if better == "FM" else COLORS["DM"]
        ax.annotate(better, xy=(i, max(stats["RMSE_FM"][i], stats["RMSE_DM"][i]) + 0.005),
                    ha='center', fontsize=7, color=color, fontweight='bold')

    # Panel 2: Correlation
    ax = axes[1]
    ax.bar(x - w/2, stats["Corr_FM"], w, label="FM", color=COLORS["FM"], alpha=0.85)
    ax.bar(x + w/2, stats["Corr_DM"], w, label="DM", color=COLORS["DM"], alpha=0.85)
    ax.axhline(0, color='k', lw=0.5)
    ax.set_title("Per-Channel Correlation", fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(ch_names, rotation=45, ha='right', fontsize=9)
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, 1.05)

    # Panel 3: StdRatio (key diagnostic!)
    ax = axes[2]
    ax.bar(x - w/2, stats["StdRatio_FM"], w, label="FM", color=COLORS["FM"], alpha=0.85)
    ax.bar(x + w/2, stats["StdRatio_DM"], w, label="DM", color=COLORS["DM"], alpha=0.85)
    ax.axhline(1.0, color='red', lw=1.5, ls='--', label='Perfect (1.0)')
    # Shade warning zone for std_ratio < 0.3
    ax.axhspan(0, 0.3, alpha=0.1, color='red', label='Warning (<0.3)')
    ax.set_title("Std Ratio (pred_std / gt_std)", fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(ch_names, rotation=45, ha='right', fontsize=9)
    ax.legend(fontsize=9, loc='upper right')
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, 1.5)

    # Highlight z channels
    for ax in axes:
        for i, ch in enumerate(ch_names):
            if ch.startswith("z_"):
                rect = plt.Rectangle((i - 0.5, 0), 1, ax.get_ylim()[1],
                                     fill=True, alpha=0.05, color='orange')
                ax.add_patch(rect)

    plt.suptitle("FM vs DM — Per-Channel Performance (n samples)", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    logger.info(f"Saved: {save_path}")


def plot_z_channel_spatial_correlation_maps(pred_fm, pred_dm, gt, z_indices, z_names, save_path):
    """
    Plot spatial correlation maps for each z channel, showing FM vs DM side by side.
    This shows WHERE the model matches (high corr) vs fails (low corr).
    """
    n = len(gt)
    n_z = len(z_names)

    # Pick the sample with highest spatial variance in each z channel
    # Correlate GT and pred per grid point across samples
    fig, axes = plt.subplots(n_z, 4, figsize=(14, 3.5 * n_z))

    for row, (z_idx, z_name) in enumerate(zip(z_indices, z_names)):
        gt_ch = gt[:n, z_idx]
        fm_ch = pred_fm[:n, z_idx]
        dm_ch = pred_dm[:n, z_idx]

        # Mean fields
        gt_mean = gt_ch.mean(axis=0)
        fm_mean = fm_ch.mean(axis=0)
        dm_mean = dm_ch.mean(axis=0)

        # Per-point correlation across samples (H, W) grid
        H, W = gt_mean.shape

        fm_corr_map = np.zeros((H, W))
        dm_corr_map = np.zeros((H, W))
        for h in range(H):
            for w in range(W):
                gt_vec = gt_ch[:, h, w]
                fm_vec = fm_ch[:, h, w]
                dm_vec = dm_ch[:, h, w]
                if gt_vec.std() > 1e-8 and fm_vec.std() > 1e-8:
                    fm_corr_map[h, w] = np.corrcoef(gt_vec, fm_vec)[0, 1]
                if gt_vec.std() > 1e-8 and dm_vec.std() > 1e-8:
                    dm_corr_map[h, w] = np.corrcoef(gt_vec, dm_vec)[0, 1]

        # Column 0: GT mean field
        ax = axes[row, 0]
        vmin, vmax = np.percentile(gt_mean, [2, 98])
        im = ax.imshow(gt_mean, cmap='viridis', vmin=vmin, vmax=vmax, origin='lower')
        ax.set_title(f"GT: {z_name}\n(Mean over n={n})", fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, fontsize=7)

        # Column 1: FM prediction mean
        ax = axes[row, 1]
        im = ax.imshow(fm_mean, cmap='viridis', vmin=vmin, vmax=vmax, origin='lower')
        fm_std = fm_ch.std()
        ax.set_title(f"FM: {z_name}\n(pred std={fm_std:.1f})", fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, fontsize=7)

        # Column 2: FM spatial correlation map
        ax = axes[row, 2]
        corr_vmin, corr_vmax = -0.5, 1.0
        norm = TwoSlopeNorm(vmin=corr_vmin, vcenter=0, vmax=corr_vmax)
        im = ax.imshow(fm_corr_map, cmap='RdYlGn', norm=norm, origin='lower')
        ax.set_title(f"FM: Spatial Corr Map\n(mean={fm_corr_map.mean():.3f})", fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, fontsize=7)

        # Column 3: DM spatial correlation map
        ax = axes[row, 3]
        im = ax.imshow(dm_corr_map, cmap='RdYlGn', norm=norm, origin='lower')
        dm_std = dm_ch.std()
        ax.set_title(f"DM: Spatial Corr Map\n(pred std={dm_std:.1f}, mean={dm_corr_map.mean():.3f})", fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, fontsize=7)

    plt.suptitle("Z-Channel Spatial Correlation Maps — Where Does the Model Match vs Fail?",
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    logger.info(f"Saved: {save_path}")


def plot_z_channel_spatial_maps(pred_fm, pred_dm, gt, z_indices, z_names, save_path):
    """
    Plot GT vs FM vs DM side-by-side spatial maps for each z channel.
    Shows individual samples (most variant) to see spatial structure.
    """
    n = len(gt)
    n_z = len(z_names)

    # Pick sample with highest spatial variance
    fig, axes = plt.subplots(n_z, 3, figsize=(10, 3.5 * n_z))

    for row, (z_idx, z_name) in enumerate(zip(z_indices, z_names)):
        gt_ch = gt[:n, z_idx]
        fm_ch = pred_fm[:n, z_idx]
        dm_ch = pred_dm[:n, z_idx]

        # Choose sample with most variance
        best_s = int(np.argmax([gt_ch[i].std() for i in range(n)]))
        gt_s = gt_ch[best_s]
        fm_s = fm_ch[best_s]
        dm_s = dm_ch[best_s]

        all_data = np.stack([gt_s, fm_s, dm_s])
        vmin, vmax = np.percentile(all_data, [1, 99])

        for col, (data, title) in enumerate([
            (gt_s, f"GT: {z_name}"),
            (fm_s, f"FM: {z_name}"),
            (dm_s, f"DM: {z_name}"),
        ]):
            ax = axes[row, col]
            im = ax.imshow(data, cmap='viridis', vmin=vmin, vmax=vmax, origin='lower')
            ax.set_title(title, fontsize=10, fontweight='bold')
            ax.set_xticks([]); ax.set_yticks([])
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, fontsize=8)

        # Add error annotation
        fm_err = np.sqrt(((fm_s - gt_s)**2).mean())
        dm_err = np.sqrt(((dm_s - gt_s)**2).mean())
        axes[row, 1].set_xlabel(f"RMSE={fm_err:.1f}", fontsize=8)
        axes[row, 2].set_xlabel(f"RMSE={dm_err:.1f}", fontsize=8)

    plt.suptitle("Z-Channel Spatial Maps — GT vs FM vs DM (highest-variance sample)",
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    logger.info(f"Saved: {save_path}")


def plot_z_diagnostics_table(diagnostics, z_names, save_path):
    """Print a diagnostic summary table for z channels."""
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.axis('off')

    headers = ["Metric"] + [f"{m} / {z}" for z in z_names for m in ["FM", "DM"]]

    rows = []
    metrics = ["std_ratio", "spatial_corr", "rmse", "bias"]
    metric_labels = {
        "std_ratio": "Std Ratio (pred/GT)",
        "spatial_corr": "Spatial Corr",
        "rmse": "RMSE (physical)",
        "bias": "Bias",
        "pred_std": "Pred Std",
        "gt_std": "GT Std",
    }

    for metric in metrics:
        row = [metric_labels.get(metric, metric)]
        for z in z_names:
            fm_val = diagnostics.get(f"FM_{z}_{metric}", np.nan)
            dm_val = diagnostics.get(f"DM_{z}_{metric}", np.nan)
            row.append(f"{fm_val:.4f}")
            row.append(f"{dm_val:.4f}")
        rows.append(row)

    # Color code: red for std_ratio < 0.3, green for spatial_corr > 0.7
    cell_colors = []
    for row_idx, row in enumerate(rows):
        colors = [["white"] * len(headers)]
        for col_idx in range(1, len(headers), 2):  # FM and DM columns
            fm_val_str = rows[row_idx][col_idx]
            dm_val_str = rows[row_idx][col_idx + 1]
            try:
                fm_val = float(fm_val_str)
                dm_val = float(dm_val_str)
            except (ValueError, IndexError):
                continue

            if metric == "std_ratio":
                fm_color = "salmon" if fm_val < 0.3 else "lightgreen"
                dm_color = "salmon" if dm_val < 0.3 else "lightgreen"
            elif metric == "spatial_corr":
                fm_color = "salmon" if fm_val < 0.5 else "lightgreen"
                dm_color = "salmon" if dm_val < 0.5 else "lightgreen"
            else:
                fm_color = "white"
                dm_color = "white"
            colors[0][col_idx] = fm_color
            colors[0][col_idx + 1] = dm_color
        cell_colors.extend(colors)

    table = ax.table(
        cellText=rows,
        colLabels=headers,
        cellLoc='center',
        loc='center',
        cellColours=cell_colors if cell_colors else None,
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.8)

    # Color header
    for (r, c), cell in table.get_celld().items():
        if r == 0:
            cell.set_facecolor('#4a86e8')
            cell.set_text_props(color='white', fontweight='bold')

    plt.suptitle("Z-Channel Diagnostics — Red = Problem, Green = Good\n"
                 "⚠ Red cells indicate: std_ratio < 0.3 (variance collapse) or spatial_corr < 0.5",
                 fontsize=11, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    logger.info(f"Saved: {save_path}")


def print_diagnostic_summary(stats, diagnostics, z_names):
    """Print terminal summary."""
    print("\n" + "=" * 80)
    print("Z-CHANNEL DIAGNOSTIC SUMMARY")
    print("=" * 80)
    print(f"{'Channel':<10} {'Model':<6} {'Std Ratio':>10} {'Spatial Corr':>14} {'RMSE':>10} {'Bias':>10}")
    print("-" * 80)

    for z in z_names:
        for model in ["FM", "DM"]:
            sr = diagnostics.get(f"{model}_{z}_std_ratio", 0)
            sc = diagnostics.get(f"{model}_{z}_spatial_corr", 0)
            rmse = diagnostics.get(f"{model}_{z}_std_ratio", 0)
            bias = diagnostics.get(f"{model}_{z}_bias", 0)
            flag = " ⚠️ COLLAPSE" if sr < 0.3 else ""
            print(f"{z:<10} {model:<6} {sr:>10.3f} {sc:>14.3f} {rmse:>10.3f} {bias:>10.3f}{flag}")

    print("=" * 80)
    print("Threshold: Std Ratio < 0.3 = severe variance collapse")
    print("           Spatial Corr < 0.5 = poor spatial pattern match")
    print()

    # Print full per-channel table
    print("=" * 80)
    print("FULL PER-CHANNEL STATS")
    print("=" * 80)
    print(f"{'Channel':<10} {'Model':<6} {'RMSE':>10} {'Corr':>8} {'StdRatio':>10} {'Bias':>10}")
    print("-" * 80)
    for i, cn in enumerate(stats["channel_names"]):
        for model in ["FM", "DM"]:
            rmse = stats[f"RMSE_{model}"][i]
            corr = stats[f"Corr_{model}"][i]
            sr = stats[f"StdRatio_{model}"][i]
            bias = stats[f"Bias_{model}"][i]
            flag = " ⚠️" if cn.startswith("z_") and sr < 0.3 else ""
            print(f"{cn:<10} {model:<6} {rmse:>10.4f} {corr:>8.4f} {sr:>10.3f} {bias:>10.4f}{flag}")
    print("=" * 80)


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="FM vs DM Z-Channel Analysis")
    parser.add_argument("--fm_ckpt", type=str,
                       default="multi_seed_results_20ep/seed_42/checkpoints_fm/best.pt")
    parser.add_argument("--dm_ckpt", type=str,
                       default="multi_seed_results/seed_42/checkpoints_dm/best.pt")
    parser.add_argument("--data_root", type=str,
                       default="/root/autodl-tmp/fyp_final/Ver4/Trajectory/preprocessed_era5")
    parser.add_argument("--norm_stats", type=str,
                       default="/root/autodl-tmp/fyp_final/Ver4/Trajectory/preprocessed_era5/norm_stats.pt")
    parser.add_argument("--output_dir", type=str, default="channel_plots")
    parser.add_argument("--num_samples", type=int, default=40)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--euler_steps", type=int, default=4)
    parser.add_argument("--ddim_steps", type=int, default=50)

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    channel_names = [f"{v}_{l}" for v in ["u", "v", "z"] for l in [850, 500, 250]]
    z_names = ["z_850", "z_500", "z_250"]
    z_indices = [6, 7, 8]  # z_850, z_500, z_250

    # Config
    data_cfg = DataConfig(
        data_root=args.data_root, era5_dir=args.data_root,
        preprocessed_dir=args.data_root, norm_stats_path=args.norm_stats,
        history_steps=5, forecast_steps=1, grid_size=40,
        pressure_level_vars=["u", "v", "z"], pressure_levels=[850, 500, 250],
        surface_vars=[], num_workers=4, pin_memory=True,
    )
    # Separate model configs for FM and DM — they were trained with DIFFERENT architectures!
    # FM checkpoint: use_grouped_conv=False, Conv3D temporal agg (temporal_conv3d shape [9,9,1,3,3])
    # DM checkpoint: use_grouped_conv=True, Conv2D local agg (temporal_conv3d shape [45,45,3,3])
    fm_model_cfg = ModelConfig(
        in_channels=data_cfg.num_channels, cond_channels=data_cfg.condition_channels,
        d_model=384, n_heads=6, n_dit_layers=12, n_cond_layers=3,
        ff_mult=4, patch_size=4, dropout=0.1,
        use_grouped_conv=False,  # FM: Conv3D temporal agg
        num_var_groups=3, time_embedding_scale=1000.0,
    )
    dm_model_cfg = ModelConfig(
        in_channels=data_cfg.num_channels, cond_channels=data_cfg.condition_channels,
        d_model=384, n_heads=6, n_dit_layers=12, n_cond_layers=3,
        ff_mult=4, patch_size=4, dropout=0.1,
        use_grouped_conv=True,   # DM: Conv2D local + Conv3D temporal agg
        num_var_groups=3, time_embedding_scale=1000.0,
    )
    train_cfg = TrainConfig(batch_size=8, seed=42, use_channel_weights=True)

    # Use newtry's own norm_stats.pt for DM evaluation (matches the DM model's training)
    newtry_norm_stats = "/root/autodl-tmp/fyp_final/Ver4/newtry/norm_stats_v.pt"
    # Load the DM normalization stats
    dm_stats = torch.load(newtry_norm_stats, weights_only=True)
    dm_norm_mean = dm_stats["mean"].numpy()
    dm_norm_std = dm_stats["std"].numpy()
    logger.info(f"Loaded newtry norm stats: std[:3]={dm_norm_std[:3]}")

    # For FM, use Trajectory norm_stats (matching the FM checkpoint training)
    fm_norm_stats = "/root/autodl-tmp/fyp_final/Ver4/Trajectory/norm_stats.pt"
    fm_stats = torch.load(fm_norm_stats, weights_only=True)
    fm_norm_mean = fm_stats["mean"].numpy()
    fm_norm_std = fm_stats["std"].numpy()
    logger.info(f"Loaded FM norm stats: std[:3]={fm_norm_std[:3]}")
    _, _, test_loader, norm_mean, norm_std = build_dataloaders(data_cfg, train_cfg)
    if isinstance(norm_mean, torch.Tensor): norm_mean = norm_mean.numpy()
    if isinstance(norm_std, torch.Tensor): norm_std = norm_std.numpy()
    logger.info(f"Test set: {len(test_loader.dataset)} samples, norm_std[:3]={norm_std[:3]}")

    # Load FM
    logger.info(f"Loading FM model: {args.fm_ckpt}")
    fm_model = UnifiedModel(fm_model_cfg, data_cfg, train_cfg, method="fm").to(device)
    fm_model, loaded = load_checkpoint(fm_model, args.fm_ckpt, device)
    if not loaded:
        logger.error(f"FM checkpoint not found: {args.fm_ckpt}")
        return
    fm_model.eval()

    # Use the dedicated newtry DM checkpoint (best_v.pt) via the adapter.
    # The DM checkpoint in multi_seed_results/ was trained with a different architecture
    # and can't be loaded with the current code.
    newtry_dm_path = "/root/autodl-tmp/fyp_final/Ver4/newtry/checkpoints/best_v.pt"
    logger.info(f"Loading DM model (newtry) from: {newtry_dm_path}")
    dm_model_cfg = ModelConfig(
        in_channels=data_cfg.num_channels, cond_channels=data_cfg.condition_channels,
        d_model=384, n_heads=6, n_dit_layers=12, n_cond_layers=3,
        ff_mult=4, patch_size=4, dropout=0.1,
        prediction_type="v",
    )
    dm_model = load_newtry_checkpoint(newtry_dm_path, data_cfg, dm_model_cfg, train_cfg, device)
    dm_model.eval()

    # Inference
    logger.info(f"Running inference on {args.num_samples} samples...")
    all_fm, all_dm, all_gt = [], [], []
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            if batch_idx * train_cfg.batch_size >= args.num_samples:
                break
            cond = batch["condition"].to(device)
            target = batch["target"].to(device)

            fm_pred = fm_model.sample_fm(cond, device,
                euler_steps=args.euler_steps, euler_mode="midpoint",
                clamp_range=None, z_clamp_range=None)
            # AdaptedDiffusionModel uses sample(), condition is already 4D (B, T*C, H, W)
            dm_pred = dm_model.sample(cond, device,
                ddim_steps=args.ddim_steps,
                clamp_range=None, z_clamp_range=None)

            all_fm.append(fm_pred.cpu().numpy())
            all_dm.append(dm_pred.cpu().numpy())
            all_gt.append(target.cpu().numpy())

    fm_preds = np.concatenate(all_fm, axis=0)
    dm_preds = np.concatenate(all_dm, axis=0)
    gts = np.concatenate(all_gt, axis=0)
    logger.info(f"Collected {len(fm_preds)} samples")

    # Denormalize
    fm_dn = denormalize(fm_preds, norm_mean, norm_std)
    dm_dn = denormalize(dm_preds, norm_mean, norm_std)
    gt_dn = denormalize(gts, norm_mean, norm_std)

    # Sanity check
    logger.info("=== Denormalized sanity check (first sample) ===")
    for ch in range(9):
        logger.info(f"  {channel_names[ch]}: GT={gt_dn[0,ch].mean():8.2f}, "
                    f"FM={fm_dn[0,ch].mean():8.2f}, DM={dm_dn[0,ch].mean():8.2f}")

    # Compute all stats
    logger.info("Computing statistics...")
    stats = compute_all_channel_stats(fm_dn, dm_dn, gt_dn, channel_names)
    diagnostics = compute_z_diagnostics(fm_dn, dm_dn, gt_dn, z_indices, z_names)

    # Print summary
    print_diagnostic_summary(stats, diagnostics, z_names)

    # Generate plots
    logger.info("Generating plots...")

    # 1. Comprehensive channel stats
    plot_comprehensive_channel_stats(
        stats,
        save_path=os.path.join(args.output_dir, "channel_stats_v4.png")
    )

    # 2. Z-channel spatial correlation maps
    plot_z_channel_spatial_correlation_maps(
        fm_dn, dm_dn, gt_dn, z_indices, z_names,
        save_path=os.path.join(args.output_dir, "channel_z_spatial_corr_v4.png")
    )

    # 3. Z-channel spatial maps
    plot_z_channel_spatial_maps(
        fm_dn, dm_dn, gt_dn, z_indices, z_names,
        save_path=os.path.join(args.output_dir, "channel_z_maps_v4.png")
    )

    # 4. Z-channel diagnostics table
    plot_z_diagnostics_table(
        diagnostics, z_names,
        save_path=os.path.join(args.output_dir, "channel_z_diagnostics_v4.png")
    )

    # 5. All-channel spatial maps (GT vs FM vs DM)
    n_show = min(8, len(fm_preds))
    fig, axes = plt.subplots(9, 4, figsize=(16, 3.5 * 9))
    for ch_idx, ch_name in enumerate(channel_names):
        best_s = int(np.argmax([gt_dn[i, ch_idx].std() for i in range(n_show)]))
        gt_s = gt_dn[best_s, ch_idx]
        fm_s = fm_dn[best_s, ch_idx]
        dm_s = dm_dn[best_s, ch_idx]
        all_d = np.stack([gt_s, fm_s, dm_s])
        vmin, vmax = np.percentile(all_d, [1, 99])
        fm_err = fm_s - gt_s
        dm_err = dm_s - gt_s

        for col, (data, title) in enumerate([
            (gt_s, "GT"),
            (fm_s, "FM"),
            (dm_s, "DM"),
        ]):
            ax = axes[ch_idx, col]
            im = ax.imshow(data, cmap='viridis', vmin=vmin, vmax=vmax, origin='lower')
            ax.set_title(f"{title}: {ch_name}", fontsize=9)
            ax.set_xticks([]); ax.set_yticks([])
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, fontsize=7)

        ax = axes[ch_idx, 3]
        err_max = max(np.abs(fm_err).max(), np.abs(dm_err).max(), 0.01)
        norm = TwoSlopeNorm(vmin=-err_max, vcenter=0, vmax=err_max)
        ax.imshow(fm_err, cmap='RdBu_r', norm=norm, origin='lower')
        ax.set_title(f"FM Err RMSE={np.sqrt((fm_err**2).mean()):.1f}", fontsize=8)
        ax.set_xticks([]); ax.set_yticks([])

    plt.suptitle("All Channels — GT vs FM vs DM (highest-variance sample)",
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "channel_comparison_v4.png"),
                dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    logger.info(f"Saved: channel_comparison_v4.png")

    logger.info(f"\nAll plots saved to: {args.output_dir}/")
    logger.info("Done!")


if __name__ == "__main__":
    main()
