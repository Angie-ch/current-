"""
ERA5 Prediction Visualization - FM Only (v3 best checkpoint)

Plots FM predictions vs ground truth for each channel.
Uses the new multi_seed_results_fixed_long checkpoint.

Usage:
    python plot_fm_v3_best.py --output_dir fm_v3_plots
"""
import os
import sys
import argparse
import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import torch
from torch.utils.data import DataLoader

current_file = os.path.abspath(__file__)
package_dir = os.path.dirname(current_file)
root_dir = os.path.dirname(package_dir)
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)
if package_dir not in sys.path:
    sys.path.insert(0, package_dir)

from configs import DataConfig, ModelConfig, TrainConfig
from data.dataset import build_dataloaders
from models.unified_model import UnifiedModel
from models.trainer import EMA

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

COLORS = {"FM": "#2E86AB", "GT": "#2D2D2D"}


def load_checkpoint(model, checkpoint_path, device, use_ema=True):
    """Load checkpoint with EMA weight application."""
    if not os.path.exists(checkpoint_path):
        return model, False

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    sd = ckpt["model_state_dict"]
    ema_shadow = ckpt.get("ema_state_dict", {}).get("shadow", {})

    result = model.load_state_dict(sd, strict=False)
    unexpected = len(result.unexpected_keys)

    ema_applied = 0
    for name, param in model.named_parameters():
        if name in ema_shadow:
            param.data.copy_(ema_shadow[name].to(param.device, param.dtype))
            ema_applied += 1

    total_params = len(list(model.named_parameters()))
    logger.info(f"Loaded checkpoint: {checkpoint_path}")
    val_loss = ckpt.get("best_val_loss", float("nan"))
    epoch = ckpt.get("epoch", "N/A")
    logger.info(f"  Val loss: {val_loss:.4f}" if isinstance(val_loss, float) else f"  Val loss: {val_loss}")
    logger.info(f"  Epoch: {epoch}")
    logger.info(f"  Model params: {total_params}, EMA applied: {ema_applied}/{total_params}")
    if ema_applied == 0:
        logger.warning("  WARNING: EMA was NOT applied!")
    if unexpected > 0:
        logger.warning(f"  Unexpected keys (ignored): {unexpected}")

    return model, True


def denormalize(data, mean, std):
    """Denormalize data using mean and std."""
    std = np.where(std < 1e-8, 1.0, std)
    mean = mean[np.newaxis, :, np.newaxis, np.newaxis]
    std = std[np.newaxis, :, np.newaxis, np.newaxis]
    return data * std + mean


def plot_channel_comparison_fm_only(
    pred_fm: np.ndarray,
    ground_truth: np.ndarray,
    channel_names: List[str],
    sample_idx: int = 0,
    save_path: str = "channel_comparison.png",
):
    """Plot FM predictions vs ground truth for all channels."""
    n_channels = len(channel_names)

    if pred_fm.ndim == 4:
        pred_fm = pred_fm[sample_idx]
    if ground_truth.ndim == 4:
        ground_truth = ground_truth[sample_idx]

    fig, axes = plt.subplots(n_channels, 3, figsize=(12, 3.5 * n_channels))

    for ch_idx, ch_name in enumerate(channel_names):
        gt_data = ground_truth[ch_idx]
        fm_data = pred_fm[ch_idx]

        all_data = np.stack([gt_data, fm_data])
        vmin, vmax = np.percentile(all_data, [1, 99])

        fm_error = fm_data - gt_data

        for col, (data, title) in enumerate([
            (gt_data, f"GT: {ch_name}"),
            (fm_data, "FM Prediction"),
        ]):
            ax = axes[ch_idx, col] if n_channels > 1 else axes[col]
            im = ax.imshow(data, cmap='RdBu_r', vmin=vmin, vmax=vmax, origin='lower')
            ax.set_title(title, fontsize=10, fontweight='bold')
            ax.set_xticks([])
            ax.set_yticks([])
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        ax = axes[ch_idx, 2] if n_channels > 1 else axes[2]
        err_max = max(np.abs(fm_error).max(), 0.01)
        norm = TwoSlopeNorm(vmin=-err_max, vcenter=0, vmax=err_max)
        ax.imshow(fm_error, cmap='RdBu_r', norm=norm, origin='lower')
        fm_rmse = np.sqrt((fm_error**2).mean())
        fm_bias = fm_error.mean()
        ax.set_title(f"FM Error\nRMSE={fm_rmse:.3f}, Bias={fm_bias:.3f}", fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])

    for col_idx, col_name in enumerate(["Ground Truth", "FM Prediction", "Error"]):
        axes[0, col_idx].annotate(
            col_name, xy=(0.5, 1.04), xycoords='axes fraction',
            ha='center', va='bottom', fontsize=13, fontweight='bold',
        )

    plt.suptitle("ERA5 Predictions: FM vs Ground Truth (v3 best)",
                 fontsize=15, fontweight='bold', y=1.005)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    logger.info(f"Saved: {save_path}")


def plot_per_channel_stats_fm(
    pred_fm: np.ndarray,
    ground_truth: np.ndarray,
    channel_names: List[str],
    save_path: str = "channel_stats.png",
):
    """Plot RMSE and bias statistics per channel."""
    n_channels = len(channel_names)

    if pred_fm.ndim == 4:
        pred_fm = pred_fm.mean(axis=0)
    if ground_truth.ndim == 4:
        ground_truth = ground_truth.mean(axis=0)

    fm_rmse_list, fm_bias_list, fm_std_ratio_list = [], [], []
    for ch_idx in range(n_channels):
        gt_ch = ground_truth[ch_idx]
        fm_ch = pred_fm[ch_idx]
        fm_rmse_list.append(np.sqrt(((fm_ch - gt_ch)**2).mean()))
        fm_bias_list.append((fm_ch - gt_ch).mean())
        gt_std = gt_ch.std() if gt_ch.std() > 1e-8 else 1.0
        fm_std = fm_ch.std() if fm_ch.std() > 1e-8 else 1.0
        fm_std_ratio_list.append(fm_std / gt_std)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    x = np.arange(n_channels)
    width = 0.6

    ax = axes[0]
    ax.bar(x, fm_rmse_list, width, label='FM', color=COLORS["FM"], alpha=0.8)
    ax.set_xlabel('Channel', fontsize=12)
    ax.set_ylabel('RMSE', fontsize=12)
    ax.set_title('Per-Channel RMSE', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(channel_names, rotation=45, ha='right')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')

    ax = axes[1]
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.bar(x, fm_bias_list, width, label='FM Bias', color=COLORS["FM"], alpha=0.8)
    ax.set_xlabel('Channel', fontsize=12)
    ax.set_ylabel('Bias (Mean Error)', fontsize=12)
    ax.set_title('Per-Channel Bias', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(channel_names, rotation=45, ha='right')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')

    ax = axes[2]
    ax.axhline(y=1.0, color='red', linestyle='--', linewidth=1.0, label='Perfect (1.0)')
    ax.bar(x, fm_std_ratio_list, width, label='FM Std Ratio', color=COLORS["FM"], alpha=0.8)
    ax.set_xlabel('Channel', fontsize=12)
    ax.set_ylabel('Std Ratio (Pred/GT)', fontsize=12)
    ax.set_title('Variance Preservation', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(channel_names, rotation=45, ha='right')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim([0.5, 1.5])

    plt.suptitle("FM Per-Channel Statistics (v3 best)", fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    logger.info(f"Saved: {save_path}")


def plot_difference_maps_fm(
    pred_fm: np.ndarray,
    ground_truth: np.ndarray,
    channel_names: List[str],
    save_path: str = "difference_maps.png",
):
    """Plot FM-GT difference maps for all channels."""
    n_channels = len(channel_names)

    if pred_fm.ndim == 4:
        pred_fm = pred_fm.mean(axis=0)
    if ground_truth.ndim == 4:
        ground_truth = ground_truth.mean(axis=0)

    fig, axes = plt.subplots(n_channels, 2, figsize=(8, 3.8 * n_channels))

    for ch_idx, ch_name in enumerate(channel_names):
        gt_ch = ground_truth[ch_idx]
        fm_ch = pred_fm[ch_idx]
        fm_error = fm_ch - gt_ch

        err_max = max(np.abs(fm_error).max(), 0.01)
        norm = TwoSlopeNorm(vmin=-err_max, vcenter=0, vmax=err_max)

        ax = axes[ch_idx, 0]
        im = ax.imshow(fm_ch, cmap='RdBu_r', origin='lower')
        ax.set_title(f"FM: {ch_name}", fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        ax = axes[ch_idx, 1]
        im = ax.imshow(fm_error, cmap='RdBu_r', norm=norm, origin='lower')
        rmse_val = np.sqrt((fm_error**2).mean())
        ax.set_title(f"FM Error: {ch_name} (RMSE={rmse_val:.3f})", fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        axes[ch_idx, 0].set_ylabel(ch_name, fontsize=10, rotation=0, ha='right', va='center')

    plt.suptitle("FM Prediction Error Maps (v3 best)", fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    logger.info(f"Saved: {save_path}")


def plot_samples_grid_fm(
    pred_fm: np.ndarray,
    ground_truth: np.ndarray,
    channel_names: List[str],
    n_samples: int = 4,
    save_path: str = "samples_grid.png",
):
    """Plot a compact grid of samples x channels."""
    n_channels = min(6, len(channel_names))
    n_show = min(n_samples, len(pred_fm))

    fig, axes = plt.subplots(
        n_show * 2, n_channels,
        figsize=(2.5 * n_channels, 2.5 * n_show * 2)
    )

    for sample_idx in range(n_show):
        gt_sample = ground_truth[sample_idx] if ground_truth.ndim == 4 else ground_truth
        fm_sample = pred_fm[sample_idx] if pred_fm.ndim == 4 else pred_fm

        all_data = np.stack([gt_sample[:n_channels], fm_sample[:n_channels]])
        vmin, vmax = np.percentile(all_data, [1, 99])

        for ch_idx in range(n_channels):
            for row_i, (data, label, color) in enumerate([
                (gt_sample[ch_idx], "GT", COLORS["GT"]),
                (fm_sample[ch_idx], "FM", COLORS["FM"]),
            ]):
                row = sample_idx * 2 + row_i
                ax = axes[row, ch_idx]
                ax.imshow(data, cmap='RdBu_r', vmin=vmin, vmax=vmax, origin='lower')
                ax.set_xticks([])
                ax.set_yticks([])
                if ch_idx == 0:
                    ax.set_ylabel(f"{label}\nS{sample_idx}", fontsize=8, color=color)
                if sample_idx == 0 and row_i == 0:
                    ax.set_title(channel_names[ch_idx], fontsize=10, fontweight='bold')

    plt.suptitle("Ground Truth | FM — Sample × Channel Grid (v3 best)",
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    logger.info(f"Saved: {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot ERA5 FM predictions (v3 best)")
    parser.add_argument("--fm_ckpt", type=str,
                       default="multi_seed_results_fixed_long/seed_42/checkpoints_fm/best.pt")
    parser.add_argument("--data_root", type=str,
                       default="/root/autodl-tmp/fyp_final/Ver4/Trajectory/preprocessed_era5")
    parser.add_argument("--norm_stats", type=str,
                       default="/root/autodl-tmp/fyp_final/Ver4/Trajectory/preprocessed_era5/norm_stats_year_split.pt")
    parser.add_argument("--output_dir", type=str, default="fm_v3_plots")
    parser.add_argument("--num_samples", type=int, default=20)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--euler_steps", type=int, default=4)

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    channel_names = [f"{v}_{l}" for v in ["u", "v", "z"] for l in [850, 500, 250]]

    data_cfg = DataConfig(
        norm_stats_path=args.norm_stats,
        data_root=args.data_root,
        era5_dir=args.data_root,
        preprocessed_dir=args.data_root,
        history_steps=5,
        forecast_steps=1,
        grid_size=40,
        pressure_level_vars=["u", "v", "z"],
        pressure_levels=[850, 500, 250],
        surface_vars=[],
        num_workers=4,
        pin_memory=True,
    )

    model_cfg = ModelConfig(
        in_channels=data_cfg.num_channels,
        cond_channels=data_cfg.condition_channels,
        d_model=384,
        n_heads=6,
        n_dit_layers=12,
        n_cond_layers=3,
        ff_mult=4,
        patch_size=4,
        dropout=0.1,
        use_grouped_conv=False,
        num_var_groups=3,
        time_embedding_scale=1000.0,
    )

    train_cfg = TrainConfig(
        batch_size=8,
        seed=42,
        use_channel_weights=True,
    )

    logger.info("Loading dataset...")
    _, _, test_loader, norm_mean, norm_std = build_dataloaders(data_cfg, train_cfg)
    logger.info(f"Test set size: {len(test_loader.dataset)}")

    if isinstance(norm_mean, torch.Tensor):
        norm_mean = norm_mean.numpy()
    if isinstance(norm_std, torch.Tensor):
        norm_std = norm_std.numpy()

    logger.info(f"Loading FM model: {args.fm_ckpt}")
    fm_model = UnifiedModel(model_cfg, data_cfg, train_cfg, method="fm").to(device)
    fm_model, loaded = load_checkpoint(fm_model, args.fm_ckpt, device, use_ema=True)
    if not loaded:
        logger.warning(f"FM checkpoint not found: {args.fm_ckpt}")
        return

    fm_model.eval()

    all_fm_preds, all_gts = [], []

    logger.info(f"Running inference on {args.num_samples} samples...")
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            if batch_idx * train_cfg.batch_size >= args.num_samples:
                break

            condition = batch["condition"].to(device)
            target = batch["target"].to(device)

            fm_pred = fm_model.sample_fm(
                condition, device,
                euler_steps=args.euler_steps,
                euler_mode="midpoint",
                clamp_range=None, z_clamp_range=None)

            all_fm_preds.append(fm_pred.cpu().numpy())
            all_gts.append(target.cpu().numpy())

    fm_preds = np.concatenate(all_fm_preds, axis=0)
    gts = np.concatenate(all_gts, axis=0)

    logger.info(f"Collected predictions: {fm_preds.shape}")

    # Denormalize
    fm_preds_denorm = denormalize(fm_preds, norm_mean, norm_std)
    gts_denorm = denormalize(gts, norm_mean, norm_std)

    # Sanity check
    logger.info("=== Sanity Check (first sample, denormalized) ===")
    for ch in range(9):
        logger.info(f"  {channel_names[ch]}: GT={gts_denorm[0,ch].mean():8.2f}, "
                    f"FM={fm_preds_denorm[0,ch].mean():8.2f}")

    # Generate plots
    logger.info("Generating plots...")
    plot_channel_comparison_fm_only(
        fm_preds_denorm, gts_denorm,
        channel_names=channel_names,
        sample_idx=0,
        save_path=os.path.join(args.output_dir, "channel_comparison.png"),
    )
    plot_per_channel_stats_fm(
        fm_preds_denorm, gts_denorm,
        channel_names=channel_names,
        save_path=os.path.join(args.output_dir, "channel_stats.png"),
    )
    plot_difference_maps_fm(
        fm_preds_denorm, gts_denorm,
        channel_names=channel_names,
        save_path=os.path.join(args.output_dir, "difference_maps.png"),
    )
    plot_samples_grid_fm(
        fm_preds_denorm, gts_denorm,
        channel_names=channel_names,
        n_samples=min(4, len(fm_preds)),
        save_path=os.path.join(args.output_dir, "samples_grid.png"),
    )

    logger.info(f"All plots saved to: {args.output_dir}/")


if __name__ == "__main__":
    main()
