"""
Geostrophic Balance Evaluation — Flow Matching vs Diffusion Paper

Computes:
1. Geostrophic imbalance: ||u - u_geo||² + ||v - v_geo||² at 500 hPa
2. Divergence RMSE: ||∇·V|| (mass continuity)
3. Per-lead-time imbalance growth (24h, 48h, 72h)
4. Comparison between DM and FM

This script evaluates whether FM or DM better preserves the physical relationship
between wind and pressure fields. FM's ODE-based transport should maintain
better geostrophic balance than DM's stochastic denoising.

Usage:
    python evaluation/geostrophic.py \
        --pred_dir table2_results/ \
        --output_dir evaluation/geostrophic_results/ \
        --method dm
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Callable

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "Trajectory"))

try:
    from compare_fm_dm.evaluation.metrics import compute_divergence, compute_geostrophic_balance
    from Trajectory.paper_eval_common import (
        traj_data_cfg,
        traj_model_cfg,
    )
except ImportError:
    from paper_eval_common import (
        traj_data_cfg,
        traj_model_cfg,
    )


# Physical constants
GRAVITY = 9.80665        # m/s²
EARTH_RADIUS = 6.371e6   # m
OMEGA = 7.2921e-5        # rad/s
DEG_TO_RAD = np.pi / 180.0


def compute_coriolis_parameter(
    lat: np.ndarray,
) -> np.ndarray:
    """
    Compute Coriolis parameter f = 2Ω sin(lat).

    Args:
        lat: latitude in degrees, shape (H,) or scalar

    Returns:
        f: Coriolis parameter in s⁻¹
    """
    lat_rad = np.asarray(lat) * DEG_TO_RAD
    f = 2 * OMEGA * np.sin(lat_rad)
    f = np.where(np.abs(f) < 1e-10, np.sign(f + 1e-10) * 1e-10, f)
    return f


def compute_central_difference_gradient(
    field: np.ndarray,
    resolution: float = 0.25,
    axis: int = -1,
) -> np.ndarray:
    """
    Compute central difference gradient on a 2D field.

    Args:
        field: (H, W) array
        resolution: grid spacing in degrees
        axis: gradient axis (-1 for x, -2 for y)

    Returns:
        gradient: same shape as input (edges are zero-padded)
    """
    H, W = field.shape
    grad = np.zeros_like(field)

    if axis == -1 or axis == 1:
        grad[:, 1:-1] = (field[:, 2:] - field[:, :-2]) / (2 * resolution)
    else:
        grad[1:-1, :] = (field[2:, :] - field[:-2, :]) / (2 * resolution)

    return grad


def compute_geostrophic_wind_from_z(
    z: np.ndarray,
    lat: np.ndarray,
    resolution: float = 0.25,
    pressure_level: int = 500,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute geostrophic wind from geopotential height field.

    u_geo = -g/f * dz/dy
    v_geo =  g/f * dz/dx

    Args:
        z: geopotential height anomaly (m), shape (H, W)
        lat: latitude for each grid row (degrees), shape (H,)
        resolution: grid spacing in degrees
        pressure_level: pressure level for label (unused, for reference)

    Returns:
        u_geo, v_geo: geostrophic wind components (m/s), shape (H, W)
    """
    g = GRAVITY
    f = compute_coriolis_parameter(lat)  # (H,)

    dzdx = compute_central_difference_gradient(z, resolution=resolution, axis=-1)
    dzdy = compute_central_difference_gradient(z, resolution=resolution, axis=-2)

    f_2d = f[:, np.newaxis]  # (H, 1)
    f_2d = np.where(np.abs(f_2d) < 1e-10, np.sign(f_2d + 1e-10) * 1e-10, f_2d)

    v_geo = (g * dzdx) / f_2d
    u_geo = -(g * dzdy) / f_2d

    return u_geo, v_geo


def compute_geostrophic_imbalance_torch(
    u: torch.Tensor,
    v: torch.Tensor,
    z: torch.Tensor,
    lat: torch.Tensor,
    resolution: float = 0.25,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute geostrophic imbalance using torch (GPU-accelerated).

    Args:
        u, v, z: (B, C, H, W) tensors
        lat: (H,) tensor of latitudes in degrees
        resolution: grid spacing in degrees

    Returns:
        u_geo, v_geo, imbalance: (B, C-1, H-1, W-1) each
    """
    g = GRAVITY

    f = 2 * OMEGA * torch.sin(lat * DEG_TO_RAD).to(u.device)
    f = torch.where(f.abs() < 1e-10, torch.sign(f + 1e-10) * 1e-10, f)
    f = f.view(1, 1, -1, 1)

    dzdx = (z[:, :, :, 2:] - z[:, :, :, :-2]) / (2 * resolution)
    dzdy = (z[:, :, 2:, :] - z[:, :, :-2, :]) / (2 * resolution)

    u_geo = -(g * dzdy) / (f[:, :, 1:-1, 1:-1] + 1e-8)
    v_geo =  (g * dzdx) / (f[:, :, 1:-1, 1:-1] + 1e-8)

    u_crop = u[:, :, 1:-1, 1:-1]
    v_crop = v[:, :, 1:-1, 1:-1]

    imbalance = ((u_crop - u_geo) ** 2 + (v_crop - v_geo) ** 2).mean(dim=(0, 2, 3))

    return u_geo, v_geo, imbalance


def compute_divergence_torch(
    u: torch.Tensor,
    v: torch.Tensor,
    resolution: float = 0.25,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute divergence ∇·V = ∂u/∂x + ∂v/∂y using torch.

    Args:
        u, v: (B, C, H, W) wind components
        resolution: grid spacing in degrees

    Returns:
        divergence: (B, C, H-1, W-1)
        div_rmse: scalar RMSE over all samples
    """
    du_dx = (u[:, :, :, 1:] - u[:, :, :, :-1]) / resolution
    dv_dy = (v[:, :, 1:, :] - v[:, :, :-1, :]) / resolution

    du_dx = du_dx[:, :, :-1, :]
    dv_dy = dv_dy[:, :, :, :-1]

    div = du_dx + dv_dy

    return div, div.abs().mean()


def geostrophic_analysis_single_case(
    pred_u: np.ndarray,
    pred_v: np.ndarray,
    pred_z: np.ndarray,
    gt_u: np.ndarray,
    gt_v: np.ndarray,
    gt_z: np.ndarray,
    lat: np.ndarray,
    resolution: float = 0.25,
) -> Dict[str, float]:
    """
    Compute geostrophic imbalance for a single case.

    Args:
        pred_u, pred_v, pred_z: (H, W) predicted wind and geopotential
        gt_u, gt_v, gt_z: (H, W) ground truth
        lat: (H,) latitude values in degrees
        resolution: grid spacing in degrees

    Returns:
        Dict with imbalance metrics
    """
    u_geo_pred, v_geo_pred = compute_geostrophic_wind_from_z(pred_z, lat, resolution)
    u_geo_gt, v_geo_gt = compute_geostrophic_wind_from_z(gt_z, lat, resolution)

    imbalance_pred = float(
        ((pred_u - u_geo_pred) ** 2 + (pred_v - v_geo_pred) ** 2).mean()
    )
    imbalance_gt = float(
        ((gt_u - u_geo_gt) ** 2 + (gt_v - v_geo_gt) ** 2).mean()
    )

    div_pred = compute_divergence_np(pred_u, pred_v, resolution)
    div_gt = compute_divergence_np(gt_u, gt_v, resolution)

    div_rmse_pred = float(np.sqrt((div_pred ** 2).mean()))
    div_rmse_gt = float(np.sqrt((div_gt ** 2).mean()))

    return {
        "imbalance_pred": imbalance_pred,
        "imbalance_gt": imbalance_gt,
        "imbalance_improvement": imbalance_gt - imbalance_pred,
        "div_rmse_pred": div_rmse_pred,
        "div_rmse_gt": div_rmse_gt,
        "div_rmse_improvement": div_rmse_gt - div_rmse_pred,
    }


def compute_divergence_np(
    u: np.ndarray,
    v: np.ndarray,
    resolution: float = 0.25,
) -> np.ndarray:
    """Compute divergence field using numpy."""
    du_dx = (u[:, 1:] - u[:, :-1]) / resolution
    dv_dy = (v[1:, :] - v[:-1, :]) / resolution

    min_H = min(du_dx.shape[0], dv_dy.shape[0])
    min_W = min(du_dx.shape[1], dv_dy.shape[1])

    du_dx = du_dx[:min_H, :min_W]
    dv_dy = dv_dy[:min_H, :min_W]

    return du_dx + dv_dy


def geostrophic_eval_from_predictions_torch(
    pred_samples: torch.Tensor,
    gt_sample: torch.Tensor,
    lat_array: np.ndarray,
    resolution: float = 0.25,
    u_channel_idx: int = 0,
    v_channel_idx: int = 3,
    z_channel_idx: int = 7,
) -> Dict[str, Dict[str, float]]:
    """
    Main geostrophic evaluation using torch for GPU acceleration.

    Args:
        pred_samples: (N, C, H, W) — N predicted samples at a single time step
        gt_sample: (C, H, W) — ground truth at that time step
        lat_array: (H,) latitudes in degrees
        resolution: grid spacing in degrees
        u_channel_idx: index of u component (default 0 = u_850)
        v_channel_idx: index of v component (default 3 = v_850)
        z_channel_idx: index of z component (default 7 = z_500)

    Returns:
        Dict with imbalance, divergence, and wind-pressure metrics
    """
    device = pred_samples.device
    lat_t = torch.from_numpy(lat_array).float().to(device)

    u = pred_samples[:, u_channel_idx].unsqueeze(1)
    v = pred_samples[:, v_channel_idx].unsqueeze(1)
    z = pred_samples[:, z_channel_idx].unsqueeze(1)

    gt_u = gt_sample[u_channel_idx].unsqueeze(0).unsqueeze(0)
    gt_v = gt_sample[v_channel_idx].unsqueeze(0).unsqueeze(0)
    gt_z = gt_sample[z_channel_idx].unsqueeze(0).unsqueeze(0)

    u_geo, v_geo, imbalance = compute_geostrophic_imbalance_torch(
        u, v, z, lat_t, resolution=resolution
    )
    _, _, gt_imbalance = compute_geostrophic_imbalance_torch(
        gt_u, gt_v, gt_z, lat_t, resolution=resolution
    )

    div_pred, div_rmse_pred = compute_divergence_torch(u, v, resolution=resolution)
    div_gt, div_rmse_gt = compute_divergence_torch(gt_u, gt_v, resolution=resolution)

    return {
        "imbalance_pred": float(imbalance.mean().item()),
        "imbalance_pred_std": float(imbalance.std().item()),
        "imbalance_gt": float(gt_imbalance.mean().item()),
        "div_rmse_pred": float(div_rmse_pred.item()),
        "div_rmse_gt": float(div_rmse_gt.item()),
        "div_improvement": float((div_rmse_gt - div_rmse_pred).item()),
        "imbalance_improvement": float((gt_imbalance.mean() - imbalance.mean()).item()),
    }


def aggregate_geostrophic_results(
    per_case_results: List[Dict],
    lead_times: List[int],
) -> Dict:
    """Aggregate geostrophic results across cases and lead times."""
    aggregated = {}

    for lead in lead_times:
        imb_list = [r.get(f"imbalance_{lead}h", np.nan) for r in per_case_results]
        div_list = [r.get(f"div_rmse_{lead}h", np.nan) for r in per_case_results]
        imp_list = [r.get(f"imbalance_improvement_{lead}h", np.nan) for r in per_case_results]

        imb_arr = np.array([x for x in imb_list if not np.isnan(x)])
        div_arr = np.array([x for x in div_list if not np.isnan(x)])
        imp_arr = np.array([x for x in imp_list if not np.isnan(x)])

        aggregated[f"imbalance_{lead}h"] = {
            "mean": float(imb_arr.mean()) if len(imb_arr) > 0 else np.nan,
            "std": float(imb_arr.std()) if len(imb_arr) > 0 else np.nan,
            "n": len(imb_arr),
        }
        aggregated[f"div_rmse_{lead}h"] = {
            "mean": float(div_arr.mean()) if len(div_arr) > 0 else np.nan,
            "std": float(div_arr.std()) if len(div_arr) > 0 else np.nan,
        }
        aggregated[f"improvement_{lead}h"] = {
            "mean": float(imp_arr.mean()) if len(imp_arr) > 0 else np.nan,
        }

    return aggregated


def generate_geostrophic_comparison_table(
    dm_results: Dict,
    fm_results: Dict,
    lead_times: List[int],
) -> str:
    """Generate formatted comparison table."""
    header = (
        f"{'Lead Time':>12} "
        f"{'DM Imbalance':>14} "
        f"{'FM Imbalance':>14} "
        f"{'DM Div RMSE':>14} "
        f"{'FM Div RMSE':>14} "
        f"{'Improv %':>10}"
    )
    divider = "-" * len(header)
    lines = [header, divider]

    for lead in lead_times:
        dm_imb = dm_results.get(f"imbalance_{lead}h", {}).get("mean", np.nan)
        fm_imb = fm_results.get(f"imbalance_{lead}h", {}).get("mean", np.nan)
        dm_div = dm_results.get(f"div_rmse_{lead}h", {}).get("mean", np.nan)
        fm_div = fm_results.get(f"div_rmse_{lead}h", {}).get("mean", np.nan)

        if not np.isnan(dm_imb) and not np.isnan(fm_imb) and dm_imb > 0:
            improvement = (dm_imb - fm_imb) / dm_imb * 100
        else:
            improvement = np.nan

        lines.append(
            f"{lead:>12}h "
            f"{(dm_imb if not np.isnan(dm_imb) else 0):>14.4f} "
            f"{(fm_imb if not np.isnan(fm_imb) else 0):>14.4f} "
            f"{(dm_div if not np.isnan(dm_div) else 0):>14.4f} "
            f"{(fm_div if not np.isnan(fm_div) else 0):>14.4f} "
            f"{(improvement if not np.isnan(improvement) else 0):>9.1f}%"
        )

    return "\n".join(lines)


def run_geostrophic_eval_from_directory(
    pred_dir: str,
    method: str = "dm",
    u_channel_idx: int = 0,
    v_channel_idx: int = 3,
    z_channel_idx: int = 7,
    resolution: float = 0.25,
    lead_times: List[int] = None,
    device: str = "cuda",
) -> List[Dict]:
    """
    Run geostrophic evaluation on all prediction files in a directory.

    Args:
        pred_dir: directory containing prediction .pt files
        method: "dm" or "fm"
        channel indices: which channels correspond to u, v, z
        resolution: grid spacing in degrees
        lead_times: list of lead times in hours to evaluate
        device: computation device

    Returns:
        List of per-case results
    """
    if lead_times is None:
        lead_times = [24, 48, 72]

    lead_time_indices = [
        lt // traj_data_cfg.time_resolution_hours - 1
        for lt in lead_times
    ]

    pred_path = Path(pred_dir)
    pt_files = sorted(pred_path.glob("ar_pred_*.pt"))
    if not pt_files:
        pt_files = sorted(pred_path.glob("*.pt"))

    lat_array = np.linspace(
        traj_data_cfg.lat_range[0],
        traj_data_cfg.lat_range[1],
        traj_data_cfg.grid_height,
    )

    results = []

    for pt_file in pt_files:
        data = torch.load(pt_file, map_location=device, weights_only=False)

        pred = None
        gt = None
        for key in ("predictions", "prediction", "prediction_samples"):
            if key in data:
                pred = data[key]
                break
        for key in ("ground_truth", "gt", "target"):
            if key in data:
                gt = data[key]
                break

        if pred is None or gt is None:
            continue

        case_result = {"case_key": pt_file.stem}

        for step_idx, lead in zip(lead_time_indices, lead_times):
            if step_idx >= pred.shape[1]:
                continue

            pred_step = pred[:, step_idx].to(device)
            gt_step = gt[step_idx].to(device)

            eval_result = geostrophic_eval_from_predictions_torch(
                pred_step.unsqueeze(1).expand(-1, pred.shape[1] if pred.ndim == 5 else 1, -1, -1, -1),
                gt_step,
                lat_array,
                resolution=resolution,
                u_channel_idx=u_channel_idx,
                v_channel_idx=v_channel_idx,
                z_channel_idx=z_channel_idx,
            )

            for k, v in eval_result.items():
                case_result[f"{k}_{lead}h"] = v

        results.append(case_result)

    return results


def parse_args():
    parser = argparse.ArgumentParser(
        description="Geostrophic Balance Evaluation: FM vs DM physical consistency"
    )
    parser.add_argument("--pred_dir", type=str, required=True,
                        help="Directory containing prediction .pt files")
    parser.add_argument("--output_dir", type=str,
                        default="evaluation/geostrophic_results",
                        help="Output directory")
    parser.add_argument("--method", type=str, choices=["dm", "fm"], default="dm",
                        help="Method being evaluated")
    parser.add_argument("--u_channel_idx", type=int, default=0,
                        help="U wind channel index (default: 0 = u_850)")
    parser.add_argument("--v_channel_idx", type=int, default=3,
                        help="V wind channel index (default: 3 = v_850)")
    parser.add_argument("--z_channel_idx", type=int, default=7,
                        help="Z channel index (default: 7 = z_500)")
    parser.add_argument("--resolution", type=float, default=0.25,
                        help="Grid resolution in degrees")
    parser.add_argument("--lead_times", type=int, nargs="+",
                        default=[24, 48, 72],
                        help="Lead times to evaluate (hours)")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device (cuda or cpu)")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"[Geostrophic] Loading predictions from: {args.pred_dir}")
    print(f"[Geostrophic] Method: {args.method}")

    results = run_geostrophic_eval_from_directory(
        pred_dir=args.pred_dir,
        method=args.method,
        u_channel_idx=args.u_channel_idx,
        v_channel_idx=args.v_channel_idx,
        z_channel_idx=args.z_channel_idx,
        resolution=args.resolution,
        lead_times=args.lead_times,
        device=args.device,
    )

    if not results:
        print(f"[ERROR] No valid prediction files found in {args.pred_dir}")
        return

    aggregated = aggregate_geostrophic_results(results, args.lead_times)

    output = {
        "method": args.method,
        "config": {
            "u_channel_idx": args.u_channel_idx,
            "v_channel_idx": args.v_channel_idx,
            "z_channel_idx": args.z_channel_idx,
            "resolution": args.resolution,
            "lead_times": args.lead_times,
        },
        "per_case_results": results,
        "aggregated": aggregated,
    }

    summary_path = os.path.join(args.output_dir, f"geostrophic_{args.method}_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"[Geostrophic] Results saved: {summary_path}")

    print("\n" + "=" * 80)
    print(f"Geostrophic Balance Results — {args.method.upper()}")
    print("=" * 80)
    print(f"{'Lead Time':>12} {'Imbalance':>12} {'Imb Std':>10} {'Div RMSE':>12} {'Improvement':>12}")
    print("-" * 62)
    for lead in args.lead_times:
        imb = aggregated.get(f"imbalance_{lead}h", {}).get("mean", np.nan)
        imb_std = aggregated.get(f"imbalance_{lead}h", {}).get("std", np.nan)
        div = aggregated.get(f"div_rmse_{lead}h", {}).get("mean", np.nan)
        imp = aggregated.get(f"improvement_{lead}h", {}).get("mean", np.nan)
        print(
            f"{lead:>11}h "
            f"{(imb if not np.isnan(imb) else 0):>12.4f} "
            f"{(imb_std if not np.isnan(imb_std) else 0):>10.4f} "
            f"{(div if not np.isnan(div) else 0):>12.4f} "
            f"{(imp if not np.isnan(imp) else 0):>12.4f}"
        )
    print("-" * 62)
    print("Imbalance units: m²/s² | Divergence units: s⁻¹")
    print(f"[Geostrophic] Processed {len(results)} cases. Done.")


if __name__ == "__main__":
    main()
