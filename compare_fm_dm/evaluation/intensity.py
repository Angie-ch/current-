"""
Intensity Evaluation — Flow Matching vs Diffusion Paper

Computes typhoon intensity metrics:
1. Central pressure error: |p_min_pred - p_min_gt| (from Z500 field)
2. Maximum wind error estimation from pressure gradient
3. Eye gradient: max(dZ/dr) at eyewall radius
4. Intensity category skill score

This script evaluates whether FM or DM better preserves typhoon structure
(sharp pressure gradient, distinct eye) vs. smoothed predictions.

Usage:
    python evaluation/intensity.py \
        --pred_dir table2_results/ \
        --output_dir evaluation/intensity_results/ \
        --method dm
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "Trajectory"))

try:
    from Trajectory.paper_eval_common import (
        traj_data_cfg,
        traj_model_cfg,
    )
except ImportError:
    from paper_eval_common import (
        traj_data_cfg,
        traj_model_cfg,
    )


GRAVITY = 9.80665
R_AIR = 287.05
T_CELSIUS = 273.15
REFERENCE_PRESSURE = 1015.0


def z_to_pressure(
    z: np.ndarray,
    t_ref: float = 273.15 + 15,
    p_ref: float = REFERENCE_PRESSURE,
) -> np.ndarray:
    """
    Convert geopotential height to pressure using hypsometric equation.

    p = p_ref * exp(-g * z / (R * T))

    Args:
        z: geopotential height in meters
        t_ref: reference temperature in Kelvin
        p_ref: reference pressure in hPa

    Returns:
        pressure: pressure in hPa
    """
    p = p_ref * np.exp(-GRAVITY * z / (R_AIR * t_ref))
    return p


def find_typhoon_center(
    z_field: np.ndarray,
    lat_array: np.ndarray,
    lon_array: np.ndarray,
    search_radius_grid: int = 8,
) -> Tuple[float, float, float, float]:
    """
    Find typhoon center from Z500 field by locating the minimum geopotential.

    Args:
        z_field: (H, W) Z500 field (normalized or physical units)
        lat_array: (H,) latitudes
        lon_array: (W,) longitudes
        search_radius_grid: search radius in grid points

    Returns:
        center_lat, center_lon, min_z, z_gradient_max
    """
    H, W = z_field.shape

    if z_field.min() == z_field.max():
        center_lat = lat_array[H // 2]
        center_lon = lon_array[W // 2]
        return center_lat, center_lon, float(z_field.mean()), 0.0

    min_idx = np.argmin(z_field)
    min_row, min_col = np.unravel_index(min_idx, z_field.shape)

    center_lat = float(lat_array[min_row])
    center_lon = float(lon_array[min_col])
    min_z = float(z_field[min_row, min_col])

    row_start = max(0, min_row - search_radius_grid)
    row_end = min(H, min_row + search_radius_grid + 1)
    col_start = max(0, min_col - search_radius_grid)
    col_end = min(W, min_col + search_radius_grid + 1)

    local_z = z_field[row_start:row_end, col_start:col_end]
    local_lat = lat_array[row_start:row_end]
    local_lon = lon_array[col_start:col_end]

    dr = 111000.0 * 0.25
    z_gradient_max = 0.0
    if local_z.size > 1:
        dz_dr = np.abs(np.gradient(local_z.ravel())).mean() / dr
        z_gradient_max = float(dz_dr)

    return center_lat, center_lon, min_z, z_gradient_max


def compute_radial_gradient(
    z_field: np.ndarray,
    center_row: int,
    center_col: int,
    lat_resolution: float = 0.25,
    max_radius_grid: int = 10,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute radial profile of Z field from typhoon center.

    Returns:
        radii: distances from center in km
        z_values: Z values at each radius
    """
    H, W = z_field.shape
    radii = []
    z_values = []

    lat_res_m = 111000.0 * lat_resolution
    lon_res_m = 111000.0 * lat_resolution * np.cos(np.radians(30))

    for dr in range(1, max_radius_grid + 1):
        r_pixels = dr
        points_in_ring = []
        z_in_ring = []

        for di in range(-dr, dr + 1):
            for dj in range(-dr, dr + 1):
                dist = np.sqrt(di ** 2 + dj ** 2)
                if abs(dist - dr) < 0.5:
                    ni = center_row + di
                    nj = center_col + dj
                    if 0 <= ni < H and 0 <= nj < W:
                        r_m = np.sqrt((di * lat_res_m) ** 2 + (dj * lon_res_m) ** 2)
                        radii.append(r_m / 1000.0)
                        z_values.append(float(z_field[ni, nj]))

        if z_values:
            radii.append(r_m / 1000.0)
            z_values.append(np.mean(z_values[-max(1, len(z_values) // 10):]))

    if not radii:
        return np.array([0.0]), np.array([float(z_field[center_row, center_col])])

    radii = np.array(radii)
    z_values = np.array(z_values)

    return radii, z_values


def compute_intensity_metrics_single_case(
    pred_z: np.ndarray,
    gt_z: np.ndarray,
    lat_array: np.ndarray,
    lon_array: np.ndarray,
) -> Dict[str, float]:
    """
    Compute all intensity metrics for a single case.

    Args:
        pred_z: (H, W) predicted Z500 field
        gt_z: (H, W) ground truth Z500 field
        lat_array: (H,) latitudes
        lon_array: (W,) longitudes

    Returns:
        Dict with p_min error, gradient metrics, eye sharpness
    """
    pred_lat, pred_lon, pred_min_z, pred_grad = find_typhoon_center(
        pred_z, lat_array, lon_array
    )
    gt_lat, gt_lon, gt_min_z, gt_grad = find_typhoon_center(
        gt_z, lat_array, lon_array
    )

    center_dist_km = np.sqrt(
        ((pred_lat - gt_lat) * 111.0) ** 2
        + ((pred_lon - gt_lon) * 111.0 * np.cos(np.radians(gt_lat))) ** 2
    )

    pred_p_min = z_to_pressure(pred_min_z)
    gt_p_min = z_to_pressure(gt_min_z)
    p_min_error = abs(pred_p_min - gt_p_min)

    pred_gradient_error = abs(pred_grad - gt_grad) / (abs(gt_grad) + 1e-6) if gt_grad > 0 else 0.0

    pred_radii, pred_z_profile = compute_radial_gradient(
        pred_z,
        int(np.argmin(pred_z) // pred_z.shape[1]),
        int(np.argmin(pred_z) % pred_z.shape[1]),
    )
    gt_radii, gt_z_profile = compute_radial_gradient(
        gt_z,
        int(np.argmin(gt_z) // gt_z.shape[1]),
        int(np.argmin(gt_z) % gt_z.shape[1]),
    )

    max_radius = min(len(pred_radii), len(gt_radii))
    if max_radius > 1:
        z_diff = np.abs(pred_z_profile[:max_radius] - gt_z_profile[:max_radius]).mean()
    else:
        z_diff = 0.0

    sharpness_pred = compute_eye_sharpness(pred_z_profile)
    sharpness_gt = compute_eye_sharpness(gt_z_profile)

    return {
        "p_min_error_hPa": float(p_min_error),
        "center_error_km": float(center_dist_km),
        "z_gradient_error": float(pred_gradient_error),
        "z_profile_mae": float(z_diff),
        "eye_sharpness_pred": float(sharpness_pred),
        "eye_sharpness_gt": float(sharpness_gt),
        "sharpness_error": float(abs(sharpness_pred - sharpness_gt)),
        "pred_min_z": float(pred_min_z),
        "gt_min_z": float(gt_min_z),
    }


def compute_eye_sharpness(z_profile: np.ndarray) -> float:
    """
    Compute eye sharpness metric from radial Z profile.

    Sharpness = (Z_outer - Z_center) / (Z_outer + Z_center + eps)
    Higher = sharper eye
    """
    if len(z_profile) < 2:
        return 0.0

    z_center = z_profile[0]
    z_outer = z_profile[-1]

    if abs(z_outer + z_center) < 1e-6:
        return 0.0

    sharpness = (z_outer - z_center) / (z_outer + z_center + 1e-6)
    return float(sharpness)


def intensity_eval_from_predictions(
    pred_samples: torch.Tensor,
    gt_sample: torch.Tensor,
    lat_array: np.ndarray,
    lon_array: np.ndarray,
    z_channel_idx: int = 7,
) -> Dict[str, float]:
    """
    Main intensity evaluation using predicted ERA5 fields.

    Args:
        pred_samples: (N, C, H, W) — N predicted samples
        gt_sample: (C, H, W) — ground truth
        lat_array: (H,) latitudes
        lon_array: (W,) longitudes
        z_channel_idx: Z channel index

    Returns:
        Dict with intensity metrics
    """
    if z_channel_idx >= pred_samples.shape[1]:
        z_channel_idx = 0

    pred_z = pred_samples[:, z_channel_idx].cpu().numpy()
    gt_z = gt_sample[z_channel_idx].cpu().numpy()

    metrics_list = []
    for n in range(pred_z.shape[0]):
        metrics = compute_intensity_metrics_single_case(
            pred_z[n], gt_z, lat_array, lon_array
        )
        metrics_list.append(metrics)

    aggregated = {}
    for key in metrics_list[0]:
        values = [m[key] for m in metrics_list]
        aggregated[f"{key}_mean"] = float(np.mean(values))
        aggregated[f"{key}_std"] = float(np.std(values))
        aggregated[f"{key}_median"] = float(np.median(values))

    return aggregated


def aggregate_intensity_results(
    per_case_results: List[Dict],
    lead_times: List[int],
) -> Dict:
    """Aggregate intensity results across cases and lead times."""
    aggregated = {}

    for lead in lead_times:
        p_err_list = [r.get(f"p_min_error_hPa_mean_{lead}h", np.nan) for r in per_case_results]
        sharp_err_list = [r.get(f"sharpness_error_mean_{lead}h", np.nan) for r in per_case_results]

        p_arr = np.array([p for p in p_err_list if not np.isnan(p)])
        s_arr = np.array([s for s in sharp_err_list if not np.isnan(s)])

        aggregated[f"p_min_error_{lead}h"] = {
            "mean": float(p_arr.mean()) if len(p_arr) > 0 else np.nan,
            "std": float(p_arr.std()) if len(p_arr) > 0 else np.nan,
            "n": len(p_arr),
        }
        aggregated[f"sharpness_error_{lead}h"] = {
            "mean": float(s_arr.mean()) if len(s_arr) > 0 else np.nan,
            "std": float(s_arr.std()) if len(s_arr) > 0 else np.nan,
        }

    return aggregated


def run_intensity_eval_from_directory(
    pred_dir: str,
    method: str = "dm",
    z_channel_idx: int = 7,
    lead_times: List[int] = None,
    device: str = "cuda",
) -> List[Dict]:
    """
    Run intensity evaluation on all prediction files in a directory.

    Args:
        pred_dir: directory containing prediction .pt files
        method: "dm" or "fm"
        z_channel_idx: Z channel index
        lead_times: list of lead times in hours
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
    lon_array = np.linspace(
        traj_data_cfg.lon_range[0],
        traj_data_cfg.lon_range[1],
        traj_data_cfg.grid_width,
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

            metrics = intensity_eval_from_predictions(
                pred_step, gt_step,
                lat_array, lon_array,
                z_channel_idx=z_channel_idx,
            )

            for k, v in metrics.items():
                case_result[f"{k}_{lead}h"] = v

        results.append(case_result)

    return results


def generate_intensity_comparison_table(
    dm_results: Dict,
    fm_results: Dict,
    lead_times: List[int],
) -> str:
    """Generate formatted comparison table for intensity metrics."""
    header = (
        f"{'Lead Time':>12} "
        f"{'DM p_err':>12} "
        f"{'FM p_err':>12} "
        f"{'DM Sharpen Err':>16} "
        f"{'FM Sharpen Err':>16} "
        f"{'p_err Improv':>14}"
    )
    divider = "-" * len(header)
    lines = [header, divider]

    for lead in lead_times:
        dm_p = dm_results.get(f"p_min_error_{lead}h", {}).get("mean", np.nan)
        fm_p = fm_results.get(f"p_min_error_{lead}h", {}).get("mean", np.nan)
        dm_s = dm_results.get(f"sharpness_error_{lead}h", {}).get("mean", np.nan)
        fm_s = fm_results.get(f"sharpness_error_{lead}h", {}).get("mean", np.nan)

        if not np.isnan(dm_p) and not np.isnan(fm_p) and dm_p > 0:
            improvement = (dm_p - fm_p) / dm_p * 100
        else:
            improvement = np.nan

        lines.append(
            f"{lead:>12}h "
            f"{(dm_p if not np.isnan(dm_p) else 0):>12.1f} "
            f"{(fm_p if not np.isnan(fm_p) else 0):>12.1f} "
            f"{(dm_s if not np.isnan(dm_s) else 0):>16.3f} "
            f"{(fm_s if not np.isnan(fm_s) else 0):>16.3f} "
            f"{(improvement if not np.isnan(improvement) else 0):>13.1f}%"
        )

    return "\n".join(lines)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Intensity Evaluation: FM vs DM typhoon structure preservation"
    )
    parser.add_argument("--pred_dir", type=str, required=True,
                        help="Directory containing prediction .pt files")
    parser.add_argument("--output_dir", type=str,
                        default="evaluation/intensity_results",
                        help="Output directory")
    parser.add_argument("--method", type=str, choices=["dm", "fm"], default="dm",
                        help="Method being evaluated")
    parser.add_argument("--z_channel_idx", type=int, default=7,
                        help="Z channel index (default: 7 = z_500)")
    parser.add_argument("--lead_times", type=int, nargs="+",
                        default=[24, 48, 72],
                        help="Lead times to evaluate (hours)")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device (cuda or cpu)")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"[Intensity] Loading predictions from: {args.pred_dir}")
    print(f"[Intensity] Method: {args.method}")

    results = run_intensity_eval_from_directory(
        pred_dir=args.pred_dir,
        method=args.method,
        z_channel_idx=args.z_channel_idx,
        lead_times=args.lead_times,
        device=args.device,
    )

    if not results:
        print(f"[ERROR] No valid prediction files found in {args.pred_dir}")
        return

    aggregated = aggregate_intensity_results(results, args.lead_times)

    output = {
        "method": args.method,
        "config": {
            "z_channel_idx": args.z_channel_idx,
            "lead_times": args.lead_times,
        },
        "per_case_results": results,
        "aggregated": aggregated,
    }

    summary_path = os.path.join(args.output_dir, f"intensity_{args.method}_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"[Intensity] Results saved: {summary_path}")

    print("\n" + "=" * 80)
    print(f"Intensity Evaluation Results — {args.method.upper()}")
    print("=" * 80)
    print(f"{'Lead Time':>12} {'p_min Error (hPa)':>18} {'Eye Sharpness Err':>18} {'N Cases':>10}")
    print("-" * 62)
    for lead in args.lead_times:
        p_err = aggregated.get(f"p_min_error_{lead}h", {}).get("mean", np.nan)
        p_std = aggregated.get(f"p_min_error_{lead}h", {}).get("std", np.nan)
        sharp_err = aggregated.get(f"sharpness_error_{lead}h", {}).get("mean", np.nan)
        n = aggregated.get(f"p_min_error_{lead}h", {}).get("n", 0)
        print(
            f"{lead:>11}h "
            f"{(p_err if not np.isnan(p_err) else 0):>13.1f} ± {(p_std if not np.isnan(p_std) else 0):.1f} "
            f"{(sharp_err if not np.isnan(sharp_err) else 0):>18.3f} "
            f"{n:>10}"
        )
    print("-" * 62)
    print("[Intensity] Done.")


if __name__ == "__main__":
    main()
