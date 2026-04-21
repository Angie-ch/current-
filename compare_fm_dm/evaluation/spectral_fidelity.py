"""
Spectral Fidelity Evaluation — Flow Matching vs Diffusion Paper

Computes:
1. Radial Power Spectral Density (PSD) from Z500 fields
2. Spectral slope beta fitting (log-log linear regression)
3. High-frequency energy ratio E(k>15) / E(total)
4. Per-lead-time spectral analysis (24h, 48h, 72h)

This script evaluates whether FM or DM better preserves atmospheric energy cascades.
Expected: FM's beta ≈ -5/3 (theoretical); DM's beta closer to 0 (oversmoothed).

Usage:
    python evaluation/spectral_fidelity.py \
        --pred_dir table2_results/ \
        --output_dir evaluation/spectral_results/ \
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

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "Trajectory"))

try:
    from compare_fm_dm.evaluation.metrics import compute_2d_psd, compute_spectral_slope
    from Trajectory.paper_eval_common import (
        traj_data_cfg,
        traj_model_cfg,
    )
except ImportError:
    from paper_eval_common import (
        traj_data_cfg,
        traj_model_cfg,
    )


def compute_radial_psd_torch(
    field: torch.Tensor,
    resolution: float = 0.25,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute radial-averaged Power Spectral Density using torch FFT.

    Args:
        field: (B, C, H, W) or (H, W) — meteorological field
        resolution: grid resolution in degrees

    Returns:
        k: wavenumbers (cycles/grid)
        E: radially-averaged power spectrum
    """
    if field.ndim == 4:
        field = field.mean(dim=(0, 1))  # (H, W)
    elif field.ndim == 2:
        field = field.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
        field = field.mean(dim=(0, 1))

    if isinstance(field, torch.Tensor):
        field = field.cpu().numpy()
    else:
        field = np.array(field)

    if field.ndim == 2:
        field = field[np.newaxis, np.newaxis, :, :]

    H, W = field.shape[-2:]
    device = "cpu"

    field_np = field.mean(axis=(0, 1))  # average over batch and channels
    field_np = field_np - field_np.mean()

    fft = np.fft.fft2(field_np)
    fft_shifted = np.fft.fftshift(fft)
    power = np.abs(fft_shifted) ** 2

    y, x = np.ogrid[:H, :W]
    center_y, center_x = H // 2, W // 2
    r = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
    r = r.astype(int)

    max_r = min(center_y, center_x)
    E = np.zeros(max_r + 1)
    counts = np.zeros(max_r + 1)

    for k_val in range(max_r + 1):
        mask = r == k_val
        E[k_val] = power[mask].sum()
        counts[k_val] = mask.sum()

    counts = np.maximum(counts, 1)
    E = E / counts

    k = np.arange(max_r + 1, dtype=np.float64)
    return k, E


def fit_power_law(
    k: np.ndarray,
    E: np.ndarray,
    k_min: float = 5.0,
    k_max: float = 20.0,
) -> Tuple[float, float, float]:
    """
    Fit E(k) = alpha * k^beta in log-log space using linear regression.

    Args:
        k: wavenumbers
        E: power spectrum
        k_min, k_max: fitting range in wavenumber space

    Returns:
        beta: spectral slope
        alpha: prefactor
        r_squared: goodness of fit
    """
    mask = (k >= k_min) & (k <= k_max) & (E > 0)
    k_fit = k[mask]
    E_fit = E[mask]

    if len(k_fit) < 3:
        return np.nan, np.nan, 0.0

    log_k = np.log(k_fit + 1e-10)
    log_E = np.log(E_fit + 1e-10)

    n = len(log_k)
    sum_x = log_k.sum()
    sum_y = log_E.sum()
    sum_xy = (log_k * log_E).sum()
    sum_x2 = (log_k * log_k).sum()
    sum_y2 = (log_E * log_E).sum()

    denom = n * sum_x2 - sum_x ** 2
    if abs(denom) < 1e-10:
        return np.nan, np.nan, 0.0

    beta = (n * sum_xy - sum_x * sum_y) / denom
    alpha = np.exp((sum_y - beta * sum_x) / n)

    ss_res = ((log_E - (beta * log_k + np.log(alpha))) ** 2).sum()
    ss_tot = ((log_E - log_E.mean()) ** 2).sum()
    r_squared = 1 - ss_res / (ss_tot + 1e-10)

    return float(beta), float(alpha), float(r_squared)


def compute_high_freq_ratio(
    k: np.ndarray,
    E: np.ndarray,
    k_threshold: float = 15.0,
) -> float:
    """
    Compute ratio of high-wavenumber energy to total energy.

    E_high = sum(E[k > k_threshold])
    E_total = sum(E[k >= 1])
    ratio = E_high / E_total
    """
    mask = k >= k_threshold
    if not mask.any():
        return 0.0
    E_high = E[mask].sum()
    E_total = E[k >= 1].sum()
    if E_total < 1e-10:
        return 0.0
    return float(E_high / E_total)


def spectral_analysis_single_field(
    field: np.ndarray,
    resolution: float = 0.25,
    k_range: Tuple[float, float] = (5.0, 15.0),
    k_threshold: float = 15.0,
) -> Dict[str, float]:
    """
    Run full spectral analysis on a single meteorological field.

    Args:
        field: (H, W) or (C, H, W)
        resolution: grid resolution in degrees
        k_range: wavenumber range for slope fitting
        k_threshold: threshold for high-frequency ratio

    Returns:
        Dict with beta, alpha, r_squared, high_freq_ratio, total_power
    """
    if field.ndim == 3:
        field = field.mean(axis=0)

    k, E = compute_radial_psd_torch(
        torch.from_numpy(field).unsqueeze(0).unsqueeze(0),
        resolution=resolution,
    )

    beta, alpha, r2 = fit_power_law(k, E, k_min=k_range[0], k_max=k_range[1])
    hf_ratio = compute_high_freq_ratio(k, E, k_threshold=k_threshold)
    total_power = float(E.sum())

    return {
        "spectral_slope_beta": beta,
        "prefactor_alpha": alpha,
        "r_squared": r2,
        "high_freq_ratio": hf_ratio,
        "total_power": total_power,
        "k_max_valid": float(k[E > 0].max()) if (E > 0).any() else 0.0,
    }


def aggregate_spectral_results(
    per_case_results: List[Dict],
    lead_times: List[int],
) -> Dict:
    """
    Aggregate spectral results across cases and lead times.

    Args:
        per_case_results: list of per-case spectral results
        lead_times: list of lead times in hours

    Returns:
        Aggregated statistics
    """
    aggregated = {}

    for lead in lead_times:
        beta_list = [r.get(f"beta_{lead}h", np.nan) for r in per_case_results]
        hf_list = [r.get(f"hf_ratio_{lead}h", np.nan) for r in per_case_results]

        beta_arr = np.array([b for b in beta_list if not np.isnan(b)])
        hf_arr = np.array([h for h in hf_list if not np.isnan(h)])

        aggregated[f"beta_{lead}h"] = {
            "mean": float(beta_arr.mean()) if len(beta_arr) > 0 else np.nan,
            "std": float(beta_arr.std()) if len(beta_arr) > 0 else np.nan,
            "n": len(beta_arr),
        }
        aggregated[f"hf_ratio_{lead}h"] = {
            "mean": float(hf_arr.mean()) if len(hf_arr) > 0 else np.nan,
            "std": float(hf_arr.std()) if len(hf_arr) > 0 else np.nan,
        }

    return aggregated


def generate_spectral_comparison_table(
    dm_results: Dict,
    fm_results: Dict,
    gt_results: Dict,
    lead_times: List[int],
) -> str:
    """
    Generate a formatted comparison table for spectral metrics.

    Returns:
        Formatted table string
    """
    header = (
        f"{'Lead Time':>12} {'GT beta':>12} {'DM beta':>14} {'FM beta':>14} "
        f"{'DM beta_err':>14} {'FM beta_err':>14} {'GT HF%':>10} "
        f"{'DM HF%':>10} {'FM HF%':>10}"
    )
    divider = "-" * len(header)
    lines = [header, divider]

    for lead in lead_times:
        gt_beta = gt_results.get(f"beta_{lead}h", {}).get("mean", np.nan)
        dm_beta = dm_results.get(f"beta_{lead}h", {}).get("mean", np.nan)
        fm_beta = fm_results.get(f"beta_{lead}h", {}).get("mean", np.nan)

        dm_err = abs(dm_beta - gt_beta) if not np.isnan(dm_beta) and not np.isnan(gt_beta) else np.nan
        fm_err = abs(fm_beta - gt_beta) if not np.isnan(fm_beta) and not np.isnan(gt_beta) else np.nan

        gt_hf = gt_results.get(f"hf_ratio_{lead}h", {}).get("mean", np.nan)
        dm_hf = dm_results.get(f"hf_ratio_{lead}h", {}).get("mean", np.nan)
        fm_hf = fm_results.get(f"hf_ratio_{lead}h", {}).get("mean", np.nan)

        line = (
            f"{lead:>12}h "
            f"{gt_beta:>12.2f} "
            f"{dm_beta:>14.2f} "
            f"{fm_beta:>14.2f} "
            f"{dm_err:>14.2f} "
            f"{fm_err:>14.2f} "
            f"{gt_hf*100 if not np.isnan(gt_hf) else 0:>10.1f} "
            f"{dm_hf*100 if not np.isnan(dm_hf) else 0:>10.1f} "
            f"{fm_hf*100 if not np.isnan(fm_hf) else 0:>10.1f}"
        )
        lines.append(line)

    return "\n".join(lines)


def run_spectral_eval_from_predictions(
    prediction_samples: torch.Tensor,
    ground_truth_samples: torch.Tensor,
    lead_time_indices: List[int],
    z_channel_idx: int = 7,
    resolution: float = 0.25,
    k_range: Tuple[float, float] = (5.0, 15.0),
    k_threshold: float = 15.0,
) -> Dict[str, Dict[str, float]]:
    """
    Main spectral evaluation function for predicted ERA5 fields.

    Args:
        prediction_samples: (N, T, C, H, W) — N samples, T time steps
        ground_truth_samples: (T, C, H, W) — single GT per time step
        lead_time_indices: which time step indices to evaluate
        z_channel_idx: which channel is Z500 (0-indexed within the T*C flatten)
        resolution: grid resolution in degrees
        k_range: wavenumber range for slope fitting
        k_threshold: threshold for high-frequency ratio

    Returns:
        {f"{lead_time}h": {beta, alpha, r_squared, high_freq_ratio, ...}, ...}
    """
    results = {}

    T = prediction_samples.shape[1]
    C = prediction_samples.shape[2]
    H = prediction_samples.shape[3]
    W = prediction_samples.shape[4]

    for step_idx in lead_time_indices:
        if step_idx >= T:
            continue

        lead_time_h = (step_idx + 1) * traj_data_cfg.time_resolution_hours

        pred_step = prediction_samples[:, step_idx]  # (N, C, H, W)
        gt_step = ground_truth_samples[step_idx]     # (C, H, W)

        if z_channel_idx >= C:
            z_channel_idx = 0

        pred_z = pred_step[:, z_channel_idx].cpu().numpy()  # (N, H, W)
        gt_z = gt_step[z_channel_idx].cpu().numpy()           # (H, W)

        pred_betas = []
        pred_hf_ratios = []
        for n in range(pred_z.shape[0]):
            res = spectral_analysis_single_field(
                pred_z[n],
                resolution=resolution,
                k_range=k_range,
                k_threshold=k_threshold,
            )
            pred_betas.append(res["spectral_slope_beta"])
            pred_hf_ratios.append(res["high_freq_ratio"])

        gt_res = spectral_analysis_single_field(
            gt_z,
            resolution=resolution,
            k_range=k_range,
            k_threshold=k_threshold,
        )

        results[f"{lead_time_h}h"] = {
            f"beta_{lead_time_h}h": np.nanmean(pred_betas),
            f"beta_{lead_time_h}h_std": np.nanstd(pred_betas),
            f"gt_beta_{lead_time_h}h": gt_res["spectral_slope_beta"],
            f"beta_err_{lead_time_h}h": abs(np.nanmean(pred_betas) - gt_res["spectral_slope_beta"]),
            f"hf_ratio_{lead_time_h}h": np.nanmean(pred_hf_ratios),
            f"gt_hf_ratio_{lead_time_h}h": gt_res["high_freq_ratio"],
            f"gt_beta_err_{lead_time_h}h": abs(gt_res["spectral_slope_beta"] - (-1.67)),
            "n_samples": len(pred_betas),
        }

    return results


def load_era5_predictions_from_directory(
    pred_dir: str,
    method: str = "dm",
) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[str]]:
    """
    Load ERA5 predictions and ground truth from saved prediction files.

    Args:
        pred_dir: directory containing ar_pred_*.pt files
        method: "dm" or "fm"

    Returns:
        predictions: list of (N_samples, T, C, H, W) tensors
        ground_truths: list of (T, C, H, W) tensors
        case_keys: list of case identifiers
    """
    pred_path = Path(pred_dir)
    pt_files = sorted(pred_path.glob("ar_pred_*.pt"))

    if not pt_files:
        pt_files = sorted(pred_path.glob("*.pt"))

    predictions = []
    ground_truths = []
    case_keys = []

    for pt_file in pt_files:
        data = torch.load(pt_file, map_location="cpu", weights_only=False)

        for key in ("predictions", "prediction", "prediction_samples"):
            if key in data:
                pred = data[key]
                break
        else:
            continue

        for key in ("ground_truth", "gt", "target"):
            if key in data:
                gt = data[key]
                break
        else:
            gt = None

        case_keys.append(pt_file.stem)
        predictions.append(pred)
        if gt is not None:
            ground_truths.append(gt)

    return predictions, ground_truths, case_keys


def parse_args():
    parser = argparse.ArgumentParser(
        description="Spectral Fidelity Evaluation: FM vs DM power spectrum analysis"
    )
    parser.add_argument(
        "--pred_dir",
        type=str,
        required=True,
        help="Directory containing prediction .pt files",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="evaluation/spectral_results",
        help="Output directory for results",
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=["dm", "fm"],
        default="dm",
        help="Method being evaluated",
    )
    parser.add_argument(
        "--z_channel_idx",
        type=int,
        default=7,
        help="Z channel index in flattened (T*C) format (default: 7 = z_500)",
    )
    parser.add_argument(
        "--resolution",
        type=float,
        default=0.25,
        help="Grid resolution in degrees",
    )
    parser.add_argument(
        "--k_min",
        type=float,
        default=5.0,
        help="Minimum wavenumber for slope fitting",
    )
    parser.add_argument(
        "--k_max",
        type=float,
        default=15.0,
        help="Maximum wavenumber for slope fitting",
    )
    parser.add_argument(
        "--k_threshold",
        type=float,
        default=15.0,
        help="Wavenumber threshold for high-frequency ratio",
    )
    parser.add_argument(
        "--lead_times",
        type=int,
        nargs="+",
        default=[24, 48, 72],
        help="Lead times to evaluate (in hours)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"[Spectral Fidelity] Loading predictions from: {args.pred_dir}")
    print(f"[Spectral Fidelity] Method: {args.method}")
    print(f"[Spectral Fidelity] K range: [{args.k_min}, {args.k_max}]")

    lead_time_indices = [
        lt // traj_data_cfg.time_resolution_hours - 1
        for lt in args.lead_times
    ]

    predictions, ground_truths, case_keys = load_era5_predictions_from_directory(
        args.pred_dir, method=args.method
    )

    if not predictions:
        print(f"[ERROR] No prediction files found in {args.pred_dir}")
        print("Expected files named: ar_pred_*.pt")
        return

    print(f"[Spectral Fidelity] Loaded {len(predictions)} cases")

    all_results = []
    for i, (pred, gt, case_key) in enumerate(zip(predictions, ground_truths, case_keys)):
        results = run_spectral_eval_from_predictions(
            pred,
            gt,
            lead_time_indices=lead_time_indices,
            z_channel_idx=args.z_channel_idx,
            resolution=args.resolution,
            k_range=(args.k_min, args.k_max),
            k_threshold=args.k_threshold,
        )
        results["case_key"] = case_key
        all_results.append(results)

    aggregated = aggregate_spectral_results(all_results, args.lead_times)

    output = {
        "method": args.method,
        "config": {
            "z_channel_idx": args.z_channel_idx,
            "resolution": args.resolution,
            "k_range": [args.k_min, args.k_max],
            "k_threshold": args.k_threshold,
            "lead_times": args.lead_times,
        },
        "per_case_results": [
            {k: v for k, v in r.items() if k != "case_key"}
            for r in all_results
        ],
        "aggregated": aggregated,
    }

    summary_path = os.path.join(args.output_dir, f"spectral_{args.method}_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"[Spectral Fidelity] Results saved: {summary_path}")

    print("\n" + "=" * 80)
    print(f"Spectral Fidelity Results — {args.method.upper()}")
    print("=" * 80)
    print(f"{'Lead Time':>12} {'GT beta':>10} {'Method beta':>12} {'beta_err':>10} {'HF Ratio':>10}")
    print("-" * 60)
    for lead in args.lead_times:
        gt_beta = aggregated.get(f"beta_{lead}h", {}).get("gt_beta", {}).get(f"gt_beta_{lead}h", np.nan)
        if isinstance(gt_beta, dict):
            gt_beta = gt_beta.get(f"gt_beta_{lead}h", np.nan)
        method_beta = aggregated.get(f"beta_{lead}h", {}).get("mean", np.nan)
        if isinstance(method_beta, dict):
            method_beta = method_beta.get(f"beta_{lead}h", aggregated[f"beta_{lead}h"].get("mean", np.nan))
        beta_err = aggregated.get(f"beta_{lead}h", {}).get(f"beta_err_{lead}h", np.nan)
        if isinstance(beta_err, dict):
            beta_err = beta_err.get(f"beta_err_{lead}h", np.nan)
        hf_ratio = aggregated.get(f"hf_ratio_{lead}h", {}).get("mean", np.nan)
        if isinstance(hf_ratio, dict):
            hf_ratio = hf_ratio.get(f"hf_ratio_{lead}h", np.nan)

        print(
            f"{lead:>11}h "
            f"{gt_beta if not np.isnan(gt_beta) else -1.67:>10.2f} "
            f"{method_beta if not np.isnan(method_beta) else 0:>12.2f} "
            f"{beta_err if not np.isnan(beta_err) else 0:>10.2f} "
            f"{(hf_ratio*100) if not np.isnan(hf_ratio) else 0:>9.1f}%"
        )
    print("-" * 60)
    print("Reference: GT spectral slope ≈ -5/3 ≈ -1.67 (2D enstrophy cascade)")
    print("          DM tends toward 0 (oversmoothed); FM closer to -1.67")

    print(f"\n[Spectral Fidelity] Per-case results: {len(all_results)} cases")
    print(f"[Spectral Fidelity] Done.")


if __name__ == "__main__":
    main()
