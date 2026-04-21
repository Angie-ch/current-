"""
Spatial Structure Metrics — SEDI, FSS, Pattern Correlation

Publication-grade metrics for assessing spatial forecast quality beyond point-wise RMSE.
These metrics measure whether the model captures spatial patterns, structures, and extremes.
"""
import numpy as np
import torch
from typing import Dict, List, Optional, Tuple


def compute_pattern_correlation(
    pred: np.ndarray,
    target: np.ndarray,
    threshold: Optional[float] = None,
) -> float:
    """
    Pattern correlation (Pearson correlation of spatial fields).

    Measures how well the spatial patterns correlate between prediction and truth.

    Args:
        pred: (H, W) or (C, H, W) predicted field
        target: (H, W) or (C, H, W) ground truth field
        threshold: Optional threshold for binary pattern matching

    Returns:
        Pattern correlation coefficient (1D scalar)
    """
    if pred.ndim == 3:
        pred = pred.mean(axis=0)
    if target.ndim == 3:
        target = target.mean(axis=0)

    if threshold is not None:
        pred = (pred >= threshold).astype(float)
        target = (target >= threshold).astype(float)

    pred_flat = pred.ravel()
    target_flat = target.ravel()

    pred_centered = pred_flat - pred_flat.mean()
    target_centered = target_flat - target_flat.mean()

    numerator = np.sum(pred_centered * target_centered)
    denominator = np.sqrt(np.sum(pred_centered ** 2) * np.sum(target_centered ** 2))

    if denominator < 1e-10:
        return 0.0

    return float(numerator / denominator)


def compute_sedi(
    pred: np.ndarray,
    target: np.ndarray,
    threshold: Optional[float] = None,
) -> float:
    """
    SEDI (Spatial Efficiency Index) — alternative to FSS for binary events.

    SEDI = (FSS - FSS_random) / (1 - FSS_random)

    where FSS_random = 2 * p * (1 - p) and p is the base rate.

    SEDI ranges from -1 to 1:
        SEDI > 0: Better than random
        SEDI = 1: Perfect forecast
        SEDI = 0: No better than climatology
        SEDI < 0: Worse than random

    Args:
        pred: (H, W) predicted field
        target: (H, W) ground truth field
        threshold: threshold for binary event definition

    Returns:
        SEDI score
    """
    if threshold is not None:
        pred_binary = (pred >= threshold).astype(float)
        target_binary = (target >= threshold).astype(float)
    else:
        pred_binary = pred
        target_binary = target

    p = target_binary.mean()
    q = pred_binary.mean()

    # FSS (Fractions Skill Score)
    mse_fcst = np.mean((pred_binary - target_binary) ** 2)
    mse_base = 2 * p * (1 - p)  # Reference MSE (climatology)

    if mse_base < 1e-10:
        return 0.0

    fss = 1.0 - mse_fcst / mse_base
    fss = np.clip(fss, -1.0, 1.0)

    # SEDI transformation
    fss_random = 2 * p * (1 - p)
    if abs(1 - fss_random) < 1e-10:
        return 0.0

    sedi = (fss - fss_random) / (1 - fss_random)
    return float(np.clip(sedi, -1.0, 1.0))


def compute_fss(
    pred: np.ndarray,
    target: np.ndarray,
    threshold: float,
    scales: List[int] = None,
) -> Dict[str, float]:
    """
    Fractions Skill Score (FSS) for spatial forecast verification.

    FSS measures skill at different spatial scales. It compares the fractions
    of area exceeding a threshold within a sliding window.

    Reference: Roberts & Lean (2008)

    Args:
        pred: (H, W) predicted field
        target: (H, W) ground truth field
        threshold: threshold for binary event
        scales: list of window sizes (in grid points), e.g., [1, 3, 5, 7]

    Returns:
        Dict mapping scale -> FSS value
    """
    if scales is None:
        scales = [1, 3, 5, 7, 9]

    pred_binary = (pred >= threshold).astype(float)
    target_binary = (target >= threshold).astype(float)

    results = {}
    for scale in scales:
        if scale == 1:
            # No spatial averaging needed
            mse_fcst = np.mean((pred_binary - target_binary) ** 2)
        else:
            # Boxcar average
            pred_frac = spatial_average(pred_binary, scale)
            target_frac = spatial_average(target_binary, scale)
            mse_fcst = np.mean((pred_frac - target_frac) ** 2)

        # Reference MSE (random forecast)
        p = target_binary.mean()
        mse_ref = 2 * p * (1 - p)

        if mse_ref < 1e-10:
            fss = 1.0
        else:
            fss = 1.0 - mse_fcst / mse_ref

        results[f"scale_{scale}"] = float(np.clip(fss, 0.0, 1.0))

    return results


def spatial_average(field: np.ndarray, window_size: int) -> np.ndarray:
    """Simple spatial averaging with a boxcar filter."""
    from scipy.ndimage import uniform_filter
    return uniform_filter(field, size=window_size, mode='constant')


def compute_mae_spatial_gradient(
    pred: np.ndarray,
    target: np.ndarray,
) -> float:
    """
    MAE of spatial gradients — measures whether the model captures
    sharp gradients and fronts.

    Large gradients (fronts, sharp features) are critical for weather forecasting.
    A model that smooths gradients will score poorly on this metric.

    Args:
        pred: (H, W) predicted field
        target: (H, W) ground truth field

    Returns:
        MAE of gradient magnitude
    """
    # Compute gradients using finite differences
    pred_dx = np.gradient(pred, axis=1)
    pred_dy = np.gradient(pred, axis=0)
    target_dx = np.gradient(target, axis=1)
    target_dy = np.gradient(target, axis=0)

    pred_grad_mag = np.sqrt(pred_dx**2 + pred_dy**2)
    target_grad_mag = np.sqrt(target_dx**2 + target_dy**2)

    return float(np.mean(np.abs(pred_grad_mag - target_grad_mag)))


def compute_mae_laplacian(
    pred: np.ndarray,
    target: np.ndarray,
) -> float:
    """
    MAE of the Laplacian — measures second-order smoothness.

    The Laplacian ∇²f = ∂²f/∂x² + ∂²f/∂y² measures curvature.
    Models that over-smooth produce smaller Laplacians.
    Models that under-smooth produce erratic Laplacians.

    Args:
        pred: (H, W) predicted field
        target: (H, W) ground truth field

    Returns:
        MAE of Laplacian
    """
    from scipy.ndimage import laplace

    pred_lap = laplace(pred)
    target_lap = laplace(target)

    return float(np.mean(np.abs(pred_lap - target_lap)))


def compute_极端事件_metrics(
    pred: np.ndarray,
    target: np.ndarray,
    percentiles: List[float] = None,
) -> Dict[str, Dict[str, float]]:
    """
    Extreme event metrics — assess forecast of rare/strong events.

    These are critical for applications like typhoon forecasting where
    extreme events (high wind, low pressure) matter most.

    Args:
        pred: (H, W) predicted field
        target: (H, W) ground truth field
        percentiles: list of percentiles to evaluate (e.g., [90, 95, 99])

    Returns:
        Dict with metrics for each percentile threshold
    """
    if percentiles is None:
        percentiles = [90, 95, 99]

    results = {}
    pred_flat = pred.ravel()
    target_flat = target.ravel()

    for pct in percentiles:
        threshold_target = np.percentile(target_flat, pct)
        threshold_pred = np.percentile(pred_flat, pct)

        # POD (Probability of Detection): did we predict the extreme when it occurred?
        hit = np.sum((pred >= threshold_target) & (target >= threshold_target))
        total_observed = np.sum(target >= threshold_target)
        pod = hit / total_observed if total_observed > 0 else 0.0

        # FAR (False Alarm Ratio): of predicted extremes, how many were false?
        total_predicted = np.sum(pred >= threshold_target)
        false_alarms = total_predicted - hit
        far = false_alarms / total_predicted if total_predicted > 0 else 0.0

        # Bias
        bias = total_predicted / total_observed if total_observed > 0 else 0.0

        # RMSE in extreme region
        extreme_mask = target >= threshold_target
        if extreme_mask.sum() > 0:
            extreme_rmse = np.sqrt(np.mean((pred[extreme_mask] - target[extreme_mask]) ** 2))
        else:
            extreme_rmse = 0.0

        results[f"pct_{pct}"] = {
            "threshold": float(threshold_target),
            "pod": float(pod),
            "far": float(far),
            "bias": float(bias),
            "extreme_rmse": float(extreme_rmse),
            "n_extreme_obs": int(total_observed),
            "n_extreme_pred": int(total_predicted),
        }

    return results


def compute_all_spatial_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    channel_names: List[str] = None,
    device: str = "cpu",
) -> Dict:
    """
    Compute all spatial structure metrics for a batch.

    Args:
        pred: (B, C, H, W) predictions
        target: (B, C, H, W) ground truth
        channel_names: list of C channel names
        device: torch device

    Returns:
        Dict with all spatial metrics
    """
    pred = pred.to(device)
    target = target.to(device)
    B, C, H, W = pred.shape

    if channel_names is None:
        channel_names = [f"ch_{i}" for i in range(C)]

    results = {
        "pattern_correlation_per_channel": {},
        "sedi_per_channel": {},
        "spatial_gradient_mae_per_channel": {},
        "laplacian_mae_per_channel": {},
        "extreme_events": {},
    }

    for c, ch_name in enumerate(channel_names):
        pred_ch = pred[:, c].cpu().numpy()  # (B, H, W)
        target_ch = target[:, c].cpu().numpy()

        # Average over batch for spatial metrics
        pred_mean = pred_ch.mean(axis=0)
        target_mean = target_ch.mean(axis=0)

        # Pattern correlation
        results["pattern_correlation_per_channel"][ch_name] = float(
            compute_pattern_correlation(pred_mean, target_mean)
        )

        # SEDI (using median as threshold)
        threshold = float(target_mean.median())
        results["sedi_per_channel"][ch_name] = float(
            compute_sedi(pred_mean, target_mean, threshold=threshold)
        )

        # Spatial gradient MAE
        results["spatial_gradient_mae_per_channel"][ch_name] = float(
            compute_mae_spatial_gradient(pred_mean, target_mean)
        )

        # Laplacian MAE
        results["laplacian_mae_per_channel"][ch_name] = float(
            compute_mae_laplacian(pred_mean, target_mean)
        )

        # Extreme events
        results["extreme_events"][ch_name] = compute_极端事件_metrics(
            pred_mean, target_mean, percentiles=[90, 95, 99]
        )

    # Aggregate
    pc_vals = list(results["pattern_correlation_per_channel"].values())
    sedi_vals = list(results["sedi_per_channel"].values())

    results["pattern_correlation_mean"] = float(np.mean(pc_vals))
    results["sedi_mean"] = float(np.mean(sedi_vals))

    return results
