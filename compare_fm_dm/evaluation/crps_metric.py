"""
CRPS (Continuous Ranked Probability Score) Implementation

CRPS is the gold standard for probabilistic forecast evaluation.
It generalizes the Brier score to continuous variables and measures
both accuracy and calibration of ensemble forecasts.

Formula:
    CRPS(F, y) = ∫_{-∞}^{+∞} (F(z) - H(z - y))² dz

where F is the CDF of the ensemble and H is the Heaviside step function.
"""
import numpy as np
import torch
from typing import List, Optional


def compute_crps_ensemble(
    ensemble_members: np.ndarray,
    observations: np.ndarray,
) -> float:
    """
    Compute CRPS for a single grid point over one ensemble.

    ensemble_members: (K,) — K ensemble member values at one grid point
    observations: scalar or (1,) — observed value at that grid point

    Uses the analytical formula for CRPS:
    CRPS = (1/K²) Σ_{i=1}^{K} Σ_{j=1}^{K} |x_i - x_j| / 2 - ∫ |z - y| dF(z)
         ≈ (1/K²) Σ_{i} Σ_{j} |x_i - x_j| / 2 - Σ_{i} |x_i - y| / K

    For computational efficiency we use the sorted-pairwise difference approach.
    """
    K = ensemble_members.shape[0]
    if K < 2:
        return 0.0

    # Sort ensemble members
    sorted_ens = np.sort(ensemble_members)
    y = float(observations)

    # Analytical CRPS using the sorted ensemble
    crps = 0.0

    # Part 1: Expected absolute difference between two random ensemble members
    # E|x_i - x_j| for i ≠ j drawn uniformly
    for i in range(K):
        for j in range(i + 1, K):
            crps += np.abs(sorted_ens[i] - sorted_ens[j])

    crps = crps / (K * K)

    # Part 2: Minus the expected absolute difference from ensemble to observation
    for i in range(K):
        crps -= np.abs(sorted_ens[i] - y) / K

    # Add tail correction
    crps += np.abs(sorted_ens[0] - y) / (2 * K)
    crps += np.abs(sorted_ens[-1] - y) / (2 * K)

    return max(0.0, crps)


def compute_crps_vectorized(
    ensemble_members: np.ndarray,
    observations: np.ndarray,
) -> np.ndarray:
    """
    Vectorized CRPS computation over all grid points simultaneously.

    ensemble_members: (K, ...) or (K, H, W) — K ensemble members at each grid point
    observations: (...) or (H, W) — observed values

    K is the ensemble size, typically K=5 to K=32.
    """
    K = ensemble_members.shape[0]

    if K < 2:
        return np.zeros_like(observations)

    # Sort along ensemble dimension
    sorted_ens = np.sort(ensemble_members, axis=0)  # (K, ...)

    # Part 1: Mean pairwise absolute difference
    # E|x_i - x_j| = (1/K²) * Σ_i Σ_j |x_i - x_j|
    # This equals (1/K²) * 2 * Σ_{i} (2i - K + 1) * x[i] for sorted x
    weights = (2 * np.arange(1, K + 1) - K - 1).reshape(-1, *([1] * (sorted_ens.ndim - 1)))
    part1 = 2.0 * np.sum(weights * sorted_ens, axis=0) / (K * K)

    # Part 2: Expected absolute difference from ensemble to observation
    part2 = np.mean(np.abs(sorted_ens - observations), axis=0)

    # CRPS
    crps = part1 - part2

    # Tail corrections
    crps += np.abs(sorted_ens[0] - observations) / (2 * K)
    crps += np.abs(sorted_ens[-1] - observations) / (2 * K)

    return np.maximum(0.0, crps)


def compute_crps_spatial(
    ensemble_members: np.ndarray,
    observations: np.ndarray,
) -> float:
    """
    Compute spatially-averaged CRPS over the entire domain.

    ensemble_members: (K, C, H, W) — K ensemble members, C channels
    observations: (C, H, W) — observed field

    Returns the mean CRPS across all channels and grid points.
    """
    crps_map = compute_crps_vectorized(ensemble_members, observations)
    return float(crps_map.mean())


def compute_crps_per_channel(
    ensemble_members: np.ndarray,
    observations: np.ndarray,
    channel_names: List[str],
) -> dict:
    """
    Compute CRPS per channel.

    ensemble_members: (K, C, H, W)
    observations: (C, H, W)
    channel_names: list of C channel names

    Returns dict mapping channel_name -> crps_value
    """
    crps_map = compute_crps_vectorized(ensemble_members, observations)
    result = {}
    for i, ch_name in enumerate(channel_names):
        result[ch_name] = float(crps_map[i].mean())
    return result


def compute_spread_skill_ratio(
    ensemble_members: np.ndarray,
    observations: np.ndarray,
) -> float:
    """
    Compute the spread-skill ratio.

    Spread = mean of pairwise ensemble standard deviation
    Skill = RMSE of ensemble mean vs observation

    Ideal ratio = 1.0 (perfectly calibrated ensemble)
    Ratio < 1.0: ensemble is underdispersive (overconfident)
    Ratio > 1.0: ensemble is overdispersive (underconfident)

    ensemble_members: (K, C, H, W)
    observations: (C, H, W)
    """
    K = ensemble_members.shape[0]

    # Spread: mean pairwise standard deviation
    ens_mean = np.mean(ensemble_members, axis=0)
    spread = np.sqrt(np.mean((ensemble_members - ens_mean) ** 2, axis=0))

    # Skill: RMSE of ensemble mean
    skill = np.sqrt(np.mean((ens_mean - observations) ** 2))

    # Ratio
    spread_spatial = np.mean(spread)
    ratio = spread_spatial / (skill + 1e-8)

    return float(ratio)


def compute_reliability_diagram(
    ensemble_members: np.ndarray,
    observations: np.ndarray,
    n_bins: int = 10,
) -> dict:
    """
    Compute reliability diagram data for ensemble calibration assessment.

    For each probability bin [p_k, p_{k+1}], compute:
    - Predicted probability (histogram fraction in bin)
    - Actual frequency (fraction of observations in bin)

    ensemble_members: (K, C, H, W)
    observations: (C, H, W)

    Returns dict with 'bins', 'predicted', 'actual', 'counts'
    """
    K = ensemble_members.shape[0]

    # Ensemble CDF values for each grid point
    # For each point, the "probability" is the rank fraction
    ens_mean = np.mean(ensemble_members, axis=0)

    # Flatten for binning
    ens_flat = ens_mean.ravel()
    obs_flat = observations.ravel()

    # Compute bins
    percentiles = np.linspace(0, 100, n_bins + 1)
    bin_edges = np.percentile(ens_flat, percentiles)

    predicted_freqs = []
    actual_freqs = []
    counts = []

    for i in range(n_bins):
        lo = bin_edges[i]
        hi = bin_edges[i + 1] if i < n_bins - 1 else ens_flat.max() + 1e-6

        in_bin = (ens_flat >= lo) & (ens_flat < hi)
        count = in_bin.sum()

        if count > 0:
            predicted = count / len(ens_flat)
            actual = (obs_flat[in_bin] >= lo).sum() / count
        else:
            predicted = 0.0
            actual = 0.0

        predicted_freqs.append(predicted)
        actual_freqs.append(actual)
        counts.append(count)

    return {
        "bin_centers": (bin_edges[:-1] + bin_edges[1:]) / 2,
        "predicted": np.array(predicted_freqs),
        "actual": np.array(actual_freqs),
        "counts": np.array(counts),
    }


def compute_ensemble_entropy(
    ensemble_members: np.ndarray,
    n_bins: int = 20,
) -> float:
    """
    Compute the predictive entropy of the ensemble.

    Higher entropy = more uncertain ensemble
    Lower entropy = more confident (but possibly wrong) ensemble

    ensemble_members: (K, ...) — K ensemble members at each spatial location
    Returns: mean spatial entropy
    """
    K = ensemble_members.shape[0]

    # Discretize ensemble into bins
    all_vals = ensemble_members.ravel()
    bin_edges = np.percentile(all_vals, np.linspace(0, 100, n_bins + 1))

    # For each spatial location, compute histogram of ensemble values
    # Then compute entropy: -Σ p_k * log(p_k)

    spatial_shape = ensemble_members.shape[1:]
    total_entropy = 0.0
    n_points = 0

    # Vectorized approach
    ens_flat = ensemble_members.reshape(K, -1)  # (K, N)
    N = ens_flat.shape[1]

    # Count occurrences in each bin for each spatial point
    # (K, N) -> (N, K) -> digitize
    digitized = np.digitize(ens_flat.T, bin_edges[1:-1])  # (N, K)

    for n in range(N):
        counts = np.bincount(digitized[n], minlength=n_bins)
        probs = counts / K
        probs = probs[probs > 0]
        if len(probs) > 0:
            entropy = -np.sum(probs * np.log(probs + 1e-10))
            total_entropy += entropy
            n_points += 1

    return total_entropy / max(n_points, 1)
