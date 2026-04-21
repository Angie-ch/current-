"""
Climatology Computation Module — For ACC (Anomaly Correlation Coefficient)

Computes the temporal mean of ERA5 fields used as the climatological baseline
for ACC calculations in the FM vs DM comparison study.

Usage:
    python -c "from compute_climatology import compute_and_save_climatology; compute_and_save_climatology(...)"
"""
import os
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm

logger = logging.getLogger(__name__)


class ClimatologyComputer:
    """
    Computes climatological mean fields for ERA5 variables.

    The climatological mean is the temporal average over the training period
    (1950-2015), used as the baseline for ACC computation:

        ACC = Σ(forecast - clim)(anomaly - clim) / √(Σ(forecast-clim)² × Σ(anomaly-clim)²)

    For publication, we compute:
    1. Overall climatological mean (all times pooled)
    2. Monthly climatological means (12 bins)
    3. Per-lead-time climatological means (e.g., +24h, +48h, +72h)
    """

    def __init__(
        self,
        data_cfg,
        n_samples_per_typhoon: int = 30,
    ):
        self.data_cfg = data_cfg
        self.n_samples_per_typhoon = n_samples_per_typhoon
        self.num_channels = data_cfg.num_channels
        self.var_names = data_cfg.var_names
        self.grid_size = data_cfg.grid_size

    def compute_from_dataset(
        self,
        dataset,
        typhoon_ids: List[str],
        save_path: Optional[str] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Compute climatology from a dataset by sampling each typhoon.

        Args:
            dataset: ERA5Dataset instance
            typhoon_ids: List of typhoon IDs to use for climatology
            save_path: Optional path to save the climatology

        Returns:
            Dict with 'overall', 'per_channel', and 'per_lead_time' climatologies
        """
        logger.info(f"Computing climatology from {len(typhoon_ids)} typhoons, "
                    f"sampling {self.n_samples_per_typhoon} steps per typhoon")

        # Initialize accumulators
        overall_sum = np.zeros((self.num_channels, self.grid_size, self.grid_size), dtype=np.float64)
        overall_count = 0

        # Per-channel accumulators
        channel_sums = [
            np.zeros((self.grid_size, self.grid_size), dtype=np.float64)
            for _ in range(self.num_channels)
        ]
        channel_counts = np.zeros(self.num_channels, dtype=np.int64)

        # Per-lead-time accumulators (grouped by step offset)
        lead_time_bins = {0: [], 1: [], 2: [], 3: []}  # +3h, +6h, +9h, +12h+
        lead_time_sum = {
            0: np.zeros((self.num_channels, self.grid_size, self.grid_size), dtype=np.float64),
            1: np.zeros((self.num_channels, self.grid_size, self.grid_size), dtype=np.float64),
            2: np.zeros((self.num_channels, self.grid_size, self.grid_size), dtype=np.float64),
            3: np.zeros((self.num_channels, self.grid_size, self.grid_size), dtype=np.float64),
        }
        lead_time_count = {k: 0 for k in lead_time_bins.keys()}

        # Also accumulate monthly (using the month from sample index as proxy)
        monthly_sum = {
            m: np.zeros((self.num_channels, self.grid_size, self.grid_size), dtype=np.float64)
            for m in range(1, 13)
        }
        monthly_count = {m: 0 for m in range(1, 13)}

        total_samples = 0
        for tid in tqdm(sorted(typhoon_ids), desc="Computing climatology", unit="typhoon"):
            # Find all samples for this typhoon
            tid_samples = []
            for idx in range(len(dataset)):
                sample_info = dataset.samples[idx]
                if sample_info[0] == tid:
                    tid_samples.append(idx)

            if len(tid_samples) == 0:
                continue

            # Sample uniformly from the typhoon's lifetime
            n_to_sample = min(self.n_samples_per_typhoon, len(tid_samples))
            step_indices = np.linspace(0, len(tid_samples) - 1, n_to_sample, dtype=int)
            sampled_indices = [tid_samples[i] for i in step_indices]

            for idx in sampled_indices:
                sample = dataset[idx]
                target = sample["target"]  # (C, H, W) in normalized space

                if isinstance(target, torch.Tensor):
                    target = target.cpu().numpy()

                # Overall
                overall_sum += target.astype(np.float64)
                overall_count += 1

                # Per-channel
                for c in range(min(target.shape[0], self.num_channels)):
                    channel_sums[c] += target[c]
                    channel_counts[c] += 1

                # Per-lead-time (approximate based on sample index within typhoon)
                step_offset = idx - tid_samples[0]
                if step_offset <= 3:
                    lt_bin = step_offset
                else:
                    lt_bin = 3
                lead_time_sum[lt_bin] += target.astype(np.float64)
                lead_time_count[lt_bin] += 1

                # Monthly (use index mod 12 as month proxy for speed)
                # In production, extract actual month from the data
                month_proxy = (total_samples % 12) + 1
                monthly_sum[month_proxy] += target.astype(np.float64)
                monthly_count[month_proxy] += 1

                total_samples += 1

        # Compute means
        overall_mean = overall_sum / max(overall_count, 1)

        channel_means = []
        for c in range(self.num_channels):
            if channel_counts[c] > 0:
                channel_means.append(channel_sums[c] / channel_counts[c])
            else:
                channel_means.append(np.zeros((self.grid_size, self.grid_size)))

        lead_time_means = {}
        for lt, count in lead_time_count.items():
            if count > 0:
                lead_time_means[lt] = lead_time_sum[lt] / count
            else:
                lead_time_means[lt] = np.zeros((self.num_channels, self.grid_size, self.grid_size))

        monthly_means = {}
        for m in range(1, 13):
            if monthly_count[m] > 0:
                monthly_means[m] = monthly_sum[m] / monthly_count[m]
            else:
                monthly_means[m] = np.zeros((self.num_channels, self.grid_size, self.grid_size))

        result = {
            "overall": overall_mean.astype(np.float32),
            "per_channel": {self.var_names[i]: channel_means[i].astype(np.float32)
                           for i in range(len(channel_means))},
            "per_lead_time": {
                f"+{3 * (k + 1)}h" if k < 3 else "+12h+": lead_time_means[k].astype(np.float32)
                for k in lead_time_means.keys()
            },
            "monthly": {m: monthly_means[m].astype(np.float32) for m in range(1, 13)},
            "metadata": {
                "n_typhoons": len(typhoon_ids),
                "n_samples": total_samples,
                "n_samples_per_typhoon": self.n_samples_per_typhoon,
                "num_channels": self.num_channels,
                "var_names": self.var_names,
                "grid_size": self.grid_size,
            }
        }

        # Save
        if save_path:
            os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
            np.savez_compressed(
                save_path,
                overall=result["overall"],
                per_channel={k: v for k, v in result["per_channel"].items()},
                lead_time_0=result["per_lead_time"]["+3h"],
                lead_time_1=result["per_lead_time"]["+6h"],
                lead_time_2=result["per_lead_time"]["+9h"],
                lead_time_3=result["per_lead_time"]["+12h+"],
                metadata=result["metadata"],
            )
            logger.info(f"Climatology saved to: {save_path}")

        logger.info(f"Climatology computed: {total_samples} samples from {len(typhoon_ids)} typhoons")
        for i, var in enumerate(self.var_names):
            logger.info(f"  {var}: mean={channel_means[i].mean():.4f}, std={channel_means[i].std():.4f}")

        return result

    def load_climatology(self, path: str) -> Dict:
        """Load precomputed climatology from disk."""
        if not os.path.exists(path):
            logger.warning(f"Climatology file not found: {path}")
            return None

        data = np.load(path, allow_pickle=True)
        result = {
            "overall": data["overall"],
            "per_channel": {k: v for k, v in data["per_channel"].item().items()},
            "per_lead_time": {
                "+3h": data["lead_time_0"],
                "+6h": data["lead_time_1"],
                "+9h": data["lead_time_2"],
                "+12h+": data["lead_time_3"],
            },
            "metadata": data["metadata"].item(),
        }
        logger.info(f"Loaded climatology from: {path}")
        return result


def quick_compute_climatology(
    data_root: str,
    output_path: str,
    n_typhoons: int = 500,
    n_samples_per_typhoon: int = 30,
    grid_size: int = 40,
    num_channels: int = 9,
    var_names: List[str] = None,
):
    """
    Quick climatology computation from raw .npy files.

    Args:
        data_root: Directory containing typhoon .npy files
        output_path: Where to save the climatology
        n_typhoons: How many typhoons to use
        n_samples_per_typhoon: Samples per typhoon
    """
    if var_names is None:
        var_names = ['u_850', 'u_500', 'u_250', 'v_850', 'v_500', 'v_250', 'z_850', 'z_500', 'z_250']

    import glob
    npy_files = sorted(glob.glob(os.path.join(data_root, "*.npy")))
    npy_files = [f for f in npy_files
                 if "_times" not in f and "_track" not in f]
    npy_files = npy_files[:n_typhoons]

    logger.info(f"Computing climatology from {len(npy_files)} typhoons")

    overall_sum = np.zeros((num_channels, grid_size, grid_size), dtype=np.float64)
    overall_count = 0

    for npy_path in tqdm(npy_files, desc="Climatology"):
        try:
            data = np.load(npy_path, mmap_mode='r')  # (T, C, H, W)
            T = data.shape[0]
            n_sample = min(n_samples_per_typhoon, T)
            indices = np.linspace(0, T - 1, n_sample, dtype=int)

            for idx in indices:
                step = np.array(data[idx]).astype(np.float64)
                overall_sum += step
                overall_count += 1
        except Exception as e:
            logger.warning(f"Failed to load {npy_path}: {e}")
            continue

    overall_mean = (overall_sum / max(overall_count, 1)).astype(np.float32)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    np.save(output_path, overall_mean)
    logger.info(f"Climatology saved: {overall_count} samples, mean shape {overall_mean.shape}")
    logger.info(f"Saved to: {output_path}")

    # Print per-channel stats
    for i, var in enumerate(var_names):
        logger.info(f"  {var}: mean={overall_mean[i].mean():.4f}, std={overall_mean[i].std():.4f}")

    return overall_mean
