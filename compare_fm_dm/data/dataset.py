"""
统一数据加载模块 — FM和DM共用

支持:
- 预处理NPY模式 (快速)
- 原始NC模式 (兼容)
"""
import os
import glob
import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

logger = logging.getLogger(__name__)


def _extract_timestamp_from_filename(filename: str) -> str:
    basename = os.path.basename(filename)
    parts = basename.replace(".nc", "").split("_")
    return parts[2]


def _load_single_nc_to_array(
    filepath: str,
    pl_vars: List[str],
    sfc_vars: List[str],
    pressure_levels: List[int],
) -> np.ndarray:
    import netCDF4 as nc

    ds = nc.Dataset(filepath, "r")
    n_times = ds.dimensions["valid_time"].size
    H = ds.dimensions["latitude"].size
    W = ds.dimensions["longitude"].size
    n_levels = len(pressure_levels)
    n_channels = len(pl_vars) * n_levels + len(sfc_vars)

    result = np.empty((n_times, n_channels, H, W), dtype=np.float32)
    ch = 0

    available_levels = ds.variables["pressure_level"][:].astype(float)
    level_indices = []
    for target_level in pressure_levels:
        idx = int(np.argmin(np.abs(available_levels - target_level)))
        level_indices.append(idx)

    for var_name in pl_vars:
        data = ds.variables[var_name][:].astype(np.float32)
        for lev_idx in level_indices:
            result[:, ch, :, :] = data[:, lev_idx, :, :]
            ch += 1

    for var_name in sfc_vars:
        data = ds.variables[var_name][:].astype(np.float32)
        result[:, ch, :, :] = data
        ch += 1

    ds.close()
    return result


def _crop_pad_to_target(data: np.ndarray, target_H: int = 40, target_W: int = 40) -> np.ndarray:
    if data.ndim == 3:
        C, H, W = data.shape
        if H == target_H and W == target_W:
            return data
        padded = np.zeros((C, target_H, target_W), dtype=np.float32)
        h = min(H, target_H)
        w = min(W, target_W)
        padded[:, :h, :w] = data[:, :h, :w]
        return padded
    elif data.ndim == 4:
        T, C, H, W = data.shape
        if H == target_H and W == target_W:
            return data
        padded = np.zeros((T, C, target_H, target_W), dtype=np.float32)
        h = min(H, target_H)
        w = min(W, target_W)
        padded[:, :, :h, :w] = data[:, :, :h, :w]
        return padded
    else:
        raise ValueError(f"Unsupported data dim: {data.ndim}")


class ERA5Dataset(Dataset):
    """
    统一ERA5数据集 — 支持预处理NPY模式和原始NC模式

    台风中心纬度通过 processed_typhoon_tracks.csv 加载，用于 GeostrophicBalanceLoss
    计算每个样本的 Coriolis 参数 f = 2Ωsin(lat)。
    """

    def __init__(
        self,
        typhoon_ids: List[str],
        data_root: str,
        pl_vars: List[str],
        sfc_vars: List[str],
        pressure_levels: List[int],
        history_steps: int = 16,
        forecast_steps: int = 1,
        norm_mean: Optional[np.ndarray] = None,
        norm_std: Optional[np.ndarray] = None,
        preprocessed_dir: Optional[str] = None,
        track_csv_path: Optional[str] = None,
        default_center_lat: float = 20.0,
    ):
        super().__init__()
        self.data_root = data_root
        self.pl_vars = pl_vars
        self.sfc_vars = sfc_vars
        self.pressure_levels = pressure_levels
        self.n_levels = len(pressure_levels)
        self.history_steps = history_steps
        self.forecast_steps = forecast_steps
        self.window_size = history_steps + forecast_steps
        self.norm_mean = norm_mean
        self.norm_std = norm_std
        self.preprocessed_dir = preprocessed_dir

        self.num_channels = len(pl_vars) * self.n_levels + len(sfc_vars)
        self.samples = []
        self._mmap_cache: Dict[str, np.ndarray] = {}
        self.default_center_lat = default_center_lat

        # Load center lats from track CSV for GeostrophicBalanceLoss
        self._track_lats: Dict[str, np.ndarray] = {}
        if track_csv_path:
            self._load_track_lats(track_csv_path)

        if preprocessed_dir:
            self._build_index_preprocessed(typhoon_ids)
        else:
            self._build_index_nc(typhoon_ids)

    def _load_track_lats(self, csv_path: str):
        """Load center lats for each typhoon timestep from the track CSV."""
        import pandas as pd
        df = pd.read_csv(csv_path)
        for tid, group in df.groupby("typhoon_id"):
            # Sort by row order (already in time order) and extract lats
            self._track_lats[tid] = group["lat"].values.astype(np.float32)
        logger.info(f"Loaded center lats for {len(self._track_lats)} typhoons from {csv_path}")

    def _build_index_preprocessed(self, typhoon_ids: List[str]):
        total_samples = 0
        for tid in tqdm(sorted(typhoon_ids), desc="扫描预处理数据", unit="个"):
            npy_path = os.path.join(self.preprocessed_dir, f"{tid}.npy")
            if not os.path.exists(npy_path):
                continue

            data = np.load(npy_path, mmap_mode='r')
            n_steps = data.shape[0]

            if n_steps < self.window_size:
                continue

            for i in range(n_steps - self.window_size + 1):
                self.samples.append((tid, i))
                total_samples += 1

        logger.info(f"数据集构建完成 (NPY模式): {len(typhoon_ids)} 个台风, {total_samples} 个样本")

    def _build_index_nc(self, typhoon_ids: List[str]):
        total_samples = 0
        for tid in tqdm(sorted(typhoon_ids), desc="扫描台风目录", unit="个"):
            typhoon_dir = os.path.join(self.data_root, tid)
            if not os.path.isdir(typhoon_dir):
                continue

            nc_files = sorted(glob.glob(os.path.join(typhoon_dir, "era5_merged_*.nc")))
            if len(nc_files) < self.window_size:
                continue

            for i in range(len(nc_files) - self.window_size + 1):
                window_files = nc_files[i : i + self.window_size]
                self.samples.append((tid, window_files))
                total_samples += 1

        logger.info(f"数据集构建完成 (NC模式): {len(typhoon_ids)} 个台风, {total_samples} 个样本")

    def _get_mmap(self, tid: str) -> np.ndarray:
        if tid not in self._mmap_cache:
            npy_path = os.path.join(self.preprocessed_dir, f"{tid}.npy")
            self._mmap_cache[tid] = np.load(npy_path, mmap_mode='r')
        return self._mmap_cache[tid]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]

        if self.preprocessed_dir:
            return self._getitem_preprocessed(sample)
        else:
            return self._getitem_nc(sample)

    def _getitem_preprocessed(self, sample) -> Dict[str, torch.Tensor]:
        tid, start_idx = sample
        mmap_data = self._get_mmap(tid)

        all_data = np.array(
            mmap_data[start_idx : start_idx + self.window_size]
        )

        condition_data = all_data[: self.history_steps]
        target_data = all_data[self.history_steps :]

        if self.norm_mean is not None and self.norm_std is not None:
            mean = self.norm_mean[None, :, None, None]
            std = self.norm_std[None, :, None, None]
            std = np.where(std < 1e-8, 1.0, std)
            condition_data = (condition_data - mean) / std
            target_data = (target_data - mean) / std

        condition = condition_data.reshape(-1, *condition_data.shape[2:])
        target = target_data.reshape(-1, *target_data.shape[2:])

        # Center latitude: mean over the forecast window (for geostrophic balance loss)
        if tid in self._track_lats:
            track_lats = self._track_lats[tid]
            window_lats = track_lats[start_idx : start_idx + self.window_size]
            center_lat = float(window_lats.mean())
        else:
            center_lat = self.default_center_lat  # default to Western Pacific center

        return {
            "condition": torch.from_numpy(condition).float(),
            "target": torch.from_numpy(target).float(),
            "typhoon_id": tid,
            "center_lats": torch.tensor(center_lat, dtype=torch.float32),
        }

    def _getitem_nc(self, sample) -> Dict[str, torch.Tensor]:
        tid, window_files = sample

        steps = []
        for f in window_files:
            try:
                arr = _load_single_nc_to_array(f, self.pl_vars, self.sfc_vars, self.pressure_levels)
                step_data = arr[0]
                step_data = _crop_pad_to_target(step_data, 40, 40)
                steps.append(step_data)
            except Exception as e:
                logger.warning(f"加载失败 {f}: {e}, 使用零填充")
                steps.append(np.zeros((self.num_channels, 40, 40), dtype=np.float32))

        all_data = np.stack(steps, axis=0)
        condition_data = all_data[: self.history_steps]
        target_data = all_data[self.history_steps :]

        if self.norm_mean is not None and self.norm_std is not None:
            mean = self.norm_mean[None, :, None, None]
            std = self.norm_std[None, :, None, None]
            std = np.where(std < 1e-8, 1.0, std)
            condition_data = (condition_data - mean) / std
            target_data = (target_data - mean) / std

        condition = condition_data.reshape(-1, *condition_data.shape[2:])
        target = target_data.reshape(-1, *target_data.shape[2:])

        # Center latitude: mean over the forecast window
        if tid in self._track_lats:
            track_lats = self._track_lats[tid]
            # The NC mode doesn't have step-level indexing, so use the first track entry
            center_lat = float(track_lats[0]) if len(track_lats) > 0 else 20.0
        else:
            center_lat = self.default_center_lat

        return {
            "condition": torch.from_numpy(condition).float(),
            "target": torch.from_numpy(target).float(),
            "typhoon_id": tid,
            "center_lats": torch.tensor(center_lat, dtype=torch.float32),
        }


def compute_normalization_stats(
    typhoon_ids: List[str],
    data_root: str,
    pl_vars: List[str],
    sfc_vars: List[str],
    pressure_levels: List[int],
    max_files_per_typhoon: int = 50,
    preprocessed_dir: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """使用Welford在线算法计算全量数据的mean/std"""
    n_levels = len(pressure_levels)
    n_channels = len(pl_vars) * n_levels + len(sfc_vars)
    count = np.zeros(n_channels, dtype=np.float64)
    mean = np.zeros(n_channels, dtype=np.float64)
    M2 = np.zeros(n_channels, dtype=np.float64)

    for tid in tqdm(sorted(typhoon_ids), desc="计算归一化统计", unit="个台风"):
        if preprocessed_dir:
            npy_path = os.path.join(preprocessed_dir, f"{tid}.npy")
            if not os.path.exists(npy_path):
                continue
            data = np.load(npy_path, mmap_mode='r')
            n_steps = data.shape[0]
            if n_steps > max_files_per_typhoon:
                indices = np.linspace(0, n_steps - 1, max_files_per_typhoon, dtype=int)
            else:
                indices = range(n_steps)

            for step_idx in indices:
                step = np.array(data[step_idx])
                for c in range(n_channels):
                    channel_data = step[c].ravel().astype(np.float64)
                    valid = channel_data[~np.isnan(channel_data)]
                    if len(valid) == 0:
                        continue
                    n = len(valid)
                    batch_mean = valid.mean()
                    batch_var = valid.var()
                    delta = batch_mean - mean[c]
                    new_count = count[c] + n
                    mean[c] += delta * n / new_count
                    M2[c] += batch_var * n + delta ** 2 * count[c] * n / new_count
                    count[c] = new_count
        else:
            typhoon_dir = os.path.join(data_root, tid)
            if not os.path.isdir(typhoon_dir):
                continue

            nc_files = sorted(glob.glob(os.path.join(typhoon_dir, "era5_merged_*.nc")))
            if len(nc_files) > max_files_per_typhoon:
                indices = np.linspace(0, len(nc_files) - 1, max_files_per_typhoon, dtype=int)
                nc_files = [nc_files[j] for j in indices]

            for filepath in nc_files:
                try:
                    arr = _load_single_nc_to_array(filepath, pl_vars, sfc_vars, pressure_levels)
                    step = arr[0]
                    step = _crop_pad_to_target(step, 40, 40)

                    for c in range(n_channels):
                        channel_data = step[c].ravel().astype(np.float64)
                        valid = channel_data[~np.isnan(channel_data)]
                        if len(valid) == 0:
                            continue
                        n = len(valid)
                        batch_mean = valid.mean()
                        batch_var = valid.var()
                        delta = batch_mean - mean[c]
                        new_count = count[c] + n
                        mean[c] += delta * n / new_count
                        M2[c] += batch_var * n + delta ** 2 * count[c] * n / new_count
                        count[c] = new_count
                except Exception as e:
                    logger.warning(f"统计时跳过 {filepath}: {e}")

    std = np.sqrt(M2 / np.maximum(count - 1, 1))
    return mean.astype(np.float32), std.astype(np.float32)


def split_typhoon_ids_by_year(
    data_root: str,
    train_years: Tuple[int, int],
    val_years: Tuple[int, int],
    test_years: Tuple[int, int],
) -> Tuple[List[str], List[str], List[str]]:
    """
    Year-based typhoon split — no data leakage.

    Split by extracting the 4-digit year prefix from each typhoon ID filename
    (e.g. "2019001N09151.npy" -> year 2019).

    Args:
        data_root: directory containing .npy typhoon files
        train_years: (start, end) inclusive range for training
        val_years:   (start, end) inclusive range for validation
        test_years:  (start, end) inclusive range for test

    Returns:
        (train_ids, val_ids, test_ids) — typhoon IDs (without .npy extension)
    """
    all_files = sorted(glob.glob(os.path.join(data_root, '*.npy')))
    data_files = [f for f in all_files if not os.path.basename(f).endswith('_times.npy')]

    train_set = set(range(train_years[0], train_years[1] + 1))
    val_set   = set(range(val_years[0],   val_years[1]   + 1))
    test_set  = set(range(test_years[0],  test_years[1]  + 1))

    train_ids, val_ids, test_ids = [], [], []

    for fpath in data_files:
        basename = os.path.basename(fpath)
        typhoon_id = os.path.splitext(basename)[0]
        try:
            year = int(typhoon_id[:4])
        except ValueError:
            logger.warning(f"Cannot extract year from filename: {basename}, skipping")
            continue

        if year in train_set:
            train_ids.append(typhoon_id)
        elif year in val_set:
            val_ids.append(typhoon_id)
        elif year in test_set:
            test_ids.append(typhoon_id)
        else:
            logger.warning(f"Year {year} from {basename} not in any split range, skipping")

    logger.info(
        f"Year-based split: train={len(train_ids)} typhoons "
        f"(years {train_years[0]}-{train_years[1]}), "
        f"val={len(val_ids)} typhoons "
        f"(years {val_years[0]}-{val_years[1]}), "
        f"test={len(test_ids)} typhoons "
        f"(years {test_years[0]}-{test_years[1]})"
    )
    return train_ids, val_ids, test_ids


def build_dataloaders(
    data_cfg,
    train_cfg=None,
    norm_mean: Optional[np.ndarray] = None,
    norm_std: Optional[np.ndarray] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader, np.ndarray, np.ndarray]:
    """
    Build train/val/test DataLoaders with year-based split.

    Normalization stats are computed on the TRAINING SET ONLY to prevent data leakage.
    If norm_stats_path exists, load from disk; otherwise compute from training data.
    """
    preprocessed_dir = getattr(data_cfg, 'preprocessed_dir', None) or getattr(
        data_cfg, 'era5_dir', getattr(data_cfg, 'data_root', None)
    )

    # Year-based split
    train_years = data_cfg.train_years
    val_years   = data_cfg.val_years
    test_years  = data_cfg.test_years

    train_ids, val_ids, test_ids = split_typhoon_ids_by_year(
        preprocessed_dir,
        train_years,
        val_years,
        test_years,
    )

    # Normalization: MUST use training set only
    if norm_mean is None or norm_std is None:
        stats_path = getattr(data_cfg, 'norm_stats_path', None) or ""
        if stats_path and os.path.exists(stats_path):
            logger.info(f"Loading normalization stats: {stats_path}")
            stats = torch.load(stats_path, weights_only=True)
            norm_mean = stats["mean"].numpy()
            norm_std  = stats["std"].numpy()
        else:
            logger.info(
                f"Computing normalization stats from TRAINING SET ONLY "
                f"(years {train_years[0]}-{train_years[1]}) to prevent data leakage..."
            )
            norm_mean, norm_std = compute_normalization_stats(
                train_ids,
                preprocessed_dir,
                data_cfg.pressure_level_vars,
                data_cfg.surface_vars,
                data_cfg.pressure_levels,
                preprocessed_dir=preprocessed_dir,
            )
            if stats_path:
                os.makedirs(os.path.dirname(stats_path) or ".", exist_ok=True)
                torch.save(
                    {"mean": torch.from_numpy(norm_mean), "std": torch.from_numpy(norm_std)},
                    stats_path,
                )
                logger.info(f"Normalization stats saved: {stats_path}")

    batch_size = train_cfg.batch_size if train_cfg else 16

    common_kwargs = dict(
        data_root=preprocessed_dir,
        pl_vars=data_cfg.pressure_level_vars,
        sfc_vars=data_cfg.surface_vars,
        pressure_levels=data_cfg.pressure_levels,
        history_steps=data_cfg.history_steps,
        forecast_steps=data_cfg.forecast_steps,
        norm_mean=norm_mean,
        norm_std=norm_std,
        preprocessed_dir=preprocessed_dir,
        track_csv_path=getattr(data_cfg, 'track_csv_path', None),
        default_center_lat=getattr(data_cfg, 'default_center_lat', 20.0),
    )

    train_dataset = ERA5Dataset(train_ids, **common_kwargs)
    val_dataset   = ERA5Dataset(val_ids,   **common_kwargs)
    test_dataset  = ERA5Dataset(test_ids,  **common_kwargs)

    logger.info(
        f"Dataset sizes: train={len(train_dataset)} samples, "
        f"val={len(val_dataset)} samples, test={len(test_dataset)} samples"
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=data_cfg.num_workers,
        pin_memory=data_cfg.pin_memory,
        prefetch_factor=getattr(data_cfg, 'prefetch_factor', 4),
        drop_last=True,
        persistent_workers=True if data_cfg.num_workers > 0 else False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=data_cfg.num_workers,
        pin_memory=data_cfg.pin_memory,
        drop_last=False,
        persistent_workers=True if data_cfg.num_workers > 0 else False,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=data_cfg.num_workers,
        pin_memory=data_cfg.pin_memory,
        drop_last=False,
    )

    return train_loader, val_loader, test_loader, norm_mean, norm_std
