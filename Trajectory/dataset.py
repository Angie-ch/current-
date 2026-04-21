"""
LT3P风格数据集：48小时历史轨迹 + 72小时未来ERA5 → 72小时未来轨迹
时间分辨率：3小时
"""
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Dict
import random

from config import model_cfg, train_cfg, data_cfg
from data_structures import StormSample


# ============== ERA5 逐通道标准化常量 ==============
# 通道顺序 (9ch, 与扩散模型完全一致):
#   u_850, u_500, u_250, v_850, v_500, v_250,
#   z_850, z_500, z_250
#
# 数据来源: 扩散模型 newtry/norm_stats.pt (数据驱动统计)
# 两个模型必须使用完全相同的归一化统计，避免级联偏差

def _load_norm_stats():
    """加载归一化统计量（优先从 norm_stats.pt，回退到硬编码值）"""
    expected_channels = model_cfg.era5_channels  # 当前配置的通道数 (9)

    # 尝试从扩散模型的 norm_stats.pt 加载（保证完全一致）
    possible_paths = [
        os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'newtry', 'norm_stats.pt'),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), 'norm_stats.pt'),
    ]

    for path in possible_paths:
        if os.path.exists(path):
            stats = torch.load(path, weights_only=True, map_location='cpu')
            mean = stats['mean'].numpy()
            std = stats['std'].numpy()
            loaded_ch = len(mean)
            if loaded_ch != expected_channels:
                raise ValueError(
                    f"norm_stats.pt 通道数不匹配! 加载到 {loaded_ch}ch, 但配置期望 {expected_channels}ch。"
                    f"请用新的扩散模型重新训练生成 norm_stats.pt (路径: {os.path.abspath(path)})"
                )
            print(f"[dataset.py] 从 {os.path.abspath(path)} 加载归一化统计 ({loaded_ch}ch)")
            return mean, std

    # 回退: 硬编码与扩散模型 norm_stats.pt 一致的值 (9通道)
    # 这些值来自诊断脚本读取的 newtry/norm_stats.pt
    print("[dataset.py] 警告: 未找到 norm_stats.pt，使用硬编码回退值")
    mean = np.array([
        # u (m/s): 850, 500, 250
        -1.29, -0.28, 0.39,
        # v (m/s): 850, 500, 250
        1.74, 2.38, 2.60,
        # z (m²/s²): 850, 500, 250
        14253.13, 56708.52, 106498.13,
    ], dtype=np.float32)

    std = np.array([
        # u (m/s)
        10.02, 10.24, 12.65,
        # v (m/s)
        9.04, 8.48, 8.56,
        # z (m²/s²)
        1320.93, 4869.39, 9103.74,
    ], dtype=np.float32)

    return mean, std


ERA5_CHANNEL_MEAN, ERA5_CHANNEL_STD = _load_norm_stats()


def normalize_era5(era5: np.ndarray) -> np.ndarray:
    """
    对 ERA5 数据做逐通道 z-score 标准化
    Args:
        era5: (T, C, H, W) 或 (C, H, W)
    Returns:
        标准化后的数组，同形状
    """
    if era5.ndim == 4:
        # (T, C, H, W)
        C = era5.shape[1]
        mean = ERA5_CHANNEL_MEAN[:C].reshape(1, C, 1, 1)
        std = ERA5_CHANNEL_STD[:C].reshape(1, C, 1, 1)
    elif era5.ndim == 3:
        # (C, H, W)
        C = era5.shape[0]
        mean = ERA5_CHANNEL_MEAN[:C].reshape(C, 1, 1)
        std = ERA5_CHANNEL_STD[:C].reshape(C, 1, 1)
    else:
        return era5
    return (era5 - mean) / (std + 1e-8)


def normalize_coords(lat, lon, lat_range=None, lon_range=None):
    """归一化坐标到 [0, 1] 范围"""
    if lat_range is None:
        lat_range = data_cfg.lat_range
    if lon_range is None:
        lon_range = data_cfg.lon_range
    lat_norm = (lat - lat_range[0]) / (lat_range[1] - lat_range[0])
    lon_norm = (lon - lon_range[0]) / (lon_range[1] - lon_range[0])
    return lat_norm, lon_norm


def denormalize_coords(lat_norm, lon_norm, lat_range=None, lon_range=None):
    """反归一化坐标"""
    if lat_range is None:
        lat_range = data_cfg.lat_range
    if lon_range is None:
        lon_range = data_cfg.lon_range
    lat = lat_norm * (lat_range[1] - lat_range[0]) + lat_range[0]
    lon = lon_norm * (lon_range[1] - lon_range[0]) + lon_range[0]
    return lat, lon


class LT3PDataset(Dataset):
    """
    LT3P风格台风数据集
    
    输入:
        - history_coords: (T_history, 2) 过去48小时轨迹 [lat, lon]
        - future_era5: (T_future, C, H, W) 未来72小时ERA5气象场
    
    输出:
        - target_coords: (T_future, 2) 未来72小时轨迹 [lat, lon]
    
    时间配置:
        - T_history = 16 (48h / 3h)
        - T_future = 24 (72h / 3h)
        - 总窗口 = 48 + 72 = 120小时 = 40个时间步
    """
    
    def __init__(
        self,
        storm_samples: List[StormSample],
        t_history: int = None,
        t_future: int = None,
        stride: int = 1,
        era5_channels: int = None,
        time_resolution_hours: int = None,
    ):
        self.storm_samples = storm_samples
        self.t_history = t_history or model_cfg.t_history  # 16
        self.t_future = t_future or model_cfg.t_future      # 24
        self.stride = stride
        self.era5_channels = era5_channels or model_cfg.era5_channels  # 9
        self.time_resolution_hours = time_resolution_hours or data_cfg.time_resolution_hours  # 3
        
        # 总窗口长度
        self.total_length = self.t_history + self.t_future  # 16 + 24 = 40
        
        # 构建索引
        self.samples_index = self._build_samples_index()
        
        print(f"LT3PDataset: t_history={self.t_history}, t_future={self.t_future}, "
              f"total_length={self.total_length}, samples={len(self.samples_index)}")
    
    def _build_samples_index(self) -> List[Tuple[int, int]]:
        """构建滑动窗口索引"""
        index = []
        
        for storm_idx, sample in enumerate(self.storm_samples):
            T = len(sample)
            # 需要足够长的序列：至少 t_history + t_future 个时间步
            if T < self.total_length:
                continue
            
            for start in range(0, T - self.total_length + 1, self.stride):
                index.append((storm_idx, start))
        
        return index
    
    def __len__(self) -> int:
        return len(self.samples_index)
    
    def _get_era5_video(self, sample: StormSample, start: int, end: int) -> np.ndarray:
        """获取ERA5视频数据"""
        T = end - start
        
        if sample.era5_array is not None:
            # 直接从预加载的数组获取
            era5 = sample.era5_array[start:end]
            # 确保通道数正确（截取前era5_channels个通道）
            if era5.shape[1] > self.era5_channels:
                era5 = era5[:, :self.era5_channels]
            elif era5.shape[1] < self.era5_channels:
                # 填充零
                pad = np.zeros((T, self.era5_channels - era5.shape[1], 
                               era5.shape[2], era5.shape[3]), dtype=np.float32)
                era5 = np.concatenate([era5, pad], axis=1)
            return era5
        elif sample.era5_dataset is not None:
            # 从xarray Dataset获取
            frames = []
            for t in range(start, end):
                frame = sample.get_era5_at_time(t)
                if frame is not None:
                    if frame.shape[0] > self.era5_channels:
                        frame = frame[:self.era5_channels]
                    frames.append(frame)
                else:
                    frames.append(np.zeros((self.era5_channels, 
                                           data_cfg.grid_height, 
                                           data_cfg.grid_width), dtype=np.float32))
            return np.stack(frames, axis=0)
        else:
            # 返回虚拟数据
            return np.zeros((T, self.era5_channels, 
                           data_cfg.grid_height, data_cfg.grid_width), dtype=np.float32)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        storm_idx, start_idx = self.samples_index[idx]
        sample = self.storm_samples[storm_idx]
        
        # 计算索引范围
        history_start = start_idx
        history_end = start_idx + self.t_history
        future_start = history_end
        future_end = history_end + self.t_future
        
        # === 历史轨迹 (过去48小时) ===
        history_lat = sample.track_lat[history_start:history_end]
        history_lon = sample.track_lon[history_start:history_end]
        
        # 归一化
        h_lat_n, h_lon_n = normalize_coords(history_lat, history_lon)
        history_coords = np.stack([h_lat_n, h_lon_n], axis=-1)  # (T_history, 2)
        
        # === 未来ERA5气象场 (未来72小时) ===
        # 关键：使用未来时间段的ERA5数据作为条件输入
        future_era5 = self._get_era5_video(sample, future_start, future_end)  # (T_future, C, H, W)
        # 逐通道标准化（消除 z 与 wind 的量级差异）
        future_era5 = normalize_era5(future_era5)

        # === 过去ERA5气象场 (过去48小时) ===
        past_era5 = self._get_era5_video(sample, history_start, history_end)  # (T_history, C, H, W)
        past_era5 = normalize_era5(past_era5)
        
        # === 未来轨迹目标 (未来72小时) ===
        future_lat = sample.track_lat[future_start:future_end]
        future_lon = sample.track_lon[future_start:future_end]
        
        # 归一化
        f_lat_n, f_lon_n = normalize_coords(future_lat, future_lon)
        target_coords = np.stack([f_lat_n, f_lon_n], axis=-1)  # (T_future, 2)
        
        # 样本权重（基于真实数据比例）
        sample_weight = 1.0
        if sample.is_real is not None:
            real_ratio = sample.is_real[future_start:future_end].mean()
            sample_weight = train_cfg.interp_sample_weight + \
                real_ratio * (train_cfg.real_sample_weight - train_cfg.interp_sample_weight)
        
        return {
            'history_coords': torch.from_numpy(history_coords).float(),
            'past_era5': torch.from_numpy(past_era5).float(),
            'future_era5': torch.from_numpy(future_era5).float(),
            'target_coords': torch.from_numpy(target_coords).float(),
            'sample_weight': torch.tensor(sample_weight).float(),
            'storm_id': sample.storm_id,
            # 保留原始坐标用于评估
            'target_lat_raw': torch.from_numpy(future_lat).float(),
            'target_lon_raw': torch.from_numpy(future_lon).float(),
            'history_lat_raw': torch.from_numpy(history_lat).float(),
            'history_lon_raw': torch.from_numpy(history_lon).float(),
        }


def split_storms_by_id(
    storm_samples: List[StormSample],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    seed: int = 42
) -> Tuple[List[StormSample], List[StormSample], List[StormSample]]:
    """按storm_id划分训练/验证/测试集"""
    random.seed(seed)
    samples = storm_samples.copy()
    random.shuffle(samples)
    
    n = len(samples)
    
    if n <= 3:
        return samples, [], []
    elif n <= 6:
        n_train = n - 2
        n_val = 1
    else:
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        if n_val == 0:
            n_val = 1
        if n_train + n_val >= n:
            n_train = n - n_val - 1
    
    train_samples = samples[:n_train]
    val_samples = samples[n_train:n_train + n_val]
    test_samples = samples[n_train + n_val:]
    
    return train_samples, val_samples, test_samples


def split_storms_by_year(
    storm_samples: List[StormSample],
    train_years: List[int],
    val_years: List[int],
    test_years: List[int]
) -> Tuple[List[StormSample], List[StormSample], List[StormSample]]:
    """按年份划分"""
    train_samples = [s for s in storm_samples if s.year in train_years]
    val_samples = [s for s in storm_samples if s.year in val_years]
    test_samples = [s for s in storm_samples if s.year in test_years]
    return train_samples, val_samples, test_samples


def filter_short_storms(
    storm_samples: List[StormSample],
    min_duration_hours: int = 120,
    time_resolution_hours: int = 3
) -> List[StormSample]:
    """过滤掉持续时间太短的台风"""
    min_steps = min_duration_hours // time_resolution_hours
    filtered = [s for s in storm_samples if len(s) >= min_steps]
    print(f"Filtered storms: {len(storm_samples)} -> {len(filtered)} "
          f"(min duration: {min_duration_hours}h = {min_steps} steps)")
    return filtered


def filter_out_of_range_storms(
    storm_samples: List[StormSample],
    lat_range: tuple = None,
    lon_range: tuple = None,
) -> List[StormSample]:
    """过滤掉轨迹坐标超出归一化范围的台风（非目标洋区）"""
    if lat_range is None:
        lat_range = data_cfg.lat_range
    if lon_range is None:
        lon_range = data_cfg.lon_range

    filtered = []
    removed_ids = []
    for s in storm_samples:
        lat_ok = (s.track_lat.min() >= lat_range[0]) and (s.track_lat.max() <= lat_range[1])
        lon_ok = (s.track_lon.min() >= lon_range[0]) and (s.track_lon.max() <= lon_range[1])
        if lat_ok and lon_ok:
            filtered.append(s)
        else:
            removed_ids.append(s.storm_id)

    print(f"Filtered out-of-range storms: {len(storm_samples)} -> {len(filtered)} "
          f"(removed {len(removed_ids)} storms outside lat{lat_range} lon{lon_range})")
    return filtered


def create_dataloaders(
    storm_samples: List[StormSample],
    batch_size: int = None,
    split_by: str = 'storm_id',
    **split_kwargs
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """创建训练/验证/测试DataLoader"""
    batch_size = batch_size or train_cfg.batch_size
    
    # 过滤太短的台风
    min_duration = train_cfg.min_typhoon_duration_hours
    storm_samples = filter_short_storms(storm_samples, min_duration)

    # 过滤非目标洋区台风
    storm_samples = filter_out_of_range_storms(storm_samples)
    
    # 划分数据
    if split_by == 'storm_id':
        train_s, val_s, test_s = split_storms_by_id(
            storm_samples,
            train_cfg.train_ratio,
            train_cfg.val_ratio
        )
    else:
        train_s, val_s, test_s = split_storms_by_year(storm_samples, **split_kwargs)
    
    # 创建Dataset
    train_ds = LT3PDataset(train_s, stride=1)
    val_ds = LT3PDataset(val_s, stride=model_cfg.t_future)  # 验证集不重叠
    test_ds = LT3PDataset(test_s, stride=model_cfg.t_future)  # 测试集不重叠
    
    # 创建DataLoader
    num_workers = train_cfg.num_workers
    pin_memory = train_cfg.pin_memory
    
    train_loader = DataLoader(
        train_ds, batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
        drop_last=True
    )
    val_loader = DataLoader(
        val_ds, batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
        persistent_workers=num_workers > 0
    )
    test_loader = DataLoader(
        test_ds, batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
        persistent_workers=num_workers > 0
    )
    
    print(f"Dataset sizes - Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")
    print(f"Storms - Train: {len(train_s)}, Val: {len(val_s)}, Test: {len(test_s)}")
    
    return train_loader, val_loader, test_loader
