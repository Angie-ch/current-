"""
基础数据结构定义
Video2Video 条件扩散模型 - 台风路径预测
"""
from dataclasses import dataclass, field
from typing import Optional, Dict, List
import numpy as np
import xarray as xr


@dataclass
class StormSample:
    """
    单个台风的完整数据样本
    包含轨迹信息和对应的 ERA5 环境场数据
    """
    storm_id: str                          # 台风编号
    times: np.ndarray                      # 时间序列 (T,) datetime64
    track_lat: np.ndarray                  # 台风中心纬度 (T,)
    track_lon: np.ndarray                  # 台风中心经度 (T,)
    track_vmax: np.ndarray                 # 最大风速 (T,)
    track_pmin: Optional[np.ndarray] = None  # 中心最低气压 (T,)
    
    # ERA5 数据（可以是 xarray 或已转换的 numpy 数组）
    era5_dataset: Optional[xr.Dataset] = None
    era5_array: Optional[np.ndarray] = None   # 形状: (T, C, H, W)
    
    # 网格坐标
    lat_grid: Optional[np.ndarray] = None     # 纬度网格 (H,)
    lon_grid: Optional[np.ndarray] = None     # 经度网格 (W,)
    
    # 标记：每个时间步是否为真实观测（3h间隔）还是插值数据
    is_real: Optional[np.ndarray] = None      # (T,) bool
    
    # 附加属性
    basin: Optional[str] = None               # 洋区
    year: Optional[int] = None                # 年份
    
    def __len__(self) -> int:
        return len(self.times)
    
    def get_era5_at_time(self, t_idx: int) -> Optional[np.ndarray]:
        """获取指定时间步的 ERA5 场数据"""
        if self.era5_array is not None:
            return self.era5_array[t_idx]
        elif self.era5_dataset is not None:
            # 从 Dataset 提取并转换
            return self._extract_era5_frame(t_idx)
        return None
    
    def _extract_era5_frame(self, t_idx: int) -> np.ndarray:
        """从 xarray Dataset 提取单帧，转换为 (C, H, W) 数组

        通道顺序与 ERA5-Diffusion 扩散模型一致:
        [u_850, u_500, u_250, v_850, v_500, v_250, z_850, z_500, z_250]
        """
        from config import data_cfg

        ds = self.era5_dataset.isel(valid_time=t_idx)
        channels = []

        # 提取 3D 变量（按配置的气压层逐层提取）
        for var in data_cfg.era5_3d_vars:
            if var in ds:
                for level in data_cfg.pressure_levels:
                    try:
                        # 选择指定气压层: (lat, lon)
                        data = ds[var].sel(pressure_level=level).values
                        channels.append(data[np.newaxis])  # (1, lat, lon)
                    except (KeyError, ValueError):
                        # 该气压层不存在, 用最近的层替代
                        if 'pressure_level' in ds[var].dims:
                            available = ds[var].pressure_level.values
                            nearest = available[np.argmin(np.abs(available - level))]
                            data = ds[var].sel(pressure_level=nearest).values
                            channels.append(data[np.newaxis])

        # 提取 2D 变量
        for var in data_cfg.era5_2d_vars:
            if var in ds:
                data = ds[var].values
                channels.append(data[np.newaxis])  # (1, lat, lon)

        if channels:
            return np.concatenate(channels, axis=0)  # (C, H, W)
        else:
            return np.zeros((len(data_cfg.era5_3d_vars) * len(data_cfg.pressure_levels) + len(data_cfg.era5_2d_vars),
                           data_cfg.grid_height, data_cfg.grid_width), dtype=np.float32)


@dataclass
class TrainingSample:
    """
    训练样本：混合式条件扩散
    输入轨迹坐标 + ERA5视频 + 环境特征 → 直接生成坐标增量
    """
    # === 条件输入 ===
    # 过去轨迹坐标序列
    cond_coords: np.ndarray         # (T_cond, 4) - lat, lon, vmax, pmin
    # ERA5 局地场视频
    cond_era5: np.ndarray           # (T_cond, C, H, W)
    # 环境特征序列
    cond_features: np.ndarray       # (T_cond, D)

    # === 目标输出 ===
    # 未来坐标增量序列（主输出）
    target_deltas: np.ndarray       # (T_future, 3) - Δlat, Δlon, Δvmax
    # 未来绝对坐标（用于辅助 heatmap 重建和评估）
    target_coords: np.ndarray       # (T_future, 4) - lat, lon, vmax, pmin

    # 元信息
    storm_id: Optional[str] = None
    start_time: Optional[np.datetime64] = None

    # 权重（真实样本 vs 插值样本）
    sample_weight: float = 1.0


@dataclass
class PredictionResult:
    """
    预测结果：直接坐标预测 + 可选的辅助 heatmap
    """
    # 预测的坐标增量
    predicted_deltas: np.ndarray     # (T_future, 3) - Δlat, Δlon, Δvmax

    # 恢复的绝对坐标
    predicted_lat: np.ndarray        # (T_future,)
    predicted_lon: np.ndarray        # (T_future,)
    predicted_vmax: np.ndarray       # (T_future,)

    # 辅助 heatmap（弱监督输出，可选）
    auxiliary_heatmap: Optional[np.ndarray] = None  # (T_future, 1, H, W)

    # 如果有多次采样（集合预测）
    all_samples: Optional[List[np.ndarray]] = None  # 每个是 (T_future, 3)

    # 不确定性估计
    uncertainty: Optional[np.ndarray] = None  # (T_future, 3)

    # 元信息
    storm_id: Optional[str] = None
    lead_times_hours: Optional[np.ndarray] = None


@dataclass
class EvaluationMetrics:
    """
    评估指标
    """
    # 距离误差（km）
    track_errors_km: np.ndarray           # (T_future,) 每个 lead time 的误差
    mean_track_error_km: float            # 平均轨迹误差
    
    # 分 lead time 的统计
    errors_by_lead_time: Dict[float, float] = field(default_factory=dict)
    
    # 强度误差（如果预测强度）
    vmax_errors: Optional[np.ndarray] = None
    
    # 多样本统计（如果进行集合预测）
    ensemble_spread: Optional[float] = None
    crps: Optional[float] = None  # Continuous Ranked Probability Score

