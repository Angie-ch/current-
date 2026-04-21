"""
损失函数集合：物理约束等
"""
from typing import Optional
import torch
import torch.nn.functional as F

from config import data_cfg


def physics_loss(
    pred_deltas: torch.Tensor,
    cond_coords: torch.Tensor,
    cond_features: Optional[torch.Tensor] = None,
    dt_hours: float = 0.5,
) -> torch.Tensor:
    """
    台风运动的物理约束

    Args:
        pred_deltas: (B, T, 3) 预测的 Δlat, Δlon, Δvmax（归一化坐标系）
        cond_coords: (B, T_cond, 4) 历史坐标 [lat, lon, vmax, pmin]（归一化）
        cond_features: (可选) 历史特征序列
        dt_hours: 时间步长（默认 0.5h）

    Returns:
        标量 loss
    """
    # 取出归一化范围
    lat_range = data_cfg.lat_range
    lon_range = data_cfg.lon_range

    # 重建预测坐标（归一化）
    last_coord = cond_coords[:, -1:, :3]  # (B,1,3)
    pred_coords = last_coord + torch.cumsum(pred_deltas, dim=1)  # (B,T,3)
    pred_lat = pred_coords[..., 0]
    pred_lon = pred_coords[..., 1]
    pred_vmax = pred_coords[..., 2]

    # 反归一化到物理量
    lat_deg = pred_lat * (lat_range[1] - lat_range[0]) + lat_range[0]
    lon_deg = pred_lon * (lon_range[1] - lon_range[0]) + lon_range[0]

    dlat_deg = pred_deltas[..., 0] * (lat_range[1] - lat_range[0])
    dlon_deg = pred_deltas[..., 1] * (lon_range[1] - lon_range[0])

    # 速度（km/h）
    dist_lat_km = dlat_deg * 111.0
    dist_lon_km = dlon_deg * 111.0 * torch.cos(torch.deg2rad(lat_deg + 1e-6))
    speed_kmh = torch.sqrt(dist_lat_km**2 + dist_lon_km**2) / dt_hours

    # 1) 速度约束：>100 km/h 惩罚
    speed_penalty = F.relu(speed_kmh - 100.0)

    # 2) 平滑性约束：相邻速度差
    accel = torch.diff(speed_kmh, dim=1)
    smooth_penalty = torch.abs(accel)

    # 3) 方向平滑：相邻方向角差
    direction = torch.atan2(dlon_deg, dlat_deg + 1e-6)
    dir_diff = torch.diff(direction, dim=1)
    dir_penalty = torch.abs(torch.sin(dir_diff / 2.0))  # 小角度近似

    # 4) 强度约束：30kt/24h ≈ 0.625kt/0.5h，归一化后阈值 ~0.0063
    vmax_change = pred_deltas[..., 2] * 100.0  # 转回 kt
    vmax_penalty = F.relu(torch.abs(vmax_change) - 0.625)

    # 聚合
    loss = (
        speed_penalty.mean()
        + smooth_penalty.mean()
        + dir_penalty.mean()
        + vmax_penalty.mean()
    )
    return loss


def geo_distance_loss(
    pred_coords_norm: torch.Tensor,
    target_coords_norm: torch.Tensor,
    normalize_factor_km: float = 500.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    基于经纬度的地理距离损失

    Args:
        pred_coords_norm: (B, T, 3) 归一化坐标 [lat, lon, vmax]
        target_coords_norm: (B, T, 3) 归一化坐标 [lat, lon, vmax]
        normalize_factor_km: 将 km 尺度缩放后参与总损失，默认 /100 降权

    Returns:
        (loss, mae_km, rmse_km)
    """
    lat_range = data_cfg.lat_range
    lon_range = data_cfg.lon_range

    pred_lat = pred_coords_norm[..., 0] * (lat_range[1] - lat_range[0]) + lat_range[0]
    pred_lon = pred_coords_norm[..., 1] * (lon_range[1] - lon_range[0]) + lon_range[0]
    tgt_lat = target_coords_norm[..., 0] * (lat_range[1] - lat_range[0]) + lat_range[0]
    tgt_lon = target_coords_norm[..., 1] * (lon_range[1] - lon_range[0]) + lon_range[0]

    lat_err_km = (pred_lat - tgt_lat) * 111.0
    lon_err_km = (pred_lon - tgt_lon) * 111.0 * torch.cos(torch.deg2rad(tgt_lat))
    dist_km = torch.sqrt(lat_err_km**2 + lon_err_km**2)

    mae_km = dist_km.mean()
    rmse_km = torch.sqrt((dist_km**2).mean())
    loss = mae_km / normalize_factor_km
    return loss, mae_km, rmse_km
