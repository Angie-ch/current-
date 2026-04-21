"""
评价指标工具：MAE / RMSE（km）与强度误差
"""
from typing import Dict, Tuple
import torch
import math


def _mae_rmse(pred: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    mae = torch.mean(torch.abs(pred - target), dim=0)
    rmse = torch.sqrt(torch.mean((pred - target) ** 2, dim=0))
    return mae, rmse


def geo_errors_km(
    pred_lat: torch.Tensor,
    pred_lon: torch.Tensor,
    target_lat: torch.Tensor,
    target_lon: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    基于球面近似计算纬向/经向误差和总距离误差（km）
    输入单位均为度，形状 (B, T)
    """
    lat_err_km = (pred_lat - target_lat) * 111.0
    lon_err_km = (pred_lon - target_lon) * 111.0 * torch.cos(torch.deg2rad(target_lat))
    dist_km = torch.sqrt(lat_err_km**2 + lon_err_km**2)
    return lat_err_km, lon_err_km, dist_km


def evaluate_coords(
    pred_lat: torch.Tensor,
    pred_lon: torch.Tensor,
    pred_vmax: torch.Tensor,
    target_lat: torch.Tensor,
    target_lon: torch.Tensor,
    target_vmax: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """
    对绝对坐标序列进行评估
    形状： (B, T)
    """
    lat_err_km, lon_err_km, dist_km = geo_errors_km(pred_lat, pred_lon, target_lat, target_lon)

    # 按步统计
    mae_km, rmse_km = _mae_rmse(dist_km, torch.zeros_like(dist_km))
    vmax_mae, vmax_rmse = _mae_rmse(pred_vmax, target_vmax)

    return {
        "mae_km": mae_km,             # (T,)
        "rmse_km": rmse_km,           # (T,)
        "mean_dist_km": dist_km.mean(dim=0),
        "vmax_mae": vmax_mae,         # (T,)
        "vmax_rmse": vmax_rmse,       # (T,)
        "overall_dist_mean": dist_km.mean(),
        "overall_dist_rmse": torch.sqrt(torch.mean(dist_km**2)),
    }
