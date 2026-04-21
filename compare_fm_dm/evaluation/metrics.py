"""
对比评估模块 — Flow Matching vs Diffusion 论文核心实验

评估指标:
1. 确定性精度 (Deterministic Accuracy)
   - 纬度加权RMSE
   - 异常相关系数 (ACC)
   - 分通道逐时效RMSE

2. 频谱特性 (Spectral Analysis) — 论文核心卖点
   - 功率谱密度 (PSD) — 2D动能谱
   - 预期: Diffusion高频能量衰减(过平滑)，FM保持更好的谱斜率

3. 物理一致性 (Physical Consistency)
   - 散度违背分析
   - 地转平衡分析

4. 推理效率 (Efficiency)
   - NFE (Number of Function Evaluations) vs 精度曲线
   - FM宣称: 4步 ≈ DM 50步精度

5. 时间相干性 (Temporal Coherence)
   - 相邻帧之间的统计相关性
"""
import os
import json
import logging
from typing import Dict, List, Optional, Tuple, Callable
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

logger = logging.getLogger(__name__)


# ============================================================
# 基础评估指标
# ============================================================

def compute_rmse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """逐通道RMSE: (C,)"""
    mse = ((pred - target) ** 2).mean(dim=(0, 2, 3))
    return torch.sqrt(mse)


def compute_mae(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """逐通道MAE: (C,)"""
    return (pred - target).abs().mean(dim=(0, 2, 3))


def compute_lat_weighted_rmse(
    pred: torch.Tensor,
    target: torch.Tensor,
    grid_size: int = 40,
    lat_range: Tuple[float, float] = (0.0, 60.0),
) -> torch.Tensor:
    """
    纬度加权RMSE

    极地网格点面积小，误差权重应较小
    赤道网格点面积大，误差权重应较大
    """
    # 纬度权重: cos(lat)
    lats = np.linspace(lat_range[0], lat_range[1], grid_size)
    lat_weights = np.cos(np.radians(lats))
    lat_weights = lat_weights / lat_weights.mean()  # 归一化

    # 扩展到 (1, 1, H, W)
    weights = torch.from_numpy(lat_weights).float().view(1, 1, -1, 1).to(pred.device)

    # 加权MSE
    diff_sq = (pred - target) ** 2
    weighted_mse = (diff_sq * weights).mean(dim=(0, 2, 3))
    return torch.sqrt(weighted_mse)


def compute_acc(
    pred: torch.Tensor,
    target: torch.Tensor,
    clim_mean: torch.Tensor,
) -> torch.Tensor:
    """
    异常相关系数 ACC（逐通道）

    ACC = Σ(pred-clim)*(target-clim) / √(Σ(pred-clim)² * Σ(target-clim)²)
    """
    pred_anom = pred - clim_mean
    target_anom = target - clim_mean

    numerator = (pred_anom * target_anom).sum(dim=(0, 2, 3))
    denominator = torch.sqrt(
        (pred_anom ** 2).sum(dim=(0, 2, 3)) * (target_anom ** 2).sum(dim=(0, 2, 3))
    )
    return numerator / (denominator + 1e-8)


def compute_channel_bias(
    pred: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """逐通道系统性偏差 (mean(pred - target))"""
    return (pred - target).mean(dim=(0, 2, 3))


# ============================================================
# 功率谱密度 (PSD) — 论文核心加分项
# ============================================================

def compute_2d_psd(
    field: torch.Tensor,
    resolution: float = 0.25,
    device: torch.device = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    计算2D场的径向平均功率谱密度 (Radial Mean Power Spectral Density)

    气象学中常用2D动能谱来衡量模型在不同尺度上的表现:
    - 大尺度 (低波数): 大气环流主导
    - 小尺度 (高波数): 对流、涡旋等精细结构

    Diffusion模型的"过平滑"问题会导致高频部分能量衰减

    Args:
        field: (B, C, H, W) 气象场
        resolution: 网格分辨率 (度)
        device: 计算设备

    Returns:
        k: 波数数组 (x轴)
        psd: 径向平均功率谱密度 (y轴)
    """
    if device is None:
        device = field.device

    field = field.to(device)
    B, C, H, W = field.shape

    # 对batch和channel取平均
    field_mean = field.mean(dim=(0, 1))  # (H, W)

    # 去均值 (只保留波动)
    field_mean = field_mean - field_mean.mean()

    # 2D FFT
    fft = torch.fft.fft2(field_mean)
    fft_shifted = torch.fft.fftshift(fft)
    power = torch.abs(fft_shifted) ** 2  # (H, W)

    # 转换为numpy
    power_np = power.cpu().numpy()

    # 创建波数网格 (FFT频率)
    h, w = power_np.shape
    y_freq = np.fft.fftfreq(h, d=resolution)
    x_freq = np.fft.fftfreq(w, d=resolution)
    X, Y = np.meshgrid(x_freq, y_freq)

    # 到实际波数的转换 (假设等距网格)
    # k = √(kx² + ky²)，单位: cycles/degree
    k_grid = np.sqrt(X**2 + Y**2)

    # 定义径向bins (波数从低到高)
    k_min = k_grid.min()
    k_max = k_grid.max()
    n_bins = min(H, W) // 2

    k_bins = np.linspace(k_min, k_max, n_bins + 1)
    k_centers = 0.5 * (k_bins[:-1] + k_bins[1:])

    # 径向平均
    psd = np.zeros(n_bins)
    counts = np.zeros(n_bins)

    for i in range(h):
        for j in range(w):
            bin_idx = int((k_grid[i, j] - k_min) / (k_max - k_min) * n_bins)
            bin_idx = min(max(bin_idx, 0), n_bins - 1)
            psd[bin_idx] += power_np[i, j]
            counts[bin_idx] += 1

    # 避免除零
    counts = np.maximum(counts, 1)
    psd = psd / counts

    # 归一化 (方便比较)
    psd = psd / psd.sum()

    return k_centers, psd


def compute_kinetic_energy_spectrum(
    fields: List[torch.Tensor],
    u_channel: int,
    v_channel: int,
    resolution: float = 0.25,
    device: torch.device = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    计算2D动能谱 E(k)

    E(k) = 0.5 * (u^2 + v^2) 在波数k处的功率

    这是气象学中衡量模型小尺度涡旋保留能力的标准方法

    Args:
        fields: 预测/真值场列表
        u_channel, v_channel: u/v通道索引
        resolution: 网格分辨率

    Returns:
        k: 波数
        psd_fm: FM的动能谱
        psd_dm: DM的动能谱 (如果有)
        psd_gt: 真值的动能谱
    """
    if device is None:
        device = fields[0].device

    def single_psd(field):
        u = field[u_channel].cpu().numpy()
        v = field[v_channel].cpu().numpy()
        ke = 0.5 * (u**2 + v**2)
        return ke

    ke_fields = [single_psd(f) for f in fields]
    ke_mean = np.mean(ke_fields, axis=0)

    H, W = ke_mean.shape
    h, w = H, W

    # 2D FFT
    fft = np.fft.fft2(ke_mean)
    fft_shifted = np.fft.fftshift(fft)
    power = np.abs(fft_shifted) ** 2

    # 波数网格
    y_freq = np.fft.fftfreq(H, d=resolution)
    x_freq = np.fft.fftfreq(W, d=resolution)
    X, Y = np.meshgrid(x_freq, y_freq)
    k_grid = np.sqrt(X**2 + Y**2)

    k_min = k_grid.min()
    k_max = k_grid.max()
    n_bins = H // 2

    k_bins = np.linspace(k_min, k_max, n_bins + 1)
    k_centers = 0.5 * (k_bins[:-1] + k_bins[1:])

    psd = np.zeros(n_bins)
    counts = np.zeros(n_bins)

    for i in range(H):
        for j in range(W):
            bin_idx = int((k_grid[i, j] - k_min) / (k_max - k_min) * n_bins)
            bin_idx = min(max(bin_idx, 0), n_bins - 1)
            psd[bin_idx] += power[i, j]
            counts[bin_idx] += 1

    counts = np.maximum(counts, 1)
    psd = psd / counts

    return k_centers, psd, None


def compute_spectral_slope(
    k: np.ndarray,
    psd: np.ndarray,
    k_range: Tuple[float, float] = (0.05, 0.3),
) -> float:
    """
    计算功率谱在对数-对数空间中的斜率

    理论斜率:
    - 2D湍流: -5/3 (Kolmogorov)
    - 大气运动: -3 (charney)

    如果FM的斜率更接近理论值，说明保留了更多小尺度湍流

    Returns:
        斜率 (在log-log空间中)
    """
    mask = (k >= k_range[0]) & (k <= k_range[1])
    k_sel = k[mask]
    psd_sel = psd[mask]

    if len(k_sel) < 3:
        return np.nan

    log_k = np.log10(k_sel + 1e-10)
    log_psd = np.log10(psd_sel + 1e-10)

    # 线性回归
    coeffs = np.polyfit(log_k, log_psd, 1)
    return coeffs[0]


# ============================================================
# 物理一致性评估
# ============================================================

def compute_divergence(
    u: torch.Tensor,
    v: torch.Tensor,
    resolution: float = 0.25,
) -> torch.Tensor:
    """
    计算风场散度 ∇·u = ∂u/∂x + ∂v/∂y

    物理意义:
    - 不可压大气应该有接近零的散度
    - 模型预测的散度越小，物理一致性越好

    Args:
        u: 东西风分量, shape (B, H, W) 或 (B, 1, H, W)
        v: 南北风分量, shape (B, H, W) 或 (B, 1, H, W)
    """
    # 确保输入为4D: (B, 1, H, W)
    if u.ndim == 3:
        u = u.unsqueeze(1)
    if v.ndim == 3:
        v = v.unsqueeze(1)

    du_dx = u[:, :, :, 1:] - u[:, :, :, :-1]
    dv_dy = v[:, :, 1:, :] - v[:, :, :-1, :]
    du_dx = du_dx[:, :, :-1, :]
    dv_dy = dv_dy[:, :, :, :-1]

    div = du_dx + dv_dy
    return div


def compute_divergence_rmse(
    pred: torch.Tensor,
    target: torch.Tensor,
    u_channel: int = 0,
    v_channel: int = 3,
    resolution: float = 0.25,
) -> Tuple[float, float]:
    """
    计算散度RMSE

    散度RMSE衡量模型预测的风场是否满足质量守恒

    Returns:
        (pred_div_rmse, target_div_rmse)
    """
    pred_u = pred[:, u_channel]
    pred_v = pred[:, v_channel]
    target_u = target[:, u_channel]
    target_v = target[:, v_channel]

    pred_div = compute_divergence(pred_u, pred_v, resolution)
    target_div = compute_divergence(target_u, target_v, resolution)

    pred_div_rmse = torch.sqrt((pred_div ** 2).mean()).item()
    target_div_rmse = torch.sqrt((target_div ** 2).mean()).item()

    return pred_div_rmse, target_div_rmse


def compute_geostrophic_balance(
    u: torch.Tensor,
    v: torch.Tensor,
    z: torch.Tensor,
    f_corio: Optional[torch.Tensor] = None,
    resolution: float = 0.25,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    计算地转平衡残差

    地转风近似:
        u_geo = -g/f * ∂z/∂y
        v_geo = g/f * ∂z/∂x

    地转平衡残差:
        R = √((u - u_geo)² + (v - v_geo)²)

    Returns:
        (balance_residual_u, balance_residual_v) — 残差的均值
    """
    import math
    R, Omega = 6.371e6, 7.2921e-5
    g = 9.80665

    # 科里奥利参数 f = 2Ωsin(lat)
    if f_corio is None:
        lat_size = z.shape[-2]  # H dimension = latitude
        lon_size = z.shape[-1]   # W dimension = longitude
        lats = torch.linspace(0, 60, lat_size, device=z.device)
        f = 2 * Omega * torch.sin(torch.deg2rad(lats))  # (H,)
        f = f.view(1, 1, -1, 1).expand(B, 1, lat_size, lon_size)  # (B, 1, H, W)
    else:
        f = f_corio

    # 计算位势梯度
    dzdx = (z[:, :, :, 2:] - z[:, :, :, :-2]) / (2 * resolution)
    dzdy = (z[:, :, 2:, :] - z[:, :, :-2, :]) / (2 * resolution)

    # 裁剪到内部区域
    f_x = f[:, :, 1:-1, 1:-1]
    f_y = f[:, :, 1:-1, 1:-1]

    # 地转风
    v_geo = g * dzdx / (f_x + 1e-8)
    u_geo = -g * dzdy / (f_y + 1e-8)

    # 残差
    u_crop = u[:, :, 1:-1, 1:-1]
    v_crop = v[:, :, 1:-1, 1:-1]

    residual_u = (u_crop - u_geo).abs().mean()
    residual_v = (v_crop - v_geo).abs().mean()

    return residual_u, residual_v


def compute_vorticity(
    u: torch.Tensor,
    v: torch.Tensor,
    resolution: float = 0.25,
) -> torch.Tensor:
    """
    计算相对涡度 ζ = ∂v/∂x - ∂u/∂y
    """
    dv_dx = v[:, :, :, 1:] - v[:, :, :, :-1]
    du_dy = u[:, :, 1:, :] - u[:, :, :-1, :]
    dv_dx = dv_dx[:, :, :-1, :]
    du_dy = du_dy[:, :, :, :-1]

    zeta = dv_dx - du_dy
    return zeta


# ============================================================
# Per-channel spatial correlation (not ACC, but actual spatial pattern match)
# ============================================================

def compute_spatial_correlation(
    pred: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """
    Per-channel spatial correlation.

    ACC = correlation(pred.flatten(), target.flatten()) across all samples.
    This = spatial correlation per-channel, per-sample then averaged.

    Returns: (C,) tensor of spatial correlations.
    """
    B, C, H, W = pred.shape
    corrs = []
    for ch in range(C):
        p = pred[:, ch].reshape(B, -1)
        t = target[:, ch].reshape(B, -1)
        p_mean = p.mean(dim=1, keepdim=True)
        t_mean = t.mean(dim=1, keepdim=True)
        p_std = p.std(dim=1, keepdim=True).clamp(min=1e-8)
        t_std = t.std(dim=1, keepdim=True).clamp(min=1e-8)
        cov = ((p - p_mean) * (t - t_mean)).mean(dim=1)
        corr = cov / (p_std.squeeze(-1) * t_std.squeeze(-1) + 1e-8)
        corrs.append(corr.mean())
    return torch.tensor(corrs)


# ============================================================
# Z-channel diagnostics (variance collapse detection)
# ============================================================

def compute_z_channel_diagnostics(
    pred: torch.Tensor,
    target: torch.Tensor,
    z_indices: List[int],
    z_names: List[str],
) -> Dict[str, float]:
    """
    Comprehensive Z-channel diagnostics for variance collapse detection.

    For each z channel:
    - std_ratio: pred_std / gt_std (collapse = low ratio)
    - spatial_corr: pattern correlation
    - rmse: absolute error in physical units
    - bias: systematic over/under prediction
    - cv_ratio: CV(pred) / CV(gt) where CV = std/mean
    """
    results = {}
    for z_idx, z_name in zip(z_indices, z_names):
        p_ch = pred[:, z_idx]
        t_ch = target[:, z_idx]

        p_std = p_ch.std().item()
        t_std = t_ch.std().item()
        std_ratio = p_std / (t_std + 1e-8)

        p_mean = p_ch.mean().item()
        t_mean = t_ch.mean().item()
        p_cv = p_std / (abs(p_mean) + 1e-8)
        t_cv = t_std / (abs(t_mean) + 1e-8)
        cv_ratio = p_cv / (t_cv + 1e-8)

        bias = (p_ch - t_ch).mean().item()
        rmse = torch.sqrt(((p_ch - t_ch) ** 2).mean()).item()

        p_flat = p_ch.reshape(-1)
        t_flat = t_ch.reshape(-1)
        spatial_corr = torch.corrcoef(torch.stack([p_flat, t_flat]))[0, 1].item()

        results[f"{z_name}_std_ratio"] = std_ratio
        results[f"{z_name}_spatial_corr"] = spatial_corr
        results[f"{z_name}_rmse"] = rmse
        results[f"{z_name}_bias"] = bias
        results[f"{z_name}_cv_ratio"] = cv_ratio
        results[f"{z_name}_pred_std"] = p_std
        results[f"{z_name}_gt_std"] = t_std

    return results


# ============================================================
# 时间相干性评估
# ============================================================

def compute_temporal_coherence(
    predictions: List[torch.Tensor],
    ground_truth: List[torch.Tensor],
) -> Dict[str, float]:
    """
    计算时间相干性指标

    衡量相邻预测帧之间的变化是否平滑合理
    """
    if len(predictions) < 2:
        return {}

    pred_diffs = []
    gt_diffs = []

    for t in range(len(predictions) - 1):
        pred_diff = (predictions[t + 1] - predictions[t]).abs().mean().item()
        gt_diff = (ground_truth[t + 1] - ground_truth[t]).abs().mean().item()
        pred_diffs.append(pred_diff)
        gt_diffs.append(gt_diff)

    return {
        "mean_pred_diff": np.mean(pred_diffs),
        "mean_gt_diff": np.mean(gt_diffs),
        "diff_ratio": np.mean(pred_diffs) / (np.mean(gt_diffs) + 1e-8),
    }


# ============================================================
# NFE效率曲线
# ============================================================

def compute_nfe_efficiency(
    model,
    conditions: List[torch.Tensor],
    targets: List[torch.Tensor],
    nfe_steps_list: List[int],
    method: str,
    device: torch.device,
            clamp_range: Optional[Tuple[float, float]] = None,  # was (-5.0, 5.0)
            z_clamp_range: Optional[Tuple[float, float]] = None,  # was (-1.0, 1.0)
    z_channel_indices: List[int] = None,
) -> Dict[int, Dict[str, float]]:
    """
    计算不同NFE步数下的精度

    核心对比实验:
    - FM: 测试 {1, 2, 4, 8} 步
    - DM: 测试 {5, 10, 20, 50} 步

    Returns:
        {nfe_steps: {"rmse": ..., "mae": ...}, ...}
    """
    results = {}

    for nfe in nfe_steps_list:
        rmses = []
        maes = []

        for cond, tgt in zip(conditions, targets):
            cond = cond.unsqueeze(0).to(device)
            tgt = tgt.unsqueeze(0).to(device)

            with torch.no_grad():
                if method == "fm":
                    pred = model.sample_fm(
                        cond, device,
                        euler_steps=nfe,
                        euler_mode="midpoint",
                        clamp_range=clamp_range,
                        z_clamp_range=z_clamp_range,
                    )
                else:
                    pred = model.sample_dm(
                        cond, device,
                        ddim_steps=nfe,
                        clamp_range=clamp_range,
                        z_clamp_range=z_clamp_range,
                    )

                pred = pred[:, :tgt.shape[1]]  # 裁剪到相同通道数

                # 反归一化
                rmse = compute_rmse(pred, tgt)
                mae = compute_mae(pred, tgt)

                rmses.append(rmse.mean().item())
                maes.append(mae.mean().item())

        results[nfe] = {
            "rmse_mean": np.mean(rmses),
            "rmse_std": np.std(rmses),
            "mae_mean": np.mean(maes),
            "mae_std": np.std(maes),
        }
        logger.info(f"NFE={nfe}: RMSE={results[nfe]['rmse_mean']:.4f} ± {results[nfe]['rmse_std']:.4f}")

    return results


# ============================================================
# 综合评估器
# ============================================================

class ComparisonEvaluator:
    """
    FM vs DM 综合对比评估器
    """

    def __init__(
        self,
        data_cfg,
        device: torch.device = None,
        norm_mean: np.ndarray = None,
        norm_std: np.ndarray = None,
        clim_path: str = None,
    ):
        self.data_cfg = data_cfg
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.norm_mean = norm_mean
        self.norm_std = norm_std
        self.var_names = data_cfg.var_names
        self.z_channel_indices = data_cfg.z_channel_indices
        self.grid_size = data_cfg.grid_size

        # 气候态均值 (用于ACC计算)
        if clim_path and os.path.exists(clim_path):
            self.clim_mean = torch.from_numpy(np.load(clim_path)).float().to(self.device)
        else:
            self.clim_mean = None

    def denormalize(self, data: torch.Tensor) -> torch.Tensor:
        """反归一化"""
        if self.norm_mean is None or self.norm_std is None:
            return data

        mean_t = torch.from_numpy(self.norm_mean).float().to(data.device)
        std_t = torch.from_numpy(self.norm_std).float().to(data.device)
        std_t = torch.where(std_t < 1e-8, torch.ones_like(std_t), std_t)

        if data.ndim == 4:
            mean_t = mean_t.reshape(1, -1, 1, 1)
            std_t = std_t.reshape(1, -1, 1, 1)
        return data * std_t + mean_t

    def evaluate_single(
        self,
        predictions: List[torch.Tensor],
        ground_truth: List[torch.Tensor],
        method_name: str = "model",
    ) -> Dict:
        """
        对单个模型进行全面评估

        Args:
            predictions: 预测列表，每个元素 (B, C, H, W)
            ground_truth: 真值列表
            method_name: 方法名 ("FM" 或 "DM")

        Returns:
            评估结果字典
        """
        results = {
            "method": method_name,
            "n_samples": len(predictions),
        }

        # 合并batch维度
        all_preds = torch.cat([p.cpu() for p in predictions], dim=0)
        all_gts = torch.cat([t.cpu() for t in ground_truth], dim=0)

        # 反归一化到物理单位
        all_preds_phys = self.denormalize(all_preds)
        all_gts_phys = self.denormalize(all_gts)

        n_vars = len(self.var_names)

        # 1. 逐通道RMSE
        rmse = compute_rmse(all_preds_phys, all_gts_phys).numpy()
        results["rmse_per_channel"] = {self.var_names[i]: float(rmse[i]) for i in range(n_vars)}
        results["rmse_mean"] = float(rmse.mean())

        # 2. 纬度加权RMSE
        lat_rmse = compute_lat_weighted_rmse(
            all_preds_phys, all_gts_phys, self.grid_size
        ).numpy()
        results["lat_weighted_rmse_per_channel"] = {
            self.var_names[i]: float(lat_rmse[i]) for i in range(n_vars)
        }
        results["lat_weighted_rmse_mean"] = float(lat_rmse.mean())

        # 3. 逐通道MAE
        mae = compute_mae(all_preds_phys, all_gts_phys).numpy()
        results["mae_per_channel"] = {self.var_names[i]: float(mae[i]) for i in range(n_vars)}
        results["mae_mean"] = float(mae.mean())

        # 4. 系统性偏差
        bias = compute_channel_bias(all_preds_phys, all_gts_phys).numpy()
        results["bias_per_channel"] = {self.var_names[i]: float(bias[i]) for i in range(n_vars)}

        # 5. 散度RMSE (针对u/v通道)
        if len(self.var_names) >= 6:
            pred_div, gt_div = compute_divergence_rmse(
                all_preds_phys, all_gts_phys,
                u_channel=0, v_channel=3,
            )
            results["divergence_rmse_pred"] = float(pred_div)
            results["divergence_rmse_gt"] = float(gt_div)

        # 6. Z通道专项统计
        for z_idx, z_name in enumerate(["z_850", "z_500", "z_250"]):
            if z_idx < len(self.z_channel_indices):
                ch = self.z_channel_indices[z_idx]
                pred_z = all_preds_phys[:, ch]
                gt_z = all_gts_phys[:, ch]
                results[f"{z_name}_pred_mean"] = float(pred_z.mean().item())
                results[f"{z_name}_pred_std"] = float(pred_z.std().item())
                results[f"{z_name}_gt_mean"] = float(gt_z.mean().item())
                results[f"{z_name}_gt_std"] = float(gt_z.std().item())

        # 7. 打印表格
        self._print_rmse_table(results)

        return results

    def _print_rmse_table(self, results: Dict):
        """打印RMSE表格"""
        logger.info("=" * 70)
        logger.info(f"评估结果 ({results['method']}, n={results['n_samples']})")
        logger.info("=" * 70)

        header = f"{'Variable':>12}"
        header += f"  {'RMSE':>10}"
        header += f"  {'Lat-W.RMSE':>12}"
        header += f"  {'Bias':>10}"
        logger.info(header)
        logger.info("-" * 70)

        for i, var in enumerate(self.var_names):
            row = f"{var:>12}"
            row += f"  {results['rmse_per_channel'].get(var, 0):>10.4f}"
            lw_rmse = results.get('lat_weighted_rmse_per_channel', {}).get(var, 0)
            row += f"  {lw_rmse:>12.4f}"
            bias = results.get('bias_per_channel', {}).get(var, 0)
            row += f"  {bias:>10.4f}"
            logger.info(row)

        logger.info("-" * 70)
        avg_row = f"{'MEAN':>12}"
        avg_row += f"  {results['rmse_mean']:>10.4f}"
        avg_row += f"  {results.get('lat_weighted_rmse_mean', 0):>12.4f}"
        logger.info(avg_row)
        logger.info("")

    def evaluate_spectral(
        self,
        predictions: List[torch.Tensor],
        ground_truth: List[torch.Tensor],
        method_name: str = "model",
    ) -> Dict:
        """
        频谱分析评估 (论文核心)

        计算:
        - 2D动能谱
        - 谱斜率
        - 高频能量比
        """
        all_preds = torch.cat([p.cpu() for p in predictions], dim=0)
        all_gts = torch.cat([t.cpu() for t in ground_truth], dim=0)

        # 找u/v通道索引
        var_names = self.var_names
        u_idx = var_names.index("u_850") if "u_850" in var_names else 0
        v_idx = var_names.index("v_850") if "v_850" in var_names else 3

        results = {"method": method_name}

        # 对预测和真值场分组计算
        # 动能谱
        k, psd_pred, _ = compute_kinetic_energy_spectrum(
            [all_preds], u_idx, v_idx, resolution=0.25
        )
        _, psd_gt, _ = compute_kinetic_energy_spectrum(
            [all_gts], u_idx, v_idx, resolution=0.25
        )

        results["k"] = k.tolist()
        results["psd_pred"] = psd_pred.tolist()
        results["psd_gt"] = psd_gt.tolist()

        # 计算谱斜率
        slope_pred = compute_spectral_slope(k, psd_pred, k_range=(0.05, 0.3))
        slope_gt = compute_spectral_slope(k, psd_gt, k_range=(0.05, 0.3))

        results["spectral_slope_pred"] = float(slope_pred)
        results["spectral_slope_gt"] = float(slope_gt)

        # 高频能量比 (k > 0.2 的能量 / 总能量)
        high_freq_mask = k > 0.2
        if high_freq_mask.any():
            hf_ratio_pred = psd_pred[high_freq_mask].sum() / (psd_pred.sum() + 1e-10)
            hf_ratio_gt = psd_gt[high_freq_mask].sum() / (psd_gt.sum() + 1e-10)
            results["high_freq_energy_ratio_pred"] = float(hf_ratio_pred)
            results["high_freq_energy_ratio_gt"] = float(hf_ratio_gt)

        logger.info(f"谱分析 ({method_name}): slope={slope_pred:.3f} (真值={slope_gt:.3f})")

        return results

    def save_results(self, results: Dict, output_path: str):
        """保存评估结果到JSON"""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        logger.info(f"评估结果已保存: {output_path}")
