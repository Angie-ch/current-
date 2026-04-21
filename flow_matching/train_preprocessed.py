"""
Flow Matching 训练脚本 — 适配 preprocessed_9ch_40x40 数据
三阶段训练策略实现

阶段1: 预热期 (Epoch 1-100) - 只训练loss_mse (纯MSE warmup)
阶段2: 物理炼金期 (Epoch 101-160) - 线性引入物理约束
阶段3: 微调收敛期 (Epoch 161-200) - 学习率下调，稳定收敛

数据来源: /root/autodl-tmp/fyp_final/preprocessed_9ch_40x40
- 格式: (T, 9, 40, 40) 每个台风一个 .npy 文件
- 通道: u_850, u_500, u_250, v_850, v_500, v_250, z_850, z_500, z_250

数据集划分 (ICLR 2024 论文标准):
- 训练集: 1950-2015年
- 验证集: 2016-2018年
- 测试集: 2019-2021年
"""
import os
import sys
import copy
import time
import math
import logging
import argparse
from pathlib import Path
from typing import Optional, Tuple, Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from configs.config import DataConfig, ModelConfig, TrainConfig, InferenceConfig, get_config

FLOW_MATCHING_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(FLOW_MATCHING_DIR)
sys.path.insert(0, PARENT_DIR)

from models.flow_matching_model import ERA5FlowMatchingModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ============================================================
# 数据集: 使用 preprocessed_9ch_40x40 数据
# ============================================================

def _load_norm_stats(norm_stats_path: str, expected_channels: int):
    """加载归一化统计量"""
    import torch as T
    if os.path.exists(norm_stats_path):
        stats = T.load(norm_stats_path, weights_only=True, map_location='cpu')
        mean = stats['mean'].numpy()
        std = stats['std'].numpy()
        logger.info(f"加载归一化统计: {norm_stats_path} ({len(mean)}ch)")
        if len(mean) > expected_channels:
            mean = mean[:expected_channels]
            std = std[:expected_channels]
        return mean, std
    else:
        logger.warning(f"未找到 {norm_stats_path}，使用硬编码统计量")
        mean = np.array([
            -1.29, -0.28, 0.39,
             1.74,  2.38,  2.60,
         14253.13, 56708.52, 106498.13,
        ], dtype=np.float32)
        std = np.array([
            10.02, 10.24, 12.65,
             9.04,  8.48,  8.56,
          1320.93, 4869.39, 9103.74,
        ], dtype=np.float32)
        return mean, std


def normalize_era5(era5: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """逐通道 z-score 标准化 ERA5"""
    if era5.ndim == 4:  # (T, C, H, W)
        C = era5.shape[1]
        m = mean[:C].reshape(1, C, 1, 1)
        s = std[:C].reshape(1, C, 1, 1)
    elif era5.ndim == 3:  # (C, H, W)
        C = era5.shape[0]
        m = mean[:C].reshape(C, 1, 1)
        s = std[:C].reshape(C, 1, 1)
    else:
        return era5
    result = (era5 - m) / (s + 1e-8)
    if not result.flags['C_CONTIGUOUS']:
        result = np.ascontiguousarray(result)
    return result


class PreprocessedCFMData:
    """preprocessed_9ch_40x40 数据加载器 (适配 Flow Matching)"""

    def __init__(
        self,
        era5_dir: str,
        csv_path: str,
        era5_channels: int = 9,
        grid_size: int = 40,
        t_history: int = 16,
        t_future: int = 24,
        norm_mean: np.ndarray = None,
        norm_std: np.ndarray = None,
    ):
        self.era5_dir = era5_dir
        self.csv_path = csv_path
        self.era5_channels = era5_channels
        self.grid_size = grid_size
        self.t_history = t_history
        self.t_future = t_future
        self.total_length = t_history + t_future
        self.norm_mean = norm_mean
        self.norm_std = norm_std

        self.storm_ids = self._scan_storms()
        self.samples_index = self._build_index()

        logger.info(f"PreprocessedCFMData: t_history={t_history}, t_future={t_future}, "
                    f"samples={len(self.samples_index)}, storms={len(self.storm_ids)}")

    def _scan_storms(self) -> List[str]:
        """扫描可用台风"""
        import glob
        import pandas as pd

        npy_files = sorted(glob.glob(os.path.join(self.era5_dir, '*.npy')))
        # 过滤只保留 .npy 文件（排除 _times.npy 等辅助文件）
        storm_ids = []
        for f in npy_files:
            basename = os.path.basename(f)
            if '_times.npy' not in basename and '_track_mapping.npy' not in basename:
                storm_id = os.path.splitext(basename)[0]
                storm_ids.append(storm_id)

        logger.info(f"扫描到 {len(storm_ids)} 个台风数据")
        return storm_ids

    def _build_index(self) -> List[Tuple[str, int]]:
        """构建样本索引: (storm_id, start_idx)"""
        import pandas as pd

        index = []
        lat_range = (0.0, 60.0)
        lon_range = (95.0, 185.0)

        # 读取 CSV 获取年份信息
        if os.path.exists(self.csv_path):
            track_df = pd.read_csv(self.csv_path)
            if 'typhoon_id' in track_df.columns:
                track_df = track_df.rename(columns={'typhoon_id': 'storm_id'})
        else:
            logger.warning(f"CSV 文件不存在: {self.csv_path}")
            track_df = None

        for sid in tqdm(self.storm_ids, desc="构建样本索引"):
            era5_path = os.path.join(self.era5_dir, f"{sid}.npy")
            if not os.path.exists(era5_path):
                continue

            # 读取 ERA5 数据获取时间步数
            try:
                era5_fp = np.lib.format.open_memmap(era5_path, mode='r')
                era5_len = era5_fp.shape[0]
                del era5_fp
            except Exception as e:
                logger.warning(f"无法读取 {era5_path}: {e}")
                continue

            if era5_len < self.total_length:
                continue

            # 滑动窗口索引
            for start in range(0, era5_len - self.total_length + 1, 4):  # stride=4
                index.append((sid, start))

        return index

    def __len__(self):
        return len(self.samples_index)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sid, start = self.samples_index[idx]
        era5_path = os.path.join(self.era5_dir, f"{sid}.npy")

        # 加载 ERA5 数据
        raw_era5 = np.load(era5_path, mmap_mode='r')
        h_start = start
        h_end = start + self.t_history
        f_end = start + self.total_length

        raw_hist = raw_era5[h_start:h_end].astype(np.float32)
        raw_fut = raw_era5[h_end:f_end].astype(np.float32)

        # 归一化
        hist_era5 = normalize_era5(raw_hist, self.norm_mean, self.norm_std)
        fut_era5 = normalize_era5(raw_fut, self.norm_mean, self.norm_std)

        # P1: 返回完整历史用于时空聚合，不再只取最后一帧
        # Flow Matching: condition = 完整历史(16,9,40,40), target = 第一帧未来(9,40,40)
        # 后续在模型中用 Conv3D 聚合历史动量信息
        condition = hist_era5  # (T_history, C, H, W) = (16, 9, 40, 40)
        target = fut_era5[0]   # (C, H, W)

        return {
            'condition': torch.from_numpy(np.ascontiguousarray(condition)).float(),
            'target': torch.from_numpy(np.ascontiguousarray(target)).float(),
            'storm_id': sid,
        }


def create_preprocessed_dataloaders(
    data_cfg: DataConfig,
    train_cfg: TrainConfig,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """创建 preprocessed_9ch_40x40 数据加载器"""

    # 加载归一化统计
    norm_mean, norm_std = _load_norm_stats(
        data_cfg.norm_stats_path, data_cfg.era5_channels
    )

    # 按年份划分台风
    import pandas as pd
    import glob

    npy_files = sorted(glob.glob(os.path.join(data_cfg.era5_dir, '*.npy')))
    storm_ids = []
    for f in npy_files:
        basename = os.path.basename(f)
        if '_times.npy' not in basename and '_track_mapping.npy' not in basename:
            storm_ids.append(os.path.splitext(basename)[0])

    # 读取 CSV 获取年份
    if os.path.exists(data_cfg.csv_path):
        track_df = pd.read_csv(data_cfg.csv_path)
        if 'typhoon_id' in track_df.columns:
            track_df = track_df.rename(columns={'typhoon_id': 'storm_id'})
        storm_years = track_df.groupby('storm_id')['year'].first()
    else:
        logger.warning(f"CSV 不存在: {data_cfg.csv_path}")
        storm_years = {}

    train_ids, val_ids, test_ids = [], [], []
    for sid in storm_ids:
        year = storm_years.get(sid, None)
        if year is None:
            continue
        if year in data_cfg.train_years:
            train_ids.append(sid)
        elif year in data_cfg.val_years:
            val_ids.append(sid)
        elif year in data_cfg.test_years:
            test_ids.append(sid)

    logger.info(f"按年份划分:")
    logger.info(f"  训练集 ({min(data_cfg.train_years)}-{max(data_cfg.train_years)}): {len(train_ids)} 个台风")
    logger.info(f"  验证集 ({min(data_cfg.val_years)}-{max(data_cfg.val_years)}): {len(val_ids)} 个台风")
    logger.info(f"  测试集 ({min(data_cfg.test_years)}-{max(data_cfg.test_years)}): {len(test_ids)} 个台风")

    # 创建数据集
    common_kwargs = dict(
        era5_dir=data_cfg.era5_dir,
        csv_path=data_cfg.csv_path,
        era5_channels=data_cfg.era5_channels,
        grid_size=data_cfg.grid_size,
        t_history=data_cfg.history_steps,
        t_future=data_cfg.forecast_steps,
        norm_mean=norm_mean,
        norm_std=norm_std,
    )

    # 使用完整数据集构建索引
    full_ds = PreprocessedCFMData(**common_kwargs)

    # 按 ID 过滤
    train_ds = [full_ds.samples_index[i] for i in range(len(full_ds))
                if full_ds.samples_index[i][0] in train_ids]
    val_ds = [full_ds.samples_index[i] for i in range(len(full_ds))
              if full_ds.samples_index[i][0] in val_ids]
    test_ds = [full_ds.samples_index[i] for i in range(len(full_ds))
               if full_ds.samples_index[i][0] in test_ids]

    # 创建包装数据集
    class FilteredDataset:
        def __init__(self, samples, full_ds):
            self.samples = samples
            self.full_ds = full_ds

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, idx):
            sid, start = self.samples[idx]
            return self.full_ds[full_ds.samples_index.index((sid, start))]

    train_loader = DataLoader(
        FilteredDataset(train_ds, full_ds),
        batch_size=train_cfg.batch_size,
        shuffle=True,
        num_workers=data_cfg.num_workers,
        pin_memory=data_cfg.pin_memory,
        drop_last=True,
    )
    val_loader = DataLoader(
        FilteredDataset(val_ds, full_ds),
        batch_size=train_cfg.batch_size,
        shuffle=False,
        num_workers=data_cfg.num_workers,
        pin_memory=data_cfg.pin_memory,
    )
    test_loader = DataLoader(
        FilteredDataset(test_ds, full_ds),
        batch_size=train_cfg.batch_size,
        shuffle=False,
        num_workers=data_cfg.num_workers,
        pin_memory=data_cfg.pin_memory,
    )

    return train_loader, val_loader, test_loader, norm_mean, norm_std


# ============================================================
# 物理一致性损失 (地转平衡 + 散度约束 + 涡度平滑)
# ============================================================

class PhysicalConsistencyLoss(nn.Module):
    """
    物理一致性损失：地转平衡 + 散度约束 + 涡度平滑
    
    基于气象学原理的三重物理约束：
    1. 地转平衡: f·v = ∂Φ/∂x, f·u = -∂Φ/∂y
    2. 散度约束: ∇·V ≈ 0 (低纬度有效)
    3. 涡度平滑: ζ = ∂v/∂x - ∂u/∂y 空间平滑
    """
    
    def __init__(self, lat_grid, lon_res=0.25, lat_res=0.25):
        """
        Args:
            lat_grid: 纬度张量 (弧度), shape [H, W] 或 [H, 1]
            lon_res: 经度分辨率 (度)
            lat_res: 纬度分辨率 (度)
        """
        super().__init__()
        self.register_buffer('lat_grid', lat_grid)
        self.lon_res = lon_res
        self.lat_res = lat_res
        
    def forward(self, u_pred, v_pred, z_pred, return_components=False):
        """
        计算物理一致性损失
        
        Args:
            u_pred: u风分量, shape [B, H, W]
            v_pred: v风分量, shape [B, H, W]
            z_pred: 位势高度, shape [B, H, W]
            return_components: 是否返回各分量损失
        
        Returns:
            如果 return_components=False: 总物理损失
            如果 return_components=True: (总损失, 地转损失, 散度损失, 涡度损失)
        """
        R, Omega = 6.371e6, 7.2921e-5
        
        # 1. 科氏力参数 f = 2Ωsin(φ)
        f = 2 * Omega * torch.sin(self.lat_grid)
        # 防止赤道除零，加小量截断
        f = torch.where(f.abs() < 1e-5, torch.sign(f + 1e-10) * 1e-5, f)
        
        # 2. 物理距离 (考虑地球曲率)
        # Δx = R·cos(φ)·Δλ, Δy = R·Δφ
        dy = R * math.pi / 180 * self.lat_res
        dx = R * math.pi / 180 * self.lon_res * torch.cos(self.lat_grid)
        
        # 3. 中心差分计算梯度 (精度更高)
        # z_pred: [B, H, W], 裁剪边缘避免边界效应
        dzdx = (z_pred[:, :, 2:] - z_pred[:, :, :-2]) / (2 * dx[:, :, 1:-1] + 1e-8)
        dzdy = (z_pred[:, 2:, :] - z_pred[:, :-2, :]) / (2 * dy + 1e-8)
        
        # 4. 地转风计算
        # v_geo = (1/f) * ∂Φ/∂x, u_geo = -(1/f) * ∂Φ/∂y
        f_mid = f[1:-1, 1:-1]
        v_geo = dzdx[:, 1:-1, :] / (f_mid + 1e-8)
        u_geo = -dzdy[:, :, 1:-1] / (f_mid + 1e-8)
        
        # 对齐到预测维度
        u_pred_crop = u_pred[:, 1:-1, 1:-1]
        v_pred_crop = v_pred[:, 1:-1, 1:-1]
        
        # 5. 地转平衡损失 (核心)
        loss_geo = ((u_pred_crop - u_geo)**2 + (v_pred_crop - v_geo)**2).mean()
        
        # 6. 散度约束 (低纬度有效)
        loss_div = self._divergence_loss(u_pred_crop, v_pred_crop, dx, dy)
        
        # 7. 涡度平滑
        loss_vort = self._vorticity_smooth_loss(u_pred_crop, v_pred_crop, dx, dy)
        
        if return_components:
            return loss_geo + loss_div + loss_vort, loss_geo, loss_div, loss_vort
        return loss_geo + loss_div + loss_vort
    
    def _divergence_loss(self, u, v, dx, dy):
        """散度约束: ∇·V ≈ 0"""
        du_dx = (u[:, :, 2:] - u[:, :, :-2]) / (2 * dx[:, :, 1:-1] + 1e-8)
        dv_dy = (v[:, 2:, :] - v[:, :-2, :]) / (2 * dy + 1e-8)
        # 对齐维度
        du_dx = du_dx[:, :, 1:-1] if du_dx.shape[2] > v.shape[2] else du_dx
        dv_dy = dv_dy[:, 1:-1, :] if dv_dy.shape[1] > v.shape[1] else dv_dy
        div = du_dx + dv_dy
        return (div ** 2).mean()
    
    def _vorticity_smooth_loss(self, u, v, dx, dy):
        """涡度平滑: ζ = ∂v/∂x - ∂u/∂y"""
        dv_dx = (v[:, :, 2:] - v[:, :, :-2]) / (2 * dx[:, :, 1:-1] + 1e-8)
        du_dy = (u[:, 2:, :] - u[:, :-2, :]) / (2 * dy + 1e-8)
        # 对齐维度
        dv_dx = dv_dx[:, :, 1:-1] if dv_dx.shape[2] > u.shape[2] else dv_dx
        du_dy = du_dy[:, 1:-1, :] if du_dy.shape[1] > u.shape[1] else du_dy
        zeta = dv_dx - du_dy
        # 拉普拉斯正则：涡度应该平滑变化
        return (zeta ** 2).mean()


def compute_physics_error(u_pred, v_pred, z_pred, lat_grid, lon_res=0.25, lat_res=0.25):
    """
    计算物理一致性误差（用于评估，非训练）
    返回: 地转平衡误差 (m/s)²
    """
    R, Omega = 6.371e6, 7.2921e-5
    
    with torch.no_grad():
        f = 2 * Omega * torch.sin(lat_grid)
        f = torch.where(f.abs() < 1e-5, torch.sign(f + 1e-10) * 1e-5, f)
        
        dy = R * math.pi / 180 * lat_res
        dx = R * math.pi / 180 * lon_res * torch.cos(lat_grid)
        
        dzdx = (z_pred[:, :, 2:] - z_pred[:, :, :-2]) / (2 * dx[:, :, 1:-1] + 1e-8)
        dzdy = (z_pred[:, 2:, :] - z_pred[:, :-2, :]) / (2 * dy + 1e-8)
        
        f_mid = f[1:-1, 1:-1]
        v_geo = dzdx[:, 1:-1, :] / (f_mid + 1e-8)
        u_geo = -dzdy[:, :, 1:-1] / (f_mid + 1e-8)
        
        u_pred_crop = u_pred[:, 1:-1, 1:-1]
        v_pred_crop = v_pred[:, 1:-1, 1:-1]
        
        error = ((u_pred_crop - u_geo)**2 + (v_pred_crop - v_geo)**2).mean()
        return error.item()


# ============================================================
# 多指标综合 Checkpoint 管理器
# ============================================================

class MultiMetricCheckpointManager:
    """
    多指标综合 checkpoint 管理器
    
    综合评估策略:
    - 70% 验证 RMSE/Loss (数值精度)
    - 30% 物理一致性误差 (气象合理性)
    
    支持:
    - 最佳 checkpoint 保存
    - Top-K checkpoints 保存
    - 多指标排序
    """
    
    def __init__(self, checkpoint_dir, weights=None, top_k=5):
        """
        Args:
            checkpoint_dir: checkpoint 保存目录
            weights: 权重 dict, {'rmse': 0.7, 'physics': 0.3}
            top_k: 保存 top-k checkpoints
        """
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # 默认权重: 70% RMSE + 30% 物理一致性
        self.weights = weights or {'rmse': 0.7, 'physics': 0.3}
        self.top_k = top_k
        
        # 最佳记录
        self.best_score = float('inf')
        self.best_rmse = float('inf')
        self.best_physics = float('inf')
        self.best_epoch = -1
        
        # Top-K 历史
        self.history = []
        
    def compute_combined_score(self, metrics):
        """
        计算综合得分
        
        Args:
            metrics: dict with keys:
                - 'val_loss' or 'val_rmse': 验证损失
                - 'val_physics_error': 物理一致性误差 (可选)
        
        Returns:
            (score, rmse, physics_error)
        """
        rmse = metrics.get('val_loss', metrics.get('val_rmse', float('inf')))
        physics = metrics.get('val_physics_error', 0.0)
        
        # 归一化：使用当前最佳值作为基准
        if self.best_rmse > 0 and self.best_rmse < float('inf'):
            rmse_norm = rmse / self.best_rmse
        else:
            rmse_norm = rmse  # 首次评估
            
        if physics > 0 and self.best_physics > 0 and self.best_physics < float('inf'):
            physics_norm = physics / self.best_physics
        else:
            physics_norm = physics if physics > 0 else 0.0
        
        score = (self.weights['rmse'] * rmse_norm + 
                 self.weights['physics'] * physics_norm)
        
        return score, rmse, physics
    
    def is_best(self, metrics):
        """判断是否为新的最佳 checkpoint"""
        score, rmse, physics = self.compute_combined_score(metrics)
        
        if score < self.best_score:
            self.best_score = score
            if rmse < self.best_rmse:
                self.best_rmse = rmse
            if physics < self.best_physics or self.best_physics == float('inf'):
                self.best_physics = physics
            return True
        return False
    
    def save_best(self, model, optimizer, scheduler, epoch, metrics):
        """保存最佳 checkpoint"""
        score, rmse, physics = self.compute_combined_score(metrics)
        self.best_epoch = epoch
        
        checkpoint = {
            'epoch': epoch,
            'global_step': metrics.get('global_step', 0),
            'optim_step': metrics.get('optim_step', 0),
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'ema_state_dict': metrics.get('ema_state_dict', None),
            'metrics': metrics,
            'best_score': self.best_score,
            'best_rmse': self.best_rmse,
            'best_physics': self.best_physics,
            'weights': self.weights,
            'method': 'flow_matching_cfm_physical',
        }
        path = os.path.join(self.checkpoint_dir, 'best_combined.pt')
        torch.save(checkpoint, path)
        logger.info(f"✓ 最佳综合 checkpoint 已保存 (score={score:.4f}, rmse={rmse:.4f}, physics={physics:.6f})")
    
    def save_top_k(self, model, optimizer, scheduler, epoch, metrics):
        """保存 top-k checkpoints"""
        score, rmse, physics = self.compute_combined_score(metrics)
        
        entry = {
            'epoch': epoch,
            'score': score,
            'rmse': rmse,
            'physics': physics,
            'metrics': metrics.copy(),
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }
        
        self.history.append(entry)
        # 按 score 排序，保留 top-k
        self.history.sort(key=lambda x: x['score'])
        self.history = self.history[:self.top_k]
        
        # 保存每个 rank
        topk_dir = os.path.join(self.checkpoint_dir, 'top_k')
        os.makedirs(topk_dir, exist_ok=True)
        
        for i, ckpt in enumerate(self.history):
            rank_path = os.path.join(topk_dir, f'rank_{i+1}.pt')
            torch.save(ckpt, rank_path)
        
        logger.info(f"Top-{self.top_k} checkpoints 已更新")


def dynamic_weight_scheduler(epoch, warmup_epochs=100, total_epochs=200,
                              geo_weight=0.1, div_weight=0.01, vort_weight=0.01):
    """
    动态权重调度器
    
    Phase 1 (0 - warmup_epochs-1): 仅 MSE (Epoch 1-100)
    Phase 2 (warmup_epochs - 160): 线性 warmup (Epoch 101-160)
    Phase 3 (160+): 锁定权重 (Epoch 161-200)
    """
    if epoch < warmup_epochs:
        # Phase 1: 仅 MSE
        return {'mse': 1.0, 'geo': 0.0, 'div': 0.0, 'vort': 0.0}
    
    elif epoch < 160:
        # Phase 2: 线性 warmup
        progress = (epoch - warmup_epochs) / (160 - warmup_epochs)
        progress = min(max(progress, 0.0), 1.0)
        return {
            'mse': 1.0,
            'geo': geo_weight * progress,
            'div': div_weight * progress,
            'vort': vort_weight * progress,
        }
    
    else:
        # Phase 3: 锁定
        return {'mse': 1.0, 'geo': geo_weight, 'div': div_weight, 'vort': vort_weight}


# ============================================================
# EMA
# ============================================================

class EMA:
    """指数移动平均"""

    def __init__(self, model: nn.Module, decay: float = 0.9999):
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    @torch.no_grad()
    def update(self, model: nn.Module):
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name].mul_(self.decay).add_(param.data, alpha=1 - self.decay)

    def apply_shadow(self, model: nn.Module):
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])

    def restore(self, model: nn.Module):
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.backup:
                param.data.copy_(self.backup[name])
        self.backup = {}

    def state_dict(self):
        return {"shadow": self.shadow, "decay": self.decay}

    def load_state_dict(self, state_dict):
        self.shadow = state_dict["shadow"]
        self.decay = state_dict["decay"]


# ============================================================
# Trainer
# ============================================================

class CFMTrainer:
    """Flow Matching 训练器"""

    def __init__(
        self,
        model: ERA5FlowMatchingModel,
        train_loader: DataLoader,
        val_loader: DataLoader,
        train_cfg: TrainConfig,
        data_cfg: DataConfig,
        work_dir: str = ".",
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.cfg = train_cfg
        self.data_cfg = data_cfg
        self.work_dir = work_dir

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"训练设备: {self.device}")
        if self.device.type == "cuda":
            logger.info(f"GPU: {torch.cuda.get_device_name()}")
            logger.info(f"显存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

        self.model = self.model.to(self.device)

        if train_cfg.cudnn_benchmark:
            torch.backends.cudnn.benchmark = True

        # 优化器
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=train_cfg.learning_rate,
            weight_decay=train_cfg.weight_decay,
            betas=train_cfg.betas,
        )

        # 学习率调度
        self.warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            self.optimizer,
            start_factor=train_cfg.warmup_start_lr / train_cfg.learning_rate,
            end_factor=1.0,
            total_iters=train_cfg.warmup_steps,
        )
        steps_per_epoch = max(len(train_loader) // train_cfg.gradient_accumulation_steps, 1)
        total_optim_steps = steps_per_epoch * train_cfg.max_epochs
        cosine_steps = max(total_optim_steps - train_cfg.warmup_steps, 1)

        self.cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=cosine_steps,
            eta_min=train_cfg.min_lr,
        )
        self.lr_scheduler = torch.optim.lr_scheduler.SequentialLR(
            self.optimizer,
            schedulers=[self.warmup_scheduler, self.cosine_scheduler],
            milestones=[train_cfg.warmup_steps],
        )
        
        # 三阶段学习率调度器
        self._init_phase_lr_scheduler()

        # 混合精度
        self.amp_dtype = (
            torch.bfloat16 if train_cfg.amp_dtype == "bfloat16" else torch.float16
        )
        self.use_amp = train_cfg.use_amp and self.device.type == "cuda"
        self.scaler = (
            torch.amp.GradScaler("cuda")
            if self.use_amp and self.amp_dtype == torch.float16
            else None
        )

        # EMA
        self.ema = EMA(self.model, decay=train_cfg.ema_decay)

        # ===== P2: 地转平衡物理损失初始化 =====
        self.use_geostrophic_physics = train_cfg.use_geostrophic_physics
        self.geostrophic_weight = train_cfg.geostrophic_weight
        self.divergence_weight = train_cfg.divergence_weight
        self.vorticity_weight = train_cfg.vorticity_weight
        
        if self.use_geostrophic_physics:
            # 创建纬度网格 (弧度) [H, W]
            lat_deg = np.linspace(data_cfg.lat_range[1], data_cfg.lat_range[0], data_cfg.grid_size)
            lon_deg = np.linspace(data_cfg.lon_range[0], data_cfg.lon_range[1], data_cfg.grid_size)
            lat_rad = np.deg2rad(lat_deg)[:, None]  # [H, 1]
            lon_grid = np.deg2rad(lon_deg)[None, :]  # [1, W]
            lat_grid = np.repeat(lat_rad, data_cfg.grid_size, axis=1)  # [H, W]
            lat_grid_tensor = torch.tensor(lat_grid, dtype=torch.float32, device=self.device)
            
            self.physics_loss = PhysicalConsistencyLoss(
                lat_grid=lat_grid_tensor,
                lon_res=data_cfg.lon_res if hasattr(data_cfg, 'lon_res') else 0.25,
                lat_res=data_cfg.lat_res if hasattr(data_cfg, 'lat_res') else 0.25,
            ).to(self.device)
            logger.info(f"地转平衡物理损失已启用 (geo={self.geostrophic_weight}, div={self.divergence_weight}, vort={self.vorticity_weight})")
        else:
            self.physics_loss = None
        
        # ===== 多指标 Checkpoint 管理器 =====
        self.ckpt_manager = MultiMetricCheckpointManager(
            checkpoint_dir=os.path.join(work_dir, train_cfg.checkpoint_dir),
            weights={'rmse': train_cfg.checkpoint_weights[0], 'physics': train_cfg.checkpoint_weights[1]},
            top_k=train_cfg.checkpoint_top_k,
        )
        
        # P2: 物理损失退火调度
        self.physics_warmup_steps = train_cfg.physics_warmup_steps
        self.physics_warmup_type = train_cfg.physics_warmup_type
        self.physics_target_weight = train_cfg.physics_target_weight
        self.physics_loss_weight = train_cfg.physics_loss_weight
        
        # 三阶段配置
        self.phase1_end = 100  # 第一阶段结束epoch (纯 MSE warmup)
        self.phase2_end = 160  # 第二阶段结束epoch
        self.physics_start_epoch = train_cfg.physics_warmup_start_epoch  # 101
        self.physics_end_epoch = train_cfg.physics_warmup_end_epoch      # 160
        self.phase3_lr_start = train_cfg.phase3_lr_start_epoch           # 80
        self.path_perturb_prob = train_cfg.path_perturb_prob
        self.path_perturb_sigma = train_cfg.path_perturb_sigma
        
        # 计算steps per epoch用于精确调度
        self.steps_per_epoch = max(len(train_loader) // train_cfg.gradient_accumulation_steps, 1)

        # TensorBoard
        self.writer = None
        if train_cfg.use_tensorboard:
            log_dir = os.path.join(work_dir, "logs_cfm_preprocessed")
            os.makedirs(log_dir, exist_ok=True)
            self.writer = SummaryWriter(log_dir)

        # Checkpoint 目录
        self.ckpt_dir = os.path.join(work_dir, train_cfg.checkpoint_dir)
        os.makedirs(self.ckpt_dir, exist_ok=True)

        # 训练状态
        self.global_step = 0
        self.optim_step = 0
        self.epoch = 0
        self.best_val_loss = float("inf")
        self.patience_counter = 0

        if train_cfg.resume_from:
            self._load_checkpoint(train_cfg.resume_from)

    def train(self):
        """主训练循环"""
        logger.info("=" * 60)
        logger.info("Flow Matching 训练 (CFM) — preprocessed_9ch_40x40")
        logger.info("三阶段训练策略:")
        logger.info("  阶段1: 预热期 (Epoch 1-100) - 仅 loss_mse (纯MSE)")
        logger.info("  阶段2: 物理炼金期 (Epoch 101-160) - 引入物理约束")
        logger.info("  阶段3: 微调收敛期 (Epoch 161-200) - 学习率降至 1e-5")
        logger.info("=" * 60)
        logger.info(f"max_epochs={self.cfg.max_epochs}")
        logger.info(
            f"batch_size={self.cfg.batch_size} × "
            f"grad_accum={self.cfg.gradient_accumulation_steps} = "
            f"effective_batch={self.cfg.batch_size * self.cfg.gradient_accumulation_steps}"
        )
        logger.info(f"训练集: {len(self.train_loader.dataset)} 样本")
        logger.info(f"验证集: {len(self.val_loader.dataset)} 样本")

        epoch_pbar = tqdm(
            range(self.epoch, self.cfg.max_epochs),
            desc="训练进度",
            unit="epoch",
            initial=self.epoch,
            total=self.cfg.max_epochs,
        )
        for epoch in epoch_pbar:
            self.epoch = epoch
            
            # 阶段3: 更新学习率
            phase = self._get_training_phase()
            if phase == 3:
                self._update_lr_for_phase3()
            
            train_loss = self._train_one_epoch()

            epoch_pbar.set_postfix(
                loss=f"{train_loss:.4f}",
                lr=f"{self.optimizer.param_groups[0]['lr']:.2e}",
                best=f"{self.best_val_loss:.4f}",
            )
            logger.info(
                f"Epoch {epoch+1}/{self.cfg.max_epochs} | "
                f"train_loss={train_loss:.6f} | "
                f"lr={self.optimizer.param_groups[0]['lr']:.2e} | "
                f"phase={phase}"
            )

            if (epoch + 1) % self.cfg.eval_every == 0:
                val_loss, val_physics_error = self._validate()
                logger.info(f"  验证 loss={val_loss:.6f} (best={self.best_val_loss:.6f}), physics={val_physics_error:.6f}")

                if self.writer:
                    self.writer.add_scalar("val/loss", val_loss, epoch + 1)
                    self.writer.add_scalar("val/physics_error", val_physics_error, epoch + 1)

                # 多指标判断
                metrics = {
                    'val_loss': val_loss,
                    'val_physics_error': val_physics_error,
                    'global_step': self.global_step,
                    'optim_step': self.optim_step,
                }
                
                is_best_combined = self.ckpt_manager.is_best(metrics)
                is_best_loss = val_loss < self.best_val_loss
                
                if is_best_loss:
                    self.best_val_loss = val_loss
                    self.patience_counter = 0
                    self._save_checkpoint(f"best_loss.pt", is_best=True)
                    logger.info(f"  ✓ 新的最佳 loss checkpoint 已保存")
                else:
                    self.patience_counter += 1
                    logger.info(
                        f"  ✗ 无改善 ({self.patience_counter}/{self.cfg.early_stopping_patience})"
                    )
                
                if is_best_combined:
                    self.ckpt_manager.save_best(
                        self.model, self.optimizer, self.lr_scheduler,
                        epoch, metrics
                    )
                
                # 保存 top-k
                self.ckpt_manager.save_top_k(
                    self.model, self.optimizer, self.lr_scheduler,
                    epoch, metrics
                )

                if self.patience_counter >= self.cfg.early_stopping_patience:
                    logger.info(f"Early Stopping: 连续 {self.patience_counter} 次验证无改善")
                    break

            if (epoch + 1) % (self.cfg.eval_every * 5) == 0:
                self._save_checkpoint("latest.pt")

        self._save_checkpoint("final.pt")
        if self.writer:
            self.writer.close()
        logger.info("Flow Matching 三阶段训练完成!")

    def _train_one_epoch(self) -> float:
        """训练一个 epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        accum_loss = 0.0

        batch_pbar = tqdm(
            enumerate(self.train_loader),
            total=len(self.train_loader),
            desc=f"Epoch {self.epoch+1}",
            unit="batch",
            leave=False,
        )
        for batch_idx, batch in batch_pbar:
            condition = batch["condition"]
            target = batch["target"]

            if not condition.is_cuda:
                condition = condition.to(self.device)
                target = target.to(self.device)

            # 三阶段：计算当前物理权重
            physics_weight = self._compute_physics_weight()
            phase = self._get_training_phase()

            # 混合精度前向
            with torch.amp.autocast("cuda", dtype=self.amp_dtype, enabled=self.use_amp):
                outputs = self.model(condition, target)

                # P0+P2: 应用物理损失（带三阶段调度）
                loss_mse = outputs["loss_mse"]
                loss_x0 = outputs["loss_x0"]
                loss_div = outputs["loss_div"]
                loss_sol = outputs["loss_sol"]  # 螺线场损失
                loss_curl = outputs["loss_curl"]
                
                # 基础损失
                x0_weight = self.cfg.x0_loss_weight
                base_loss = loss_mse + x0_weight * loss_x0 + physics_weight * (loss_div + loss_sol + loss_curl)
                
                # ===== 新增：地转平衡物理损失 =====
                geo_loss = 0.0
                if self.use_geostrophic_physics and self.physics_loss is not None and physics_weight > 0:
                    pred = outputs.get('predicted', None)
                    if pred is not None:
                        # 提取 u, v, z 通道
                        # 通道顺序: [u_850, u_500, u_250, v_850, v_500, v_250, z_850, z_500, z_250]
                        # 选择 850 hPa 层作为代表（低层更符合地转平衡）
                        u_pred = pred[:, 0, :, :]  # u_850
                        v_pred = pred[:, 3, :, :]  # v_850
                        z_pred = pred[:, 6, :, :]  # z_850
                        
                        geo_loss, geo_comp, div_comp, vort_comp = self.physics_loss(
                            u_pred, v_pred, z_pred, return_components=True
                        )
                        base_loss = base_loss + self.geostrophic_weight * geo_loss
                # ===================================
                
                loss = base_loss / self.cfg.gradient_accumulation_steps

            # 安全检测
            loss_val = loss.item()
            if not torch.isfinite(loss) or loss_val > 10.0:
                logger.warning(
                    f"[Step {self.global_step}] 异常 loss={loss_val:.4f}, 跳过此 batch"
                )
                self.optimizer.zero_grad(set_to_none=True)
                self.global_step += 1
                continue

            # 反向传播
            if self.scaler:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            accum_loss += loss.item()
            self.global_step += 1

            if (batch_idx + 1) % self.cfg.gradient_accumulation_steps == 0:
                if self.scaler:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.max_grad_norm)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.max_grad_norm)
                    self.optimizer.step()

                self.optimizer.zero_grad(set_to_none=True)
                # 阶段3使用手动学习率调度，不使用lr_scheduler
                if phase != 3:
                    self.lr_scheduler.step()
                self.ema.update(self.model)
                self.optim_step += 1

                total_loss += accum_loss
                num_batches += 1

                if self.optim_step % self.cfg.log_every == 0:
                    lr = self.optimizer.param_groups[0]["lr"]
                    mse_val = outputs["loss_mse"].item()
                    x0_val = outputs["loss_x0"].item()
                    div_val = outputs["loss_div"].item()
                    sol_val = outputs["loss_sol"].item()
                    curl_val = outputs["loss_curl"].item()
                    geo_val = geo_loss.item() if geo_loss > 0 else 0.0
                    batch_pbar.set_postfix(
                        loss=f"{accum_loss:.4f}",
                        mse=f"{mse_val:.4f}",
                        geo=f"{geo_val:.4f}",
                        phys_w=f"{physics_weight:.3f}",
                        phase=phase,
                        lr=f"{lr:.2e}",
                    )
                    if self.writer:
                        self.writer.add_scalar("train/loss_total", accum_loss, self.optim_step)
                        self.writer.add_scalar("train/loss_mse", mse_val, self.optim_step)
                        self.writer.add_scalar("train/loss_x0", x0_val, self.optim_step)
                        self.writer.add_scalar("train/loss_div", div_val, self.optim_step)
                        self.writer.add_scalar("train/loss_sol", sol_val, self.optim_step)
                        self.writer.add_scalar("train/loss_curl", curl_val, self.optim_step)
                        self.writer.add_scalar("train/loss_geo", geo_val, self.optim_step)
                        self.writer.add_scalar("train/physics_weight", physics_weight, self.optim_step)
                        self.writer.add_scalar("train/lr", lr, self.optim_step)
                        self.writer.add_scalar("train/phase", phase, self.optim_step)
                        # 逐通道损失
                        ch_losses = outputs.get("channel_losses", {})
                        for ch_name, ch_loss in ch_losses.items():
                            self.writer.add_scalar(f"train/channel_{ch_name}", ch_loss, self.optim_step)

                    logger.info(
                        f"  step={self.optim_step} | "
                        f"loss={accum_loss:.6f} | "
                        f"mse={mse_val:.6f} | x0={x0_val:.6f} | "
                        f"div={div_val:.6f} | sol={sol_val:.6f} | curl={curl_val:.6f} | "
                        f"geo={geo_val:.6f} | "
                        f"phys_w={physics_weight:.3f} | "
                        f"phase={phase} | "
                        f"lr={lr:.2e}"
                    )

                accum_loss = 0.0

        return total_loss / max(num_batches, 1)

    @torch.no_grad()
    def _validate(self) -> Tuple[float, float]:
        """
        验证（使用 EMA 参数）
        
        Returns:
            (val_loss, physics_error)
        """
        self.ema.apply_shadow(self.model)
        self.model.eval()

        total_loss = 0.0
        total_physics_error = 0.0
        num_batches = 0

        val_pbar = tqdm(self.val_loader, desc="验证中", unit="batch", leave=False)
        for batch in val_pbar:
            condition = batch["condition"].to(self.device)
            target = batch["target"].to(self.device)

            with torch.amp.autocast("cuda", dtype=self.amp_dtype, enabled=self.use_amp):
                outputs = self.model(condition, target)
                # 验证时也应用物理损失（使用固定的 physics_target_weight）
                physics_weight = self.physics_target_weight * self.physics_loss_weight
                x0_weight = self.cfg.x0_loss_weight
                loss = outputs["loss_mse"] + x0_weight * outputs["loss_x0"] + physics_weight * (
                    outputs["loss_div"] + outputs["loss_sol"] + outputs["loss_curl"]
                )
                
                # 计算地转平衡物理误差
                physics_error = 0.0
                if self.use_geostrophic_physics and self.physics_loss is not None:
                    pred = outputs.get('predicted', None)
                    if pred is not None:
                        # 提取 u, v, z 通道
                        # 通道顺序: [u_850, u_500, u_250, v_850, v_500, v_250, z_850, z_500, z_250]
                        u_ch = 0  # u_850
                        v_ch = 3  # v_850
                        z_ch = 6  # z_850
                        
                        u_pred = pred[:, u_ch, :, :]
                        v_pred = pred[:, v_ch, :, :]
                        z_pred = pred[:, z_ch, :, :]
                        
                        physics_error = compute_physics_error(
                            u_pred, v_pred, z_pred, 
                            self.physics_loss.lat_grid,
                            self.physics_loss.lon_res,
                            self.physics_loss.lat_res,
                        )

            total_loss += loss.item()
            total_physics_error += physics_error
            num_batches += 1

        self.ema.restore(self.model)
        avg_loss = total_loss / max(num_batches, 1)
        avg_physics = total_physics_error / max(num_batches, 1)
        return avg_loss, avg_physics

    def _save_checkpoint(self, filename: str, is_best: bool = False):
        """保存 checkpoint"""
        path = os.path.join(self.ckpt_dir, filename)
        state = {
            "epoch": self.epoch,
            "global_step": self.global_step,
            "optim_step": self.optim_step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "lr_scheduler_state_dict": self.lr_scheduler.state_dict(),
            "ema_state_dict": self.ema.state_dict(),
            "best_val_loss": self.best_val_loss,
            "patience_counter": self.patience_counter,
            "method": "flow_matching_cfm_preprocessed",
        }
        torch.save(state, path)
        logger.info(f"Checkpoint 已保存: {path}")

    def _load_checkpoint(self, path: str):
        """加载 checkpoint"""
        logger.info(f"加载 checkpoint: {path}")
        state = torch.load(path, map_location=self.device, weights_only=False)
        missing, unexpected = self.model.load_state_dict(
            state["model_state_dict"], strict=False
        )
        if missing:
            logger.warning(f"⚠️ 模型中有 {len(missing)} 个 key 未从 checkpoint 加载")
        if unexpected:
            logger.warning(f"⚠️ checkpoint 中有 {len(unexpected)} 个 key 在当前模型中不存在")
        if not missing and not unexpected:
            logger.info("✅ 模型权重完全匹配")
        # 只加载模型权重，不加载 optimizer 状态
        # 原因: resume 后学习率可能与当前调度器不匹配，导致训练崩溃
        # 模型权重保留了知识，optimizer 从头开始是正确的策略
        self.epoch = 0  # 重置epoch，应用新的物理损失调度
        self.global_step = 0
        self.optim_step = 0
        self.best_val_loss = state["best_val_loss"]
        self.patience_counter = 0  # 重置耐心计数器

    def _compute_physics_weight(self) -> float:
        """
        三阶段物理权重调度
        
        阶段1 (Epoch 1-20): physics_weight = 0 (完全禁用)
        阶段2 (Epoch 21-80): 线性增加到 physics_target_weight (0.1)
        阶段3 (Epoch 81-120): 锁定 physics_target_weight (0.1)
        """
        if self.physics_target_weight <= 0:
            return 0.0
        
        epoch = self.epoch
        
        # 阶段1: 完全禁用物理损失
        if epoch < self.physics_start_epoch:
            return 0.0
        
        # 阶段2: 线性增加物理权重
        if epoch < self.physics_end_epoch:
            progress = (epoch - self.physics_start_epoch + 1) / (self.physics_end_epoch - self.physics_start_epoch)
            progress = min(max(progress, 0.0), 1.0)
            
            if self.physics_warmup_type == "linear":
                weight = progress * self.physics_target_weight
            elif self.physics_warmup_type == "cosine":
                weight = (1 - math.cos(progress * math.pi)) / 2 * self.physics_target_weight
            else:
                weight = self.physics_target_weight
            return weight
        
        # 阶段3: 锁定物理权重
        return self.physics_target_weight

    def _get_training_phase(self) -> int:
        """
        获取当前训练阶段
        
        Returns:
            1: 预热期 (Epoch 1-20)
            2: 物理炼金期 (Epoch 21-80)
            3: 微调收敛期 (Epoch 81-120)
        """
        epoch = self.epoch
        if epoch < self.physics_start_epoch:
            return 1
        elif epoch < self.physics_end_epoch:
            return 2
        else:
            return 3
    
    def _should_apply_path_perturb(self) -> bool:
        """
        判断是否应用路径扰动 (Scheduled Sampling)
        
        阶段1: 不扰动（让模型学习正确的路径）
        阶段2: 20%概率扰动（增强纠错能力）
        阶段3: 不扰动（稳定收敛）
        """
        phase = self._get_training_phase()
        if phase == 2:
            return np.random.random() < self.path_perturb_prob
        return False
    
    def _get_current_lr(self) -> float:
        """获取当前学习率"""
        return self.optimizer.param_groups[0]['lr']
    
    def _init_phase_lr_scheduler(self):
        """
        初始化第三阶段学习率调度器
        
        阶段1-2: 使用原有的 warmup + cosine 调度
        阶段3: 切换到余弦退火至 min_lr (1e-5)
        """
        # 注意：第三阶段的学习率调度在训练循环中手动处理
        pass
    
    def _update_lr_for_phase3(self):
        """
        更新第三阶段学习率
        
        当进入阶段3时，将学习率从当前值余弦退火到 min_lr
        """
        if self._get_training_phase() == 3:
            # 计算阶段3内的进度 (0.0 -> 1.0)
            phase3_progress = (self.epoch - self.phase3_lr_start) / (self.cfg.max_epochs - self.phase3_lr_start)
            phase3_progress = min(max(phase3_progress, 0.0), 1.0)
            
            # 余弦退火
            cosine_factor = (1 + math.cos(phase3_progress * math.pi)) / 2
            base_lr = self.cfg.learning_rate
            min_lr = self.cfg.min_lr
            
            # 从 base_lr 退火到 min_lr
            new_lr = min_lr + (base_lr - min_lr) * cosine_factor
            
            # 更新学习率
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = new_lr


# ============================================================
# 入口
# ============================================================

def main():
    torch.set_float32_matmul_precision("high")

    parser = argparse.ArgumentParser(description="Flow Matching Training (CFM) — preprocessed_9ch_40x40")
    parser.add_argument("--era5_dir", type=str,
                        default="/root/autodl-tmp/fyp_final/preprocessed_9ch_40x40",
                        help="预处理数据目录")
    parser.add_argument("--csv_path", type=str,
                        default="/root/autodl-tmp/fyp_final/VER3_original/VER3/Trajectory/processed_typhoon_tracks.csv",
                        help="轨迹CSV文件路径")
    parser.add_argument("--work_dir", type=str, default=".")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    data_cfg, model_cfg, train_cfg, infer_cfg = get_config(era5_dir=args.era5_dir)

    if args.csv_path:
        data_cfg.csv_path = args.csv_path
    if args.batch_size:
        train_cfg.batch_size = args.batch_size
    if args.epochs:
        train_cfg.max_epochs = args.epochs
    if args.lr:
        train_cfg.learning_rate = args.lr
    if args.resume:
        train_cfg.resume_from = args.resume
    train_cfg.seed = args.seed

    logger.info("=" * 60)
    logger.info("数据集配置 (ICLR 2024 标准)")
    logger.info(f"  ERA5目录: {data_cfg.era5_dir}")
    logger.info(f"  CSV路径: {data_cfg.csv_path}")
    logger.info(f"  训练年份: {min(data_cfg.train_years)}-{max(data_cfg.train_years)}")
    logger.info(f"  验证年份: {min(data_cfg.val_years)}-{max(data_cfg.val_years)}")
    logger.info(f"  测试年份: {min(data_cfg.test_years)}-{max(data_cfg.test_years)}")
    logger.info(f"  历史步长: {data_cfg.history_steps} ({data_cfg.history_steps * 3}h)")
    logger.info(f"  预测步长: {data_cfg.forecast_steps} ({data_cfg.forecast_steps * 3}h)")
    logger.info("=" * 60)

    logger.info("构建数据加载器...")
    train_loader, val_loader, test_loader, norm_mean, norm_std = create_preprocessed_dataloaders(
        data_cfg, train_cfg
    )
    logger.info(f"训练集: {len(train_loader.dataset)} 样本, {len(train_loader)} batches")
    logger.info(f"验证集: {len(val_loader.dataset)} 样本, {len(val_loader)} batches")

    # 探测通道数
    sample_batch = next(iter(train_loader))
    # P1: condition 现在可能是 (T, C, H, W) 格式，需要正确检测
    cond_shape = sample_batch['condition'].shape
    if len(cond_shape) == 4:
        # 原始格式: (B, C, H, W)
        era5_channels = cond_shape[1]
    elif len(cond_shape) == 5:
        # P1 新格式: (B, T, C, H, W)
        era5_channels = cond_shape[2]
        logger.info(f"P1: 检测到时序条件，shape={cond_shape}")
    else:
        raise ValueError(f"未知的 condition shape: {cond_shape}")
    logger.info(f"ERA5 channels: {era5_channels}, condition shape: {cond_shape}")

    logger.info("构建 CFM 模型...")
    model_cfg.in_channels = era5_channels
    model_cfg.cond_channels = era5_channels

    model = ERA5FlowMatchingModel(model_cfg, data_cfg, train_cfg)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"模型参数: {total_params/1e6:.2f}M (可训练: {trainable_params/1e6:.2f}M)")

    trainer = CFMTrainer(model, train_loader, val_loader, train_cfg, data_cfg, args.work_dir)
    trainer.train()

    logger.info("训练完成!")


if __name__ == "__main__":
    main()
