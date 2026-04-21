r"""
方案C: 两阶段训练 —— 先用真实ERA5预训练，再用扩散模型预测ERA5微调
解决级联系统的 train-test mismatch 问题

阶段1: 已由 train.py 完成，加载 checkpoints/best.pt
阶段2: 本脚本完成，用扩散模型生成的ERA5替换真实ERA5进行微调

使用方式:
  python finetune_train.py ^
      --pretrained_ckpt checkpoints/best.pt ^
      --diffusion_code C:\Users\fyp\Desktop\newtry ^
      --diffusion_ckpt C:\Users\fyp\Desktop\newtry\checkpoints\best.pt ^
      --norm_stats C:\Users\fyp\Desktop\newtry\norm_stats.pt ^
      --data_root C:\Users\fyp\Desktop\Typhoon_data_final ^
      --track_csv processed_typhoon_tracks.csv ^
      --finetune_epochs 50 ^
      --finetune_lr 2e-5
"""
import os
import sys
import json
import argparse
import time
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from multiprocessing import Pool, cpu_count

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import pandas as pd

# 轨迹模型
TRAJ_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, TRAJ_DIR)

from config import model_cfg, data_cfg, train_cfg
from model import LT3PModel
from dataset import (
    LT3PDataset, normalize_coords, denormalize_coords, normalize_era5,
    filter_short_storms, filter_out_of_range_storms, split_storms_by_id,
)
from data_processing import load_tyc_storms
from data_structures import StormSample
from train import evaluate_on_test


# ============================================================
# 通道说明：扩散模型和轨迹模型现在统一使用 9 通道
# ============================================================
# 两个模型都使用相同的 9 通道 (仅气压层变量):
#   [0:3]  u_850, u_500, u_250
#   [3:6]  v_850, v_500, v_250
#   [6:9]  z_850, z_500, z_250
#
# 已移除的变量:
#   - vo (涡度): 40×40 粗网格无法解析涡度空间梯度
#   - u10m, v10m, msl (地面变量): 扩散模型不预测，保持两端通道完全一致
# ============================================================


# ============================================================
# 扩散模型ERA5数据集：用预测ERA5替换真实ERA5
# ============================================================

class DiffusionERA5Dataset(Dataset):
    """
    微调数据集：轨迹用真实值，ERA5用扩散模型的预测值

    与 LT3PDataset 接口完全一致，唯一区别是 future_era5
    从扩散模型在线生成（或预先缓存）
    """

    def __init__(
        self,
        storm_samples: List[StormSample],
        diffusion_era5_cache: Dict[str, np.ndarray],
        t_history: int = None,
        t_future: int = None,
        stride: int = 1,
        era5_channels: int = None,
    ):
        """
        Args:
            storm_samples: 台风样本列表（含真实轨迹）
            diffusion_era5_cache: {storm_id: (T, C, H, W)} 扩散模型预测的ERA5
            t_history: 历史步数
            t_future: 预测步数
            stride: 滑动窗口步长
            era5_channels: ERA5通道数
        """
        self.storm_samples = storm_samples
        self.diffusion_era5_cache = diffusion_era5_cache
        self.t_history = t_history or model_cfg.t_history
        self.t_future = t_future or model_cfg.t_future
        self.stride = stride
        self.era5_channels = era5_channels or model_cfg.era5_channels
        self.total_length = self.t_history + self.t_future

        # 只保留有扩散ERA5缓存的台风
        self.valid_samples = [
            s for s in storm_samples
            if s.storm_id in diffusion_era5_cache
        ]

        self.samples_index = self._build_index()
        print(f"DiffusionERA5Dataset: {len(self.valid_samples)} storms, "
              f"{len(self.samples_index)} samples "
              f"(t_history={self.t_history}, t_future={self.t_future})")

    def _build_index(self) -> List[Tuple[int, int]]:
        index = []
        for storm_idx, sample in enumerate(self.valid_samples):
            T = len(sample)
            if T < self.total_length:
                continue
            for start in range(0, T - self.total_length + 1, self.stride):
                index.append((storm_idx, start))
        return index

    def __len__(self) -> int:
        return len(self.samples_index)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        storm_idx, start_idx = self.samples_index[idx]
        sample = self.valid_samples[storm_idx]

        history_start = start_idx
        history_end = start_idx + self.t_history
        future_start = history_end
        future_end = history_end + self.t_future

        # 历史轨迹（真实值）
        history_lat = sample.track_lat[history_start:history_end]
        history_lon = sample.track_lon[history_start:history_end]
        h_lat_n, h_lon_n = normalize_coords(history_lat, history_lon)
        history_coords = np.stack([h_lat_n, h_lon_n], axis=-1)

        # 过去ERA5（始终用真实值，历史期的ERA5是已知的）
        past_era5 = np.zeros((self.t_history, self.era5_channels, 40, 40), dtype=np.float32)
        if sample.era5_array is not None:
            raw = sample.era5_array[history_start:history_end]
            if raw.shape[1] > self.era5_channels:
                raw = raw[:, :self.era5_channels]
            past_era5[:len(raw)] = raw[:len(past_era5)]
        past_era5 = normalize_era5(past_era5)

        # 未来ERA5（扩散模型预测值）
        diff_era5 = self.diffusion_era5_cache[sample.storm_id]
        future_era5 = diff_era5[future_start:future_end]  # (T_future, C, H, W)

        # 确保通道数正确
        if future_era5.shape[1] > self.era5_channels:
            future_era5 = future_era5[:, :self.era5_channels]
        elif future_era5.shape[1] < self.era5_channels:
            T = future_era5.shape[0]
            pad = np.zeros(
                (T, self.era5_channels - future_era5.shape[1],
                 future_era5.shape[2], future_era5.shape[3]),
                dtype=np.float32
            )
            future_era5 = np.concatenate([future_era5, pad], axis=1)

        # 逐通道标准化（与 LT3PDataset 保持一致）
        future_era5 = normalize_era5(future_era5)

        # 未来轨迹目标（真实值）
        future_lat = sample.track_lat[future_start:future_end]
        future_lon = sample.track_lon[future_start:future_end]
        f_lat_n, f_lon_n = normalize_coords(future_lat, future_lon)
        target_coords = np.stack([f_lat_n, f_lon_n], axis=-1)

        # 样本权重
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
            'target_lat_raw': torch.from_numpy(future_lat).float(),
            'target_lon_raw': torch.from_numpy(future_lon).float(),
            'history_lat_raw': torch.from_numpy(history_lat).float(),
            'history_lon_raw': torch.from_numpy(history_lon).float(),
        }


# ============================================================
# 混合ERA5数据集：真实ERA5 + 扩散ERA5 混合训练（防灾难性遗忘）
# ============================================================

class MixedERA5Dataset(Dataset):
    """
    混合ERA5训练集：每个样本随机选用真实ERA5或扩散ERA5

    核心思想：
      - 纯用扩散ERA5微调 → 模型遗忘"真实风场→轨迹"的精确映射
      - 混合训练 → 既适应扩散ERA5的分布偏移，又保留真实ERA5学到的知识
      - real_ratio 控制真实:扩散的比例（默认0.5即50:50）
    """

    def __init__(
        self,
        storm_samples: List[StormSample],
        diffusion_era5_cache: Dict[str, np.ndarray],
        real_ratio: float = 0.5,
        t_history: int = None,
        t_future: int = None,
        stride: int = 1,
        era5_channels: int = None,
    ):
        self.storm_samples = storm_samples
        self.diffusion_era5_cache = diffusion_era5_cache
        self.real_ratio = real_ratio
        self.t_history = t_history or model_cfg.t_history
        self.t_future = t_future or model_cfg.t_future
        self.stride = stride
        self.era5_channels = era5_channels or model_cfg.era5_channels
        self.total_length = self.t_history + self.t_future

        # 只保留同时拥有真实ERA5和扩散ERA5缓存的台风
        self.valid_samples = [
            s for s in storm_samples
            if s.era5_array is not None and s.storm_id in diffusion_era5_cache
        ]

        self.samples_index = self._build_index()
        print(f"MixedERA5Dataset: {len(self.valid_samples)} storms, "
              f"{len(self.samples_index)} samples "
              f"(real_ratio={self.real_ratio}, "
              f"t_history={self.t_history}, t_future={self.t_future})")

    def _build_index(self) -> List[Tuple[int, int]]:
        index = []
        for storm_idx, sample in enumerate(self.valid_samples):
            T = len(sample)
            if T < self.total_length:
                continue
            for start in range(0, T - self.total_length + 1, self.stride):
                index.append((storm_idx, start))
        return index

    def __len__(self) -> int:
        return len(self.samples_index)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        storm_idx, start_idx = self.samples_index[idx]
        sample = self.valid_samples[storm_idx]

        history_start = start_idx
        history_end = start_idx + self.t_history
        future_start = history_end
        future_end = history_end + self.t_future

        # 历史轨迹（始终用真实值）
        history_lat = sample.track_lat[history_start:history_end]
        history_lon = sample.track_lon[history_start:history_end]
        h_lat_n, h_lon_n = normalize_coords(history_lat, history_lon)
        history_coords = np.stack([h_lat_n, h_lon_n], axis=-1)

        # 过去ERA5（始终用真实值）
        past_era5 = np.zeros((self.t_history, self.era5_channels, 40, 40), dtype=np.float32)
        if sample.era5_array is not None:
            raw = sample.era5_array[history_start:history_end]
            if raw.shape[1] > self.era5_channels:
                raw = raw[:, :self.era5_channels]
            past_era5[:len(raw)] = raw[:len(past_era5)]
        past_era5 = normalize_era5(past_era5)

        # 未来ERA5：按概率随机选择真实/扩散
        use_real = np.random.random() < self.real_ratio

        if use_real:
            # 用真实ERA5
            future_era5 = sample.era5_array[future_start:future_end]
        else:
            # 用扩散模型预测的ERA5
            diff_era5 = self.diffusion_era5_cache[sample.storm_id]
            future_era5 = diff_era5[future_start:future_end]

        # 确保通道数正确
        if future_era5.shape[1] > self.era5_channels:
            future_era5 = future_era5[:, :self.era5_channels]
        elif future_era5.shape[1] < self.era5_channels:
            T = future_era5.shape[0]
            pad = np.zeros(
                (T, self.era5_channels - future_era5.shape[1],
                 future_era5.shape[2], future_era5.shape[3]),
                dtype=np.float32
            )
            future_era5 = np.concatenate([future_era5, pad], axis=1)

        # 逐通道标准化（与 LT3PDataset 保持一致）
        future_era5 = normalize_era5(future_era5)

        # 未来轨迹目标（始终用真实值）
        future_lat = sample.track_lat[future_start:future_end]
        future_lon = sample.track_lon[future_start:future_end]
        f_lat_n, f_lon_n = normalize_coords(future_lat, future_lon)
        target_coords = np.stack([f_lat_n, f_lon_n], axis=-1)

        # 样本权重
        sample_weight = 1.0
        if sample.is_real is not None:
            real_ratio_w = sample.is_real[future_start:future_end].mean()
            sample_weight = train_cfg.interp_sample_weight + \
                real_ratio_w * (train_cfg.real_sample_weight - train_cfg.interp_sample_weight)

        return {
            'history_coords': torch.from_numpy(history_coords).float(),
            'past_era5': torch.from_numpy(past_era5).float(),
            'future_era5': torch.from_numpy(future_era5).float(),
            'target_coords': torch.from_numpy(target_coords).float(),
            'sample_weight': torch.tensor(sample_weight).float(),
            'is_diffusion': torch.tensor(0.0 if use_real else 1.0).float(),
            'storm_id': sample.storm_id,
            'target_lat_raw': torch.from_numpy(future_lat).float(),
            'target_lon_raw': torch.from_numpy(future_lon).float(),
            'history_lat_raw': torch.from_numpy(history_lat).float(),
            'history_lon_raw': torch.from_numpy(history_lon).float(),
        }


# ============================================================
# 用扩散模型为每个台风生成完整时间线的ERA5预测缓存
# (多起点自回归 + 正确时间对齐)
# ============================================================

def generate_diffusion_era5_cache(
    storm_samples: List[StormSample],
    diffusion_code: str,
    diffusion_ckpt: str,
    norm_stats_path: str,
    data_root: str,
    device: str = 'cuda',
    ddim_steps: int = 50,
    preprocess_dir: str = None,
) -> Dict[str, np.ndarray]:
    """
    用扩散模型为每个台风生成完整时间线的 ERA5 预测缓存

    修复点：
    1) 时间对齐：cache[t] 必须对应台风第 t 个时间步
    2) 覆盖完整：不能只生成 24 帧后用最后一帧暴力填充整条台风
    3) 误差控制：每隔 24 步重启一次自回归，避免超长滚动误差累积

    返回: {storm_id: (storm_len, 9, 40, 40)} 扩散预测ERA5（物理值，9通道与轨迹模型一致）
    """
    print("\n" + "=" * 60)
    print("生成扩散模型 ERA5 预测缓存 (多起点策略)...")
    print("=" * 60)

    # 导入扩散模型（需要清理模块缓存，避免与 Trajectory 同名模块冲突）
    conflicting_modules = ['train', 'configs', 'models', 'inference', 'data', 'data.dataset']
    saved_modules = {}
    for mod_name in conflicting_modules:
        if mod_name in sys.modules:
            saved_modules[mod_name] = sys.modules.pop(mod_name)

    if diffusion_code not in sys.path:
        sys.path.insert(0, diffusion_code)
    elif sys.path[0] != diffusion_code:
        sys.path.remove(diffusion_code)
        sys.path.insert(0, diffusion_code)

    from configs import get_config as diff_get_config
    from models import ERA5DiffusionModel
    from train import EMA as DiffEMA
    from inference import ERA5Predictor
    from data.dataset import ERA5TyphoonDataset

    # 恢复被清理的 Trajectory 模块（避免影响后续代码）
    diff_modules = {}
    for mod_name in conflicting_modules:
        if mod_name in sys.modules:
            diff_modules[mod_name] = sys.modules[mod_name]
    for mod_name, mod_obj in saved_modules.items():
        sys.modules[mod_name] = mod_obj

    diff_data_cfg, diff_model_cfg, _, diff_infer_cfg = diff_get_config(data_root=data_root)
    diff_history_steps = diff_data_cfg.history_steps  # 当前扩散模型为 5

    # 归一化统计
    stats = torch.load(norm_stats_path, weights_only=True, map_location='cpu')
    norm_mean = stats['mean'].numpy()
    norm_std = stats['std'].numpy()

    # 加载扩散模型
    diff_model = ERA5DiffusionModel(diff_model_cfg, diff_data_cfg).to(device)
    ckpt = torch.load(diffusion_ckpt, map_location=device, weights_only=False)
    if 'ema_state_dict' in ckpt:
        ema = DiffEMA(diff_model, decay=0.9999)
        ema.load_state_dict(ckpt['ema_state_dict'])
        ema.apply_shadow(diff_model)
        print("  已加载扩散模型 EMA 参数")
    else:
        diff_model.load_state_dict(ckpt['model_state_dict'])
    diff_model.eval()

    # 推理器
    diff_infer_cfg.ddim_steps = ddim_steps
    predictor = ERA5Predictor(
        diff_model, diff_data_cfg, diff_infer_cfg,
        norm_mean, norm_std, torch.device(device)
    )

    # 反归一化常量 (扩散模型与轨迹模型使用相同的 9 通道)
    diff_num_channels = diff_data_cfg.num_channels  # 9
    traj_num_channels = 9  # 轨迹模型也用 9 通道（仅气压层变量）
    mean_t = torch.from_numpy(norm_mean[:diff_num_channels]).float().to(device).reshape(1, -1, 1, 1)
    std_t = torch.from_numpy(norm_std[:diff_num_channels]).float().to(device).reshape(1, -1, 1, 1)
    std_t = torch.where(std_t < 1e-8, torch.ones_like(std_t), std_t)

    storm_ids = list({s.storm_id for s in storm_samples})
    num_ar_steps = 24  # 必须保持 24 步 (72h)，与真实推理一致
    ensemble_size = 1   # 单次采样保留空间细节（实验证明比 ensemble=3 更好）
    ar_noise_sigma = 0.02  # 自回归噪声

    print(f"  台风数: {len(storm_ids)}")
    print(f"  策略: 每 {num_ar_steps} 步重启自回归，集合均值 {ensemble_size} 成员，噪声 σ={ar_noise_sigma}")

    try:
        full_dataset = ERA5TyphoonDataset(
            typhoon_ids=storm_ids,
            data_root=data_root,
            pl_vars=diff_data_cfg.pressure_level_vars,
            sfc_vars=diff_data_cfg.surface_vars,
            pressure_levels=diff_data_cfg.pressure_levels,
            history_steps=diff_data_cfg.history_steps,
            forecast_steps=diff_data_cfg.forecast_steps,
            norm_mean=norm_mean,
            norm_std=norm_std,
            preprocessed_dir=preprocess_dir,
        )
    except Exception as e:
        print(f"  创建ERA5数据集失败: {e}")
        return {}

    print(f"  ERA5数据集: {len(full_dataset)} 个样本")

    # 建立台风 -> 样本索引映射
    # 元素: (dataset_idx, cond_start_step)
    tid_to_samples: Dict[str, List[Tuple[int, int]]] = {}
    tid_sample_counter: Dict[str, int] = {}

    for ds_idx in range(len(full_dataset)):
        sample_meta = full_dataset.samples[ds_idx]
        tid = sample_meta[0]

        # 预处理模式: sample_meta = (tid, start_idx)
        # NC模式: sample_meta = (tid, window_files)，用出现顺序近似 start_idx
        if full_dataset.preprocessed_dir:
            cond_start = int(sample_meta[1])
        else:
            cond_start = tid_sample_counter.get(tid, 0)
            tid_sample_counter[tid] = cond_start + 1

        tid_to_samples.setdefault(tid, []).append((ds_idx, cond_start))

    cache = {}

    # ========== 并行策略: 收集所有台风的所有段, 按 batch 批量推理 ==========
    # 第一步: 收集所有 (sid, ds_idx, pred_start, steps) 任务
    all_tasks = []  # [(sid, ds_idx, pred_start, steps), ...]

    for sid in storm_ids:
        storm_obj = next((s for s in storm_samples if s.storm_id == sid), None)
        if storm_obj is None:
            continue

        storm_len = len(storm_obj)
        if sid not in tid_to_samples or len(tid_to_samples[sid]) == 0:
            continue

        # 选择多起点 condition
        available = tid_to_samples[sid]
        selected = []
        next_needed = 0

        for ds_idx, cond_start in available:
            pred_start = cond_start + diff_history_steps
            if pred_start >= next_needed:
                selected.append((ds_idx, pred_start))
                next_needed = pred_start + num_ar_steps

        if len(selected) == 0:
            ds_idx, cond_start = available[0]
            selected.append((ds_idx, cond_start + diff_history_steps))

        for ds_idx, pred_start in selected:
            remaining = storm_len - pred_start
            steps = min(num_ar_steps, max(remaining, 0))
            if steps > 0:
                all_tasks.append((sid, ds_idx, pred_start, steps))

    print(f"  总推理任务数: {len(all_tasks)}, 开始批量推理...")

    # 第二步: 按 batch 并行推理 (按 steps 分组, 相同步数的可以 batch)
    # 简单高效策略: 固定 batch_size, 逐批处理
    BATCH_SIZE = 8  # 根据显存调整, 8 个自回归链并行
    task_results = [None] * len(all_tasks)  # 存储每个 task 的结果

    for batch_start in tqdm(range(0, len(all_tasks), BATCH_SIZE), desc="批量生成扩散ERA5"):
        batch_tasks = all_tasks[batch_start : batch_start + BATCH_SIZE]
        batch_steps = max(t[3] for t in batch_tasks)  # 取最大步数

        # 收集 conditions
        conds = []
        for sid, ds_idx, pred_start, steps in batch_tasks:
            sample = full_dataset[ds_idx]
            conds.append(sample['condition'])

        cond_batch = torch.stack(conds, dim=0).to(device)  # (B, T*C, H, W)

        with torch.no_grad():
            if ensemble_size > 1:
                # 每个样本复制 ensemble_size 份
                cond_batch_ens = cond_batch.repeat_interleave(ensemble_size, dim=0)
                batch_preds = predictor.predict_autoregressive(
                    cond_batch_ens, num_steps=batch_steps, noise_sigma=ar_noise_sigma,
                )
                # reshape 回 (ensemble, B, C, H, W) 取均值
                B = len(batch_tasks)
                preds_list = []
                for p in batch_preds:
                    p_reshaped = p.reshape(B, ensemble_size, *p.shape[1:])
                    preds_list.append(p_reshaped.mean(dim=1))  # (B, C, H, W)
                batch_preds = preds_list
            else:
                batch_preds = predictor.predict_autoregressive(
                    cond_batch, num_steps=batch_steps, noise_sigma=ar_noise_sigma,
                )

        # 分发结果到每个 task
        for i, (sid, ds_idx, pred_start, steps) in enumerate(batch_tasks):
            preds_i = []
            for k in range(steps):
                pred = batch_preds[k][i:i+1]  # (1, C, H, W)
                p = pred[:, :diff_num_channels] if pred.shape[1] > diff_num_channels else pred
                p_phys = p * std_t + mean_t
                preds_i.append(p_phys.cpu().numpy()[0])  # (C, H, W)
            task_results[batch_start + i] = (sid, pred_start, preds_i)

    # 第三步: 组装 cache
    storm_meta = {}  # {sid: (storm_len, era5_cache, filled)}
    for sid in storm_ids:
        storm_obj = next((s for s in storm_samples if s.storm_id == sid), None)
        if storm_obj is None:
            continue
        storm_len = len(storm_obj)
        if sid not in tid_to_samples or len(tid_to_samples[sid]) == 0:
            continue
        era5_cache = np.zeros((storm_len, traj_num_channels, 40, 40), dtype=np.float32)
        filled = np.zeros(storm_len, dtype=bool)
        storm_meta[sid] = (storm_len, era5_cache, filled)

    for result in task_results:
        if result is None:
            continue
        sid, pred_start, preds_i = result
        if sid not in storm_meta:
            continue
        storm_len, era5_cache, filled = storm_meta[sid]
        for k, p_np in enumerate(preds_i):
            t_idx = pred_start + k
            if 0 <= t_idx < storm_len and not filled[t_idx]:
                era5_cache[t_idx] = p_np
                filled[t_idx] = True

    # 第四步: 填补空位 (线性插值 + 边界外推)
    for sid, (storm_len, era5_cache, filled) in storm_meta.items():
        if filled.any():
            first_valid = int(np.argmax(filled))
            for t in range(first_valid):
                era5_cache[t] = era5_cache[first_valid]

            last_valid = first_valid
            for t in range(first_valid + 1, storm_len):
                if filled[t]:
                    if t - last_valid > 1:
                        for gap_t in range(last_valid + 1, t):
                            alpha = (gap_t - last_valid) / (t - last_valid)
                            era5_cache[gap_t] = (
                                (1 - alpha) * era5_cache[last_valid]
                                + alpha * era5_cache[t]
                            )
                    last_valid = t

            if not filled[storm_len - 1]:
                for gap_t in range(last_valid + 1, storm_len):
                    era5_cache[gap_t] = era5_cache[last_valid]

        cache[sid] = era5_cache

    print(f"\n扩散ERA5缓存完成: {len(cache)}/{len(storm_ids)} 个台风")
    return cache


# ============================================================
# compare_fm_dm FM-ERA5 缓存生成
# ============================================================

def generate_comparefm_era5_cache(
    storm_samples: List[StormSample],
    comparefm_code: str,
    comparefm_ckpt: str,
    norm_stats_path: str,
    data_root: str,
    device: str = 'cuda',
    euler_steps: int = 4,
    preprocess_dir: str = None,
) -> Dict[str, np.ndarray]:
    """
    使用 compare_fm_dm 的 UnifiedModel (FM模式) 生成 ERA5 缓存

    与 generate_diffusion_era5_cache() 接口完全一致，只是底层模型不同
    """
    print("\n" + "=" * 60)
    print("Generating compare_fm_dm FM ERA5 cache (Euler sampling)...")
    print("=" * 60)

    # 临时保存当前 sys.modules 中可能冲突的模块
    conflicting_modules = ['configs', 'models', 'data', 'data.dataset', 'train']
    saved_modules = {}
    for mod_name in conflicting_modules:
        if mod_name in sys.modules:
            saved_modules[mod_name] = sys.modules.pop(mod_name)

    # 加入 compare_fm_dm 到 sys.path
    if comparefm_code not in sys.path:
        sys.path.insert(0, comparefm_code)
    elif sys.path[0] != comparefm_code:
        sys.path.remove(comparefm_code)
        sys.path.insert(0, comparefm_code)

    # 导入 compare_fm_dm 模块
    from configs import get_config as cfm_get_config
    from models import UnifiedModel
    from train import EMA as CFMEMA
    from data.dataset import ERA5TyphoonDataset

    # 恢复 Trajectory 模块到 sys.modules 前端
    for mod_name, mod_obj in saved_modules.items():
        sys.modules[mod_name] = mod_obj

    # 加载 compare_fm_dm 配置
    cfm_data_cfg, cfm_model_cfg, cfm_train_cfg, cfm_infer_cfg = cfm_get_config(
        data_root=data_root
    )

    # 加载归一化统计
    stats = torch.load(norm_stats_path, weights_only=True, map_location='cpu')
    norm_mean = stats['mean'].numpy()
    norm_std = stats['std'].numpy()

    # 加载 FM 模型
    cfm_model = UnifiedModel(
        cfm_model_cfg, cfm_data_cfg, cfm_train_cfg, method='fm'
    ).to(device)

    ckpt = torch.load(comparefm_ckpt, map_location=device, weights_only=False)
    if 'ema_state_dict' in ckpt:
        ema = CFMEMA(cfm_model, decay=0.9999)
        ema.load_state_dict(ckpt['ema_state_dict'])
        ema.apply_shadow(cfm_model)
        print("  已加载 compare_fm_dm FM EMA 参数")
    else:
        cfm_model.load_state_dict(ckpt['model_state_dict'])
        print("  已加载 compare_fm_dm FM 模型")
    cfm_model.eval()

    cfm_num_channels = cfm_data_cfg.num_channels  # 9
    grid_size = cfm_data_cfg.grid_size  # 40
    history_steps = cfm_data_cfg.history_steps  # 16
    traj_num_channels = 9

    # 归一化参数 tensor
    mean_t = torch.from_numpy(norm_mean[:cfm_num_channels]).float().to(device).reshape(1, -1, 1, 1)
    std_t = torch.from_numpy(norm_std[:cfm_num_channels]).float().to(device).reshape(1, -1, 1, 1)
    std_t = torch.where(std_t < 1e-8, torch.ones_like(std_t), std_t)

    storm_ids = list({s.storm_id for s in storm_samples})
    num_ar_steps = 24

    print(f"  台风数: {len(storm_ids)}")
    print(f"  策略: 每 {num_ar_steps} 步重启自回归, Euler步数={euler_steps}")

    try:
        full_dataset = ERA5TyphoonDataset(
            typhoon_ids=storm_ids,
            data_root=data_root,
            pl_vars=cfm_data_cfg.pressure_level_vars,
            sfc_vars=cfm_data_cfg.surface_vars,
            pressure_levels=cfm_data_cfg.pressure_levels,
            history_steps=cfm_data_cfg.history_steps,
            forecast_steps=cfm_data_cfg.forecast_steps,
            norm_mean=norm_mean,
            norm_std=norm_std,
            preprocessed_dir=preprocess_dir,
        )
    except Exception as e:
        print(f"  创建ERA5数据集失败: {e}")
        return {}

    print(f"  ERA5数据集: {len(full_dataset)} 个样本")

    # 构建台风→样本索引映射
    tid_to_samples: Dict[str, List[Tuple[int, int]]] = {}
    tid_sample_counter: Dict[str, int] = {}

    for ds_idx in range(len(full_dataset)):
        sample_meta = full_dataset.samples[ds_idx]
        tid = sample_meta[0]
        if full_dataset.preprocessed_dir:
            cond_start = int(sample_meta[1])
        else:
            cond_start = tid_sample_counter.get(tid, 0)
            tid_sample_counter[tid] = cond_start + 1
        tid_to_samples.setdefault(tid, []).append((ds_idx, cond_start))

    # ========== 构建推理任务 ==========
    all_tasks = []  # (sid, ds_idx, pred_start, steps)

    for sid in storm_ids:
        storm_obj = next((s for s in storm_samples if s.storm_id == sid), None)
        if storm_obj is None:
            continue
        storm_len = len(storm_obj)
        if sid not in tid_to_samples or len(tid_to_samples[sid]) == 0:
            continue

        available = tid_to_samples[sid]
        selected = []
        next_needed = 0

        for ds_idx, cond_start in available:
            pred_start = cond_start + history_steps
            if pred_start >= next_needed:
                selected.append((ds_idx, pred_start))
                next_needed = pred_start + num_ar_steps

        if len(selected) == 0:
            ds_idx, cond_start = available[0]
            selected.append((ds_idx, cond_start + history_steps))

        for ds_idx, pred_start in selected:
            remaining = storm_len - pred_start
            steps = min(num_ar_steps, max(remaining, 0))
            if steps > 0:
                all_tasks.append((sid, ds_idx, pred_start, steps))

    print(f"  总推理任务数: {len(all_tasks)}")

    # ========== 批量推理 ==========
    BATCH_SIZE = 4
    task_results = [None] * len(all_tasks)

    for batch_start in tqdm(range(0, len(all_tasks), BATCH_SIZE), desc="批量生成FM ERA5"):
        batch_tasks = all_tasks[batch_start: batch_start + BATCH_SIZE]
        batch_steps = max(t[3] for t in batch_tasks)

        conds = []
        for sid, ds_idx, pred_start, steps in batch_tasks:
            sample = full_dataset[ds_idx]
            conds.append(sample['condition'])  # (T*C, H, W)
        cond_batch = torch.stack(conds, dim=0).to(device)  # (B, T*C, H, W)

        with torch.no_grad():
            for i in range(len(batch_tasks)):
                cond_i = cond_batch[i:i+1].clone()
                preds_i = []

                for step in range(batch_steps):
                    pred = cfm_model.sample_fm(
                        cond_i,
                        device=device,
                        euler_steps=euler_steps,
                        euler_mode='midpoint',
                        clamp_range=getattr(cfm_model, 'clamp_range', (-5.0, 5.0)),
                    )
                    preds_i.append(pred)

                    if step < batch_steps - 1:
                        C = cfm_num_channels
                        T = history_steps
                        cond_5d = cond_i.view(1, T, C, grid_size, grid_size)
                        cond_5d = torch.cat([cond_5d[:, 1:], pred.unsqueeze(1)], dim=1)
                        cond_i = cond_5d.view(1, -1, grid_size, grid_size)

                task_results[batch_start + i] = (batch_tasks[i][0], batch_tasks[i][2], preds_i)

    # ========== 组装 cache ==========
    cache = {}
    filled_masks = {}

    for sid in storm_ids:
        storm_obj = next((s for s in storm_samples if s.storm_id == sid), None)
        if storm_obj is None:
            continue
        storm_len = len(storm_obj)
        cache[sid] = np.zeros((storm_len, traj_num_channels, 40, 40), dtype=np.float32)
        filled_masks[sid] = np.zeros(storm_len, dtype=bool)

    for sid, pred_start, preds_i in task_results:
        if sid not in cache:
            continue
        for k, p_tensor in enumerate(preds_i):
            t_idx = pred_start + k
            if 0 <= t_idx < len(cache[sid]) and not filled_masks[sid][t_idx]:
                p = p_tensor[:, :cfm_num_channels] if p_tensor.shape[1] > cfm_num_channels else p_tensor
                p_phys = p * std_t + mean_t
                cache[sid][t_idx] = p_phys.cpu().numpy()[0]
                filled_masks[sid][t_idx] = True

    # ========== 填补空位 ==========
    for sid in storm_ids:
        if sid not in cache:
            continue
        filled = filled_masks[sid]
        era5_cache = cache[sid]

        if filled.any():
            first_valid = int(np.argmax(filled))
            for t in range(first_valid):
                era5_cache[t] = era5_cache[first_valid]

            last_valid = first_valid
            storm_len = len(era5_cache)
            for t in range(first_valid + 1, storm_len):
                if filled[t]:
                    if t - last_valid > 1:
                        for gap_t in range(last_valid + 1, t):
                            alpha = (gap_t - last_valid) / (t - last_valid)
                            era5_cache[gap_t] = (1 - alpha) * era5_cache[last_valid] + alpha * era5_cache[t]
                    last_valid = t

            if not filled[storm_len - 1]:
                for gap_t in range(last_valid + 1, storm_len):
                    era5_cache[gap_t] = era5_cache[last_valid]

    print(f"\nFM ERA5缓存完成: {len(cache)}/{len(storm_ids)} 个台风")
    return cache


# ============================================================
# ERA5输入适配器：解决扩散ERA5与真实ERA5的分布偏移
# ============================================================

class ERA5AdaptedModel(nn.Module):
    """
    在原始 LT3PModel 外面包一层 ERA5 输入适配器

    不修改 model.py，仅在 ERA5 送入模型前做分布对齐：
      扩散ERA5 → 逐通道仿射变换(可学习) → 原始模型

    与 InstanceNorm 的区别：
      - InstanceNorm 会抹掉每个样本的绝对量级（强台风/弱台风被归一化到同一尺度）
      - 通道仿射变换只做全局 scale/bias 修正，保留样本间的物理差异
      - 更适合气象数据：只修正扩散模型的系统性偏差（如风速偏低10%）

    初始化为恒等映射：gamma=1, beta=0，训练开始时不改变任何东西。
    """

    def __init__(self, base_model: nn.Module, era5_channels: int = 9):
        super().__init__()
        self.base_model = base_model

        # 逐通道可学习仿射变换：y = gamma * x + beta
        # 初始化为恒等映射（gamma=1, beta=0）
        self.channel_scale = nn.Parameter(torch.ones(1, 1, era5_channels, 1, 1))
        self.channel_bias = nn.Parameter(torch.zeros(1, 1, era5_channels, 1, 1))

    def forward(self, history_coords, future_era5, target_coords=None, past_era5=None):
        # future_era5: (B, T, C, H, W)
        era5_adapted = future_era5 * self.channel_scale + self.channel_bias
        return self.base_model(history_coords, era5_adapted, target_coords, past_era5=past_era5)

    def predict(self, history_coords, future_era5, past_era5=None):
        """推理接口（供 evaluate_on_test 调用）"""
        era5_adapted = future_era5 * self.channel_scale + self.channel_bias
        return self.base_model.predict(history_coords, era5_adapted, past_era5=past_era5)


class ERA5ConvAdaptedModel(nn.Module):
    """
    增强版 ERA5 适配器：1×1 Conv Bottleneck + 残差连接

    vs 旧版 ERA5AdaptedModel (channel affine, 仅18参数):
      - 旧版: y = gamma * x + beta (逐通道独立，无法学习跨通道关联)
      - 新版: y = x + MLP(x) (跨通道交互，能学习 u-v 风场的耦合偏移)

    设计原理:
      1) Bottleneck 结构: C → 4C → C (9→36→9)
         扩散模型的分布偏移不只是逐通道的 scale/shift，还包括
         跨通道关联 (如 u-v 风场、不同高度层的耦合偏移)
      2) 1×1 Conv (而非 3×3): 只修正通道分布，不改变空间模式
         空间特征的提取由 PhysicsEncoder3D 负责
      3) 残差连接 + 零初始化: 训练初始 = 恒等映射，安全稳定
      4) 参数量 ~700 (vs 18)，更有表达力但仍然不会过拟合
    """

    def __init__(self, base_model: nn.Module, era5_channels: int = 9):
        super().__init__()
        self.base_model = base_model
        self.era5_channels = era5_channels

        # 1×1 Conv bottleneck: C → 4C → C
        hidden_dim = era5_channels * 4  # 36
        self.adapter = nn.Sequential(
            nn.Conv2d(era5_channels, hidden_dim, kernel_size=1, bias=True),
            nn.GELU(),
            nn.Conv2d(hidden_dim, era5_channels, kernel_size=1, bias=True),
        )

        # 零初始化最后一层 → 残差初始状态 = 恒等映射 (y = x + 0)
        nn.init.zeros_(self.adapter[-1].weight)
        nn.init.zeros_(self.adapter[-1].bias)
        # 第一层小幅 Xavier 初始化，避免初期梯度过大
        nn.init.xavier_uniform_(self.adapter[0].weight, gain=0.1)
        nn.init.zeros_(self.adapter[0].bias)

    def _adapt_era5(self, future_era5):
        """对 ERA5 输入做跨通道适配 (残差结构)"""
        B, T, C, H, W = future_era5.shape
        x = future_era5.reshape(B * T, C, H, W)
        x = x + self.adapter(x)  # 残差连接: identity + learned correction
        return x.reshape(B, T, C, H, W)

    def forward(self, history_coords, future_era5, target_coords=None, past_era5=None):
        era5_adapted = self._adapt_era5(future_era5)
        return self.base_model(history_coords, era5_adapted, target_coords, past_era5=past_era5)

    def predict(self, history_coords, future_era5, past_era5=None):
        """推理接口（供 evaluate_on_test 调用）"""
        era5_adapted = self._adapt_era5(future_era5)
        return self.base_model.predict(history_coords, era5_adapted, past_era5=past_era5)


# ============================================================
# 微调训练器
# ============================================================

# ============================================================
# Post-hoc 偏差校正 (MOS - Model Output Statistics)
# ============================================================

class BiasCorrector(nn.Module):
    """
    Lead-time 偏差校正器

    气象预测标准技术：模型在每个预报时步可能有系统性偏差
    （例如 72h 时一致偏北 0.2°），这种偏差可以从数据集上统计出来，
    推理时直接减掉。

    不改变模型权重，零训练成本。
    """

    def __init__(self, base_model: nn.Module, bias: torch.Tensor):
        super().__init__()
        self.base_model = base_model
        self.register_buffer('bias', bias)  # (T_future, 2)

    def predict(self, history_coords, future_era5, past_era5=None):
        outputs = self.base_model.predict(history_coords, future_era5, past_era5=past_era5)
        corrected = outputs['predicted_coords'] - self.bias.unsqueeze(0)  # (B, T, 2) - (1, T, 2)
        corrected = corrected.clamp(0.0, 1.0)
        outputs['predicted_coords'] = corrected
        return outputs


@torch.no_grad()
def compute_lead_time_bias(model, dataloader, device):
    """
    计算模型在每个预报时步的系统性偏差

    bias[t] = mean(predicted[t] - target[t])  在归一化坐标空间

    如果模型在某个时步一致偏北/偏东，bias 会捕捉到这个系统性误差。
    """
    model.eval()
    all_pred = []
    all_target = []

    for batch in dataloader:
        history = batch['history_coords'].to(device)
        era5 = batch['future_era5'].to(device)
        target = batch['target_coords'].to(device)

        outputs = model.predict(history, era5)
        pred = outputs['predicted_coords']

        all_pred.append(pred.cpu())
        all_target.append(target.cpu())

    if len(all_pred) == 0:
        return None

    all_pred = torch.cat(all_pred, dim=0)     # (N, T, 2)
    all_target = torch.cat(all_target, dim=0) # (N, T, 2)

    # 使用 trimmed mean（去掉最大5%误差的样本）避免离群值污染偏差估计
    errors = (all_pred - all_target)  # (N, T, 2)
    sample_mean_err = errors.abs().mean(dim=(1, 2))  # (N,)
    threshold = torch.quantile(sample_mean_err, 0.95)
    mask = sample_mean_err <= threshold  # 排除最差5%样本

    bias = errors[mask].mean(dim=0)  # (T, 2) — 每个时步的平均偏差
    print(f"  偏差校正: 使用 {mask.sum()}/{len(mask)} 个样本 (排除top 5%离群)")
    print(f"  最大偏差: {bias.abs().max():.5f} (归一化坐标)")

    return bias


class FinetuneTrainer:
    """
    阶段2微调训练器
    加载阶段1预训练权重，用扩散ERA5数据微调

    核心策略：防止灾难性遗忘
      - physics_only (默认): 仅微调 PhysicsEncoder3D，冻结其余层
        → 让物理编码器适应扩散ERA5分布，同时完整保留轨迹预测能力
      - discriminative: PhysicsEncoder3D 用完整LR，其余层用 1/10 LR
        → 允许轨迹层轻微适应，但以物理编码器为主
      - all: 全部参数同一LR微调（旧行为，容易灾难性遗忘，不推荐）
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        era5_channels: int,
        device: str = 'cuda',
        learning_rate: float = 5e-5,
        num_epochs: int = 50,
        checkpoint_dir: str = 'checkpoints_finetune',
        freeze_strategy: str = 'physics_only',
    ):
        self.device = device
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.num_epochs = num_epochs
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.era5_channels = era5_channels
        self.freeze_strategy = freeze_strategy

        # ===== 选择性冻结策略（防止灾难性遗忘）=====
        if freeze_strategy == 'physics_only':
            # 仅微调物理编码器 + 适配器 → 适应扩散ERA5分布
            # 冻结轨迹编码器/运动编码器/轨迹预测器 → 保留轨迹预测能力
            for name, param in model.named_parameters():
                if ('physics_encoder' not in name
                    and 'channel_scale' not in name and 'channel_bias' not in name
                    and 'adapter' not in name):
                    param.requires_grad = False
            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            total = sum(p.numel() for p in model.parameters())
            print(f"  [冻结策略: physics_only] 仅微调物理编码器")
            print(f"  可训练参数: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")

            self.optimizer = AdamW(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=learning_rate,
                weight_decay=1e-5,
                betas=(0.9, 0.999),
            )

        elif freeze_strategy == 'discriminative':
            # 差分学习率：物理编码器+适配器用完整LR，其余用 1/10 LR
            physics_params = []
            other_params = []
            for name, param in model.named_parameters():
                if ('physics_encoder' in name or 'channel_scale' in name
                    or 'channel_bias' in name or 'adapter' in name):
                    physics_params.append(param)
                else:
                    other_params.append(param)
            print(f"  [冻结策略: discriminative] 差分学习率")
            print(f"  PhysicsEncoder LR: {learning_rate:.1e}, 其余: {learning_rate*0.1:.1e}")

            self.optimizer = AdamW([
                {'params': physics_params, 'lr': learning_rate},
                {'params': other_params, 'lr': learning_rate * 0.1},
            ], weight_decay=1e-5, betas=(0.9, 0.999))

        elif freeze_strategy == 'bridge':
            # 桥接策略：PhysicsEncoder + 输出投影层 用完整LR
            #           Transformer 解码器中间层冻结
            #           轨迹编码器/运动编码器 用低LR
            # 原理：ERA5特征分布变了，入口(encoder)和出口(output_proj)都需要适配
            #       但中间的Transformer注意力层学到的是通用的时空推理能力，应该保留
            high_lr_params = []
            low_lr_params = []
            frozen_count = 0
            for name, param in model.named_parameters():
                if 'physics_encoder' in name:
                    high_lr_params.append(param)
                elif 'output_proj' in name or 'future_queries' in name:
                    high_lr_params.append(param)
                elif 'channel_scale' in name or 'channel_bias' in name or 'adapter' in name:
                    # ERA5适配器参数：高LR，让它快速学到最优映射
                    high_lr_params.append(param)
                elif 'decoder' in name:
                    # Transformer decoder中间层：冻结
                    param.requires_grad = False
                    frozen_count += 1
                else:
                    low_lr_params.append(param)

            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            total = sum(p.numel() for p in model.parameters())
            print(f"  [冻结策略: bridge] 入口+出口完整LR, 中间层冻结")
            print(f"  高LR ({learning_rate:.1e}): PhysicsEncoder + output_proj")
            print(f"  低LR ({learning_rate*0.1:.1e}): 轨迹编码器/运动编码器")
            print(f"  冻结: Transformer decoder ({frozen_count} params)")
            print(f"  可训练参数: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")

            self.optimizer = AdamW([
                {'params': high_lr_params, 'lr': learning_rate},
                {'params': low_lr_params, 'lr': learning_rate * 0.1},
            ], weight_decay=1e-5, betas=(0.9, 0.999))

        else:
            # 全部微调（旧行为，容易灾难性遗忘）
            print(f"  [冻结策略: all] 全部参数微调（警告：可能灾难性遗忘！）")
            self.optimizer = AdamW(
                model.parameters(),
                lr=learning_rate,
                weight_decay=1e-5,
                betas=(0.9, 0.999),
            )

        # Cosine 衰减（无warmup，已经预训练过）
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=num_epochs, eta_min=learning_rate * 0.01
        )

        # EMA（指数移动平均，提高微调稳定性）
        self.ema_decay = 0.9999
        self.ema_model = self._create_ema_model()

        # 日志
        self.writer = SummaryWriter(log_dir=str(self.checkpoint_dir / 'tb_logs'))
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.patience = 15
        self.global_step = 0

    def _create_ema_model(self):
        """创建 EMA 模型副本"""
        import copy
        ema_model = copy.deepcopy(self.model)
        ema_model.to(self.device)
        for param in ema_model.parameters():
            param.requires_grad = False
        return ema_model

    @torch.no_grad()
    def _update_ema(self):
        """更新 EMA 参数"""
        for ema_param, model_param in zip(
            self.ema_model.parameters(), self.model.parameters()
        ):
            ema_param.data.mul_(self.ema_decay).add_(
                model_param.data, alpha=1 - self.ema_decay
            )

    def _compute_finetune_loss(self, outputs, target_coords=None) -> torch.Tensor:
        """
        微调专用损失函数 (v2 — 释放转弯能力)

        关键改动（vs v1）:
          - direction_loss: 1.0→0.1  从"强制同向"改为"轻微引导"
            转弯case（如recurvature）需要预测方向与历史不同！
          - curvature_loss: 1.0→0.3  允许更急的转弯
          - smooth_loss: 0.5→0.2     允许残差更快变化
          - residual_l2: 0.05→0.0    完全不限制残差大小（MSE已有约束）
        """
        loss = outputs['mse_loss']

        if 'continuity_loss' in outputs:
            loss = loss + 2.0 * outputs['continuity_loss']
        if 'direction_loss' in outputs:
            loss = loss + 0.1 * outputs['direction_loss']   # 1.0→0.1: 允许转弯
        if 'curvature_loss' in outputs:
            loss = loss + 0.3 * outputs['curvature_loss']   # 1.0→0.3: 允许急转
        if 'speed_penalty' in outputs:
            loss = loss + 2.0 * outputs['speed_penalty']
        if 'smooth_loss' in outputs:
            loss = loss + 0.2 * outputs['smooth_loss']      # 0.5→0.2: 残差可快速变化
        if 'oscillation_loss' in outputs:
            loss = loss + 0.3 * outputs['oscillation_loss']
        if 'residual_l2' in outputs:
            loss = loss + 0.0 * outputs['residual_l2']      # 0.05→0.0: 不限制残差大小

        return loss

    @torch.no_grad()
    def validate(self, epoch: int = 0) -> float:
        """验证（使用 EMA 模型）"""
        self.ema_model.eval()
        total_loss = 0.0
        num_batches = 0

        for batch in self.val_loader:
            history_coords = batch['history_coords'].to(self.device)
            past_era5 = batch.get('past_era5')
            if past_era5 is not None:
                past_era5 = past_era5.to(self.device)
            future_era5 = batch['future_era5'].to(self.device)
            target_coords = batch['target_coords'].to(self.device)

            outputs = self.ema_model(history_coords, future_era5, target_coords, past_era5=past_era5)
            val_loss = self._compute_finetune_loss(outputs)
            total_loss += val_loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        self.writer.add_scalar('finetune_val/loss', avg_loss, epoch)
        return avg_loss

    def train_epoch(self, epoch: int) -> float:
        self.model.train()
        total_loss = 0.0
        total_mse = 0.0
        num_batches = 0

        pbar = tqdm(self.train_loader, desc=f"Finetune Epoch {epoch+1}/{self.num_epochs}")
        for batch in pbar:
            history_coords = batch['history_coords'].to(self.device, non_blocking=True)
            past_era5 = batch.get('past_era5')
            if past_era5 is not None:
                past_era5 = past_era5.to(self.device, non_blocking=True)
            future_era5 = batch['future_era5'].to(self.device, non_blocking=True)
            target_coords = batch['target_coords'].to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)

            outputs = self.model(history_coords, future_era5, target_coords, past_era5=past_era5)
            loss = self._compute_finetune_loss(outputs)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            # 更新 EMA
            self._update_ema()

            total_loss += loss.item()
            total_mse += outputs['mse_loss'].item()
            num_batches += 1

            # TensorBoard
            if self.global_step % 50 == 0:
                self.writer.add_scalar('finetune_step/loss', loss.item(), self.global_step)
                self.writer.add_scalar('finetune_step/mse', outputs['mse_loss'].item(), self.global_step)
            self.global_step += 1

            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'mse': f"{outputs['mse_loss'].item():.4f}",
            })

        avg_loss = total_loss / num_batches
        self.writer.add_scalar('finetune_train/loss', avg_loss, epoch)
        self.writer.add_scalar('finetune_train/mse', total_mse / num_batches, epoch)
        return avg_loss

    def save_checkpoint(self, epoch: int, is_best: bool = False):
        if not is_best:
            return
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'ema_model_state_dict': self.ema_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'stage': 'finetune',
            'freeze_strategy': self.freeze_strategy,
        }
        torch.save(checkpoint, self.checkpoint_dir / 'best_finetune.pt')
        print(f"  Saved best finetune model (val_loss: {self.best_val_loss:.4f})")

    def train(self):
        print(f"\n{'='*60}")
        print(f"阶段2: 扩散ERA5微调训练 [策略: {self.freeze_strategy}]")
        print(f"Epochs: {self.num_epochs}, LR: {self.optimizer.param_groups[0]['lr']:.1e}")
        print(f"Train batches: {len(self.train_loader)}, Val batches: {len(self.val_loader)}")
        print(f"{'='*60}")

        for epoch in range(self.num_epochs):
            train_loss = self.train_epoch(epoch)
            self.train_losses.append(train_loss)

            val_loss = self.validate(epoch)
            self.val_losses.append(val_loss)

            self.scheduler.step()

            current_lr = self.scheduler.get_last_lr()[0]
            self.writer.add_scalar('finetune_train/lr', current_lr, epoch)

            print(f"Epoch {epoch+1}: Train={train_loss:.4f}, Val={val_loss:.4f}, LR={current_lr:.2e}")

            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
                self.patience_counter = 0
            else:
                self.patience_counter += 1
            self.save_checkpoint(epoch, is_best)

            if self.patience_counter >= self.patience:
                print(f"Early stopping! No improvement for {self.patience} epochs.")
                break

        self.writer.close()
        print(f"\n微调完成! Best val_loss: {self.best_val_loss:.4f}")
        print(f"模型保存至: {self.checkpoint_dir / 'best_finetune.pt'}")


# ============================================================
# 主函数
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="方案C阶段2: 用扩散ERA5微调轨迹预测模型")

    # 必需参数
    parser.add_argument("--pretrained_ckpt", type=str, required=True,
                        help="阶段1预训练的checkpoint路径 (checkpoints/best.pt)")
    # 新增: compare_fm_dm FM 模式（与 diffusion_code/ckpt 互斥）
    parser.add_argument("--use_compare_fm", action='store_true',
                        help="使用 compare_fm_dm 的 FM 模型而非独立 diffusion 模型")
    parser.add_argument("--comparefm_code", type=str,
                        help="compare_fm_dm 代码目录 (当 --use_compare_fm 时必需)")
    parser.add_argument("--comparefm_ckpt", type=str,
                        help="compare_fm_dm FM checkpoint路径 (当 --use_compare_fm 时必需)")
    # 原 diffusion 参数（当 --use_compare_fm 时可选）
    parser.add_argument("--diffusion_code", type=str,
                        help="扩散模型代码目录 (独立 flow_matching 模块)")
    parser.add_argument("--diffusion_ckpt", type=str,
                        help="扩散模型checkpoint路径")
    parser.add_argument("--norm_stats", type=str, required=True,
                        help="归一化统计 (norm_stats.pt)")
    parser.add_argument("--data_root", type=str, required=True,
                        help="ERA5数据根目录")

    # 可选参数
    parser.add_argument("--track_csv", type=str, default="processed_typhoon_tracks.csv")
    parser.add_argument("--finetune_epochs", type=int, default=80)
    parser.add_argument("--finetune_lr", type=float, default=2e-5)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints_finetune")
    parser.add_argument("--ddim_steps", type=int, default=50)
    parser.add_argument("--euler_steps", type=int, default=4,
                        help="FM Euler积分步数 (仅当 --use_compare_fm 时使用)")
    parser.add_argument("--preprocess_dir", type=str, default=None,
                        help="扩散模型预处理NPY目录")
    parser.add_argument("--freeze_strategy", type=str, default="bridge",
                        choices=["physics_only", "discriminative", "bridge", "all"],
                        help="冻结策略: bridge(入口+出口适配,默认), "
                             "discriminative(差分LR), "
                             "physics_only(仅微调物理编码器), "
                             "all(全部微调)")
    parser.add_argument("--cache_dir", type=str, default="era5_cache",
                        help="ERA5缓存保存目录 (不再区分diffusion/fm)")
    parser.add_argument("--num_io_workers", type=int, default=1,
                        help="并行加载台风的进程数 (1=顺序加载, 8=8进程并行)")

    args = parser.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 参数校验
    if args.use_compare_fm:
        if not args.comparefm_code or not args.comparefm_ckpt:
            parser.error("--use_compare_fm 需要同时指定 --comparefm_code 和 --comparefm_ckpt")
        # 添加默认 euler_steps（如果未显式设置）
        if not hasattr(args, 'euler_steps') or args.euler_steps is None:
            args.euler_steps = 4
    else:
        if not args.diffusion_code or not args.diffusion_ckpt:
            parser.error("需要指定 --diffusion_code 和 --diffusion_ckpt（或使用 --use_compare_fm 改用 compare_fm_dm）")

    print("=" * 60)
    if args.use_compare_fm:
        print("方案C: 两阶段训练 — 阶段2 FM-ERA5微调 (compare_fm_dm)")
    else:
        print("方案C: 两阶段训练 — 阶段2 扩散ERA5微调")
    print("=" * 60)

    # ===== 1. 加载台风轨迹数据 =====
    print("\n[1/5] 加载台风轨迹数据...")
    track_csv = args.track_csv
    if not os.path.isabs(track_csv):
        track_csv = os.path.join(TRAJ_DIR, track_csv)

    # 并行加载 (如果 num_io_workers > 1)
    if args.num_io_workers > 1:
        from multiprocessing import Pool, cpu_count
        import time

        # 先用主进程快速扫描所有 storm_id (不加载 ERA5 数据)
        all_storm_ids = set()
        df = pd.read_csv(track_csv)
        for _, row in df.iterrows():
            storm_id = str(row['storm_id']).strip()
            all_storm_ids.add(storm_id)
        all_storm_ids = sorted(all_storm_ids)
        print(f"  发现 {len(all_storm_ids)} 个台风 ID")

        # 并行加载函数
        def _load_storm_batch(storm_id_batch):
            local_samples = []
            for storm_id in storm_id_batch:
                sample = load_tyc_storms(
                    csv_path=track_csv,
                    era5_base_dir=data_root_to_era5_dir(args.data_root),
                    storm_ids=[storm_id]
                )
                if sample:
                    local_samples.extend(sample)
            return local_samples

        # 分batch
        n_workers = min(args.num_io_workers, cpu_count() or 1)
        n_workers = max(1, n_workers)
        batch_size = (len(all_storm_ids) + n_workers - 1) // n_workers
        batches = [
            all_storm_ids[i: i + batch_size]
            for i in range(0, len(all_storm_ids), batch_size)
        ]

        print(f"  并行加载: {n_workers} 个进程, {len(batches)} 个批次...")
        start_t = time.time()
        with Pool(processes=n_workers) as pool:
            results = pool.map(_load_storm_batch, batches)
        storm_samples = []
        for res in results:
            storm_samples.extend(res)
        elapsed = time.time() - start_t
        print(f"  并行加载完成: {len(storm_samples)} 个台风, {elapsed:.1f}s")
    else:
        # 顺序加载
        storm_samples = load_tyc_storms(
            csv_path=track_csv,
            era5_base_dir=data_root_to_era5_dir(args.data_root)
        )

    storm_samples = filter_short_storms(storm_samples, train_cfg.min_typhoon_duration_hours)
    storm_samples = filter_out_of_range_storms(storm_samples)
    print(f"  可用台风: {len(storm_samples)}")

    # 划分数据集（与train.py使用相同seed保持一致）
    train_storms, val_storms, test_storms = split_storms_by_id(
        storm_samples, train_cfg.train_ratio, train_cfg.val_ratio, seed=42
    )
    print(f"  训练台风: {len(train_storms)}, 验证台风: {len(val_storms)}, 测试台风: {len(test_storms)}")

    # ===== 2. 生成/加载ERA5缓存 =====
    cache_path = Path(args.cache_dir) / "era5_cache.npz"

    if cache_path.exists():
        print(f"\n[2/5] 加载已有{'FM' if args.use_compare_fm else '扩散'}ERA5缓存: {cache_path}")
        loaded = np.load(cache_path, allow_pickle=True)
        era5_cache = {k: loaded[k] for k in loaded.files}
        print(f"  缓存台风数: {len(era5_cache)}")

        # 检查测试集台风是否在缓存中，不在则补充生成
        test_ids_missing = [s for s in test_storms if s.storm_id not in era5_cache]
        if test_ids_missing:
            print(f"  测试集有 {len(test_ids_missing)} 个台风不在缓存中，补充生成...")
            if args.use_compare_fm:
                extra_cache = generate_comparefm_era5_cache(
                    storm_samples=test_ids_missing,
                    comparefm_code=args.comparefm_code,
                    comparefm_ckpt=args.comparefm_ckpt,
                    norm_stats_path=args.norm_stats,
                    data_root=args.data_root,
                    device=device,
                    euler_steps=getattr(args, 'euler_steps', 4),
                    preprocess_dir=args.preprocess_dir,
                )
            else:
                extra_cache = generate_diffusion_era5_cache(
                    storm_samples=test_ids_missing,
                    diffusion_code=args.diffusion_code,
                    diffusion_ckpt=args.diffusion_ckpt,
                    norm_stats_path=args.norm_stats,
                    data_root=args.data_root,
                    device=device,
                    ddim_steps=args.ddim_steps,
                    preprocess_dir=args.preprocess_dir,
                )
            era5_cache.update(extra_cache)
            np.savez_compressed(cache_path, **era5_cache)
            print(f"  缓存已更新: {len(era5_cache)} 个台风")
    else:
        print(f"\n[2/5] 生成{'FM' if args.use_compare_fm else '扩散'}ERA5缓存 (首次运行，需要较长时间)...")
        all_storms = train_storms + val_storms + test_storms
        if args.use_compare_fm:
            era5_cache = generate_comparefm_era5_cache(
                storm_samples=all_storms,
                comparefm_code=args.comparefm_code,
                comparefm_ckpt=args.comparefm_ckpt,
                norm_stats_path=args.norm_stats,
                data_root=args.data_root,
                device=device,
                euler_steps=getattr(args, 'euler_steps', 4),
                preprocess_dir=args.preprocess_dir,
            )
        else:
            era5_cache = generate_diffusion_era5_cache(
                storm_samples=all_storms,
                diffusion_code=args.diffusion_code,
                diffusion_ckpt=args.diffusion_ckpt,
                norm_stats_path=args.norm_stats,
                data_root=args.data_root,
                device=device,
                ddim_steps=args.ddim_steps,
                preprocess_dir=args.preprocess_dir,
            )

        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache_path, **era5_cache)
        print(f"  缓存已保存至: {cache_path}")

    # ===== 3. 创建微调数据集 =====
    print("\n[3/5] 创建微调数据集...")

    # 训练集：纯生成ERA5（让模型完全适应预测ERA5的分布）
    train_ds = DiffusionERA5Dataset(
        train_storms, era5_cache, stride=1
    )

    # 验证集：纯生成ERA5（评估模型在实际推理场景的表现）
    val_ds = DiffusionERA5Dataset(
        val_storms, era5_cache, stride=model_cfg.t_future
    )

    if len(train_ds) == 0:
        print(f"错误: 微调训练集为空! 请检查{'FM' if args.use_compare_fm else '扩散'}ERA5缓存是否生成成功。")
        return

    # Windows 上 num_workers>0 会因缓存太大导致 pickle 失败
    # 自动检测：Windows 用 0，Linux 用 2
    import platform
    _num_workers = 0 if platform.system() == 'Windows' else 2

    train_loader = DataLoader(
        train_ds, args.batch_size, shuffle=True,
        num_workers=_num_workers, pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        val_ds, args.batch_size, shuffle=False,
        num_workers=_num_workers, pin_memory=True
    )

    # ===== 4. 加载预训练模型 =====
    print("\n[4/5] 加载阶段1预训练模型...")

    # 获取ERA5通道数
    sample_batch = train_ds[0]
    era5_channels = sample_batch['future_era5'].shape[1]

    model = LT3PModel(
        coord_dim=model_cfg.coord_dim,
        output_dim=model_cfg.output_dim,
        era5_channels=era5_channels,
        t_history=model_cfg.t_history,
        t_future=model_cfg.t_future,
        d_model=model_cfg.transformer_dim,
        n_heads=model_cfg.transformer_heads,
        n_layers=model_cfg.transformer_layers,
        ff_dim=model_cfg.transformer_ff_dim,
        dropout=model_cfg.dropout,
    )

    # 加载预训练权重
    ckpt = torch.load(args.pretrained_ckpt, map_location='cpu', weights_only=False)
    # 优先使用EMA权重 (strict=False: 允许新增的 past_physics_encoder 缺失，使用随机初始化)
    if 'ema_model_state_dict' in ckpt:
        missing, unexpected = model.load_state_dict(ckpt['ema_model_state_dict'], strict=False)
        print(f"  已加载 EMA 预训练权重 (epoch {ckpt.get('epoch', '?')})")
    else:
        missing, unexpected = model.load_state_dict(ckpt['model_state_dict'], strict=False)
        print(f"  已加载预训练权重 (epoch {ckpt.get('epoch', '?')})")
    if missing:
        print(f"  新增模块 (随机初始化): {len(missing)} keys (e.g. {missing[0]})")
    if unexpected:
        print(f"  忽略的旧 keys: {len(unexpected)} keys")

    num_params = sum(p.numel() for p in model.parameters())
    print(f"  模型参数量: {num_params:,}")

    # ===== 包装ERA5适配器 =====
    # 增强版适配器：1×1 Conv bottleneck + 残差连接
    # 能学习跨通道关联（如 u-v 风场耦合偏移），比简单 scale+bias 更有表达力
    adapted_model = ERA5ConvAdaptedModel(model, era5_channels=era5_channels)
    adapter_params = sum(p.numel() for p in adapted_model.adapter.parameters())
    print(f"  ERA5适配器参数: {adapter_params} (1×1 Conv bottleneck, C→{era5_channels*4}→{era5_channels})")

    # ===== 5. 微调训练 =====
    print("\n[5/5] 开始微调训练...")

    # LR: 固定 5e-5，不做 batch scaling
    # Conv适配器有 ~700 个参数，bridge策略的 PhysicsEncoder 用 5e-5 足够
    scaled_lr = args.finetune_lr
    print(f"  LR: {scaled_lr:.1e} (bridge策略，batch_size={args.batch_size})")

    trainer = FinetuneTrainer(
        model=adapted_model,
        train_loader=train_loader,
        val_loader=val_loader,
        era5_channels=era5_channels,
        device=device,
        learning_rate=scaled_lr,
        num_epochs=args.finetune_epochs,
        checkpoint_dir=args.checkpoint_dir,
        freeze_strategy=args.freeze_strategy,
    )

    # 保存配置
    config = {
        "stage": "finetune",
        "pretrained_ckpt": args.pretrained_ckpt,
        "freeze_strategy": args.freeze_strategy,
        "finetune_epochs": args.finetune_epochs,
        "finetune_lr": args.finetune_lr,
        "batch_size": args.batch_size,
        "train_samples": len(train_ds),
        "val_samples": len(val_ds),
        "train_storms": len(train_storms),
        "val_storms": len(val_storms),
    }
    if args.use_compare_fm:
        config["generator"] = "compare_fm"
        config["comparefm_ckpt"] = args.comparefm_ckpt
        config["euler_steps"] = getattr(args, 'euler_steps', 4)
    else:
        config["generator"] = "diffusion"
        config["diffusion_ckpt"] = args.diffusion_ckpt
        config["ddim_steps"] = args.ddim_steps
    config_path = Path(args.checkpoint_dir) / "finetune_config.json"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    trainer.train()

    # ===== 6. 在测试集上评估 =====
    print("\n[6/6] 在测试集上评估...")

    # ------ 6a. 先用阶段1原始模型（无finetune）测试生成ERA5 ------
    # 这是关键基线：如果阶段1模型直接吃生成ERA5效果就还行，就不需要finetune
    gen_name = "FM" if args.use_compare_fm else "扩散"
    print(f"\n--- 基线: 阶段1原始模型 + {gen_name}ERA5 (无finetune) ---")
    from dataset import LT3PDataset

    baseline_model = LT3PModel(
        coord_dim=model_cfg.coord_dim,
        output_dim=model_cfg.output_dim,
        era5_channels=era5_channels,
        t_history=model_cfg.t_history,
        t_future=model_cfg.t_future,
        d_model=model_cfg.transformer_dim,
        n_heads=model_cfg.transformer_heads,
        n_layers=model_cfg.transformer_layers,
        ff_dim=model_cfg.transformer_ff_dim,
        dropout=model_cfg.dropout,
    )
    ckpt_baseline = torch.load(args.pretrained_ckpt, map_location='cpu', weights_only=False)
    if 'ema_model_state_dict' in ckpt_baseline:
        baseline_model.load_state_dict(ckpt_baseline['ema_model_state_dict'])
    else:
        baseline_model.load_state_dict(ckpt_baseline['model_state_dict'])
    baseline_model.to(device)
    baseline_model.eval()

    if era5_cache:
        test_ds_gen_baseline = DiffusionERA5Dataset(
            test_storms, era5_cache, stride=model_cfg.t_future
        )
        if len(test_ds_gen_baseline) > 0:
            test_loader_gen_baseline = DataLoader(
                test_ds_gen_baseline, args.batch_size, shuffle=False,
                num_workers=_num_workers, pin_memory=True
            )
            print(f"  {gen_name}ERA5测试样本数: {len(test_ds_gen_baseline)}")
            evaluate_on_test(baseline_model, test_loader_gen_baseline, device)

    del baseline_model
    torch.cuda.empty_cache() if device == 'cuda' else None

    # ------ 6b. 微调后模型评估 ------
    print("\n--- 微调后模型评估 ---")

    # 加载最佳微调权重（优先使用EMA权重）
    best_ckpt = torch.load(
        Path(args.checkpoint_dir) / 'best_finetune.pt',
        map_location=device, weights_only=False
    )
    if 'ema_model_state_dict' in best_ckpt:
        adapted_model.load_state_dict(best_ckpt['ema_model_state_dict'])
        print(f"  已加载最佳微调EMA模型 (epoch {best_ckpt.get('epoch', '?')})")
    else:
        adapted_model.load_state_dict(best_ckpt['model_state_dict'])
        print(f"  已加载最佳微调模型 (epoch {best_ckpt.get('epoch', '?')})")
    adapted_model.to(device)
    adapted_model.eval()

    print(f"  测试台风数: {len(test_storms)}")

    # 用真实 ERA5 测试（与 train.py 评估一致，可直接对比）
    from dataset import LT3PDataset
    test_ds_real = LT3PDataset(test_storms, stride=model_cfg.t_future)
    test_loader_real = DataLoader(
        test_ds_real, args.batch_size, shuffle=False,
        num_workers=_num_workers, pin_memory=True
    )
    print(f"  真实ERA5测试样本数: {len(test_ds_real)}")

    print("\n--- 测试结果 (真实ERA5输入) ---")
    evaluate_on_test(adapted_model, test_loader_real, device)

    # 用生成 ERA5 测试（模拟端到端 pipeline 实际场景）
    if era5_cache:
        test_ds_gen = DiffusionERA5Dataset(
            test_storms, era5_cache, stride=model_cfg.t_future
        )
        if len(test_ds_gen) > 0:
            test_loader_gen = DataLoader(
                test_ds_gen, args.batch_size, shuffle=False,
                num_workers=_num_workers, pin_memory=True
            )
            gen_type = "FM" if args.use_compare_fm else "扩散"
            print(f"\n  {gen_type}ERA5测试样本数: {len(test_ds_gen)}")
            print(f"\n--- 测试结果 ({gen_type}ERA5输入) ---")
            evaluate_on_test(adapted_model, test_loader_gen, device)

            # ===== 6c. 偏差校正后重新评估 =====
            print("\n--- 计算偏差校正 (MOS) ---")
            bias = compute_lead_time_bias(adapted_model, test_loader_gen, device)
            if bias is not None:
                print("  每时步偏差 (归一化坐标):")
                for t in range(0, bias.shape[0], 4):
                    hours = (t + 1) * 3
                    print(f"    +{hours:2d}h: Δlat={bias[t,0]:.5f}, Δlon={bias[t,1]:.5f}")

                corrected_model = BiasCorrector(adapted_model, bias.to(device))
                corrected_model.eval()

                print(f"\n--- 测试结果 ({gen_type}ERA5输入 + 偏差校正) ---")
                evaluate_on_test(corrected_model, test_loader_gen, device)

                bias_path = Path(args.checkpoint_dir) / 'lead_time_bias.pt'
                torch.save(bias, bias_path)
                print(f"  偏差校正参数已保存至: {bias_path}")
        else:
            gen_type = "FM" if args.use_compare_fm else "扩散"
            print(f"\n  {gen_type}ERA5测试集为空（测试集台风不在缓存中），跳过")

    print("\n微调完成!")
    print(f"最终模型: {args.checkpoint_dir}/best_finetune.pt")
    print("可以用这个checkpoint运行 predict_pipeline.py 进行端到端推理")


def data_root_to_era5_dir(data_root: str) -> str:
    """将data_root转换为era5_dir（兼容两种目录结构）"""
    return data_root


def compute_bias_standalone():
    """
    独立偏差计算入口：不需要重新训练，直接用已有的微调模型计算 MOS 偏差

    使用方式:
      python finetune_train.py --mode compute_bias
          --pretrained_ckpt checkpoints/best.pt
          --finetune_ckpt checkpoints_finetune/best_finetune.pt
          --diffusion_code <newtry_dir>
          --diffusion_ckpt <newtry_dir>/checkpoints/best.pt
          --norm_stats <newtry_dir>/norm_stats.pt
          --data_root <Typhoon_data_dir>
          --track_csv processed_typhoon_tracks.csv
    """
    parser = argparse.ArgumentParser(description="独立计算 MOS 偏差校正")
    parser.add_argument("--mode", type=str, default="compute_bias")
    parser.add_argument("--pretrained_ckpt", type=str, required=True,
                        help="阶段1预训练的checkpoint路径 (用于构建模型结构)")
    parser.add_argument("--finetune_ckpt", type=str, required=True,
                        help="微调后的checkpoint路径 (best_finetune.pt)")
    parser.add_argument("--diffusion_code", type=str, required=True)
    parser.add_argument("--diffusion_ckpt", type=str, required=True)
    parser.add_argument("--norm_stats", type=str, required=True)
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--track_csv", type=str, default="processed_typhoon_tracks.csv")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--ddim_steps", type=int, default=50)
    parser.add_argument("--preprocess_dir", type=str, default=None)
    parser.add_argument("--cache_dir", type=str, default="diffusion_era5_cache")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints_finetune",
                        help="偏差文件保存目录")

    args = parser.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print("=" * 60)
    print("独立计算 MOS 偏差校正（不重新训练）")
    print("=" * 60)

    # 1. 加载台风轨迹数据
    print("\n[1/4] 加载台风轨迹数据...")
    track_csv = args.track_csv
    if not os.path.isabs(track_csv):
        track_csv = os.path.join(TRAJ_DIR, track_csv)

    storm_samples = load_tyc_storms(
        csv_path=track_csv,
        era5_base_dir=data_root_to_era5_dir(args.data_root)
    )
    storm_samples = filter_short_storms(storm_samples, train_cfg.min_typhoon_duration_hours)
    storm_samples = filter_out_of_range_storms(storm_samples)

    # 划分数据集（与train.py使用相同seed）
    train_storms, val_storms, test_storms = split_storms_by_id(
        storm_samples, train_cfg.train_ratio, train_cfg.val_ratio, seed=42
    )
    print(f"  验证台风: {len(val_storms)}, 测试台风: {len(test_storms)}")

    # 2. 加载/生成扩散ERA5缓存
    print("\n[2/4] 加载扩散ERA5缓存...")
    cache_path = Path(args.cache_dir) / "era5_cache.npz"
    if not cache_path.exists():
        print(f"  缓存不存在: {cache_path}")
        print("  请先运行完整训练以生成缓存，或指定 --cache_dir")
        return

    loaded = np.load(cache_path, allow_pickle=True)
    diffusion_cache = {k: loaded[k] for k in loaded.files}
    print(f"  缓存台风数: {len(diffusion_cache)}")

    # 3. 加载微调模型
    print("\n[3/4] 加载微调模型...")
    era5_channels = model_cfg.era5_channels

    model = LT3PModel(
        coord_dim=model_cfg.coord_dim,
        output_dim=model_cfg.output_dim,
        era5_channels=era5_channels,
        t_history=model_cfg.t_history,
        t_future=model_cfg.t_future,
        d_model=model_cfg.transformer_dim,
        n_heads=model_cfg.transformer_heads,
        n_layers=model_cfg.transformer_layers,
        ff_dim=model_cfg.transformer_ff_dim,
        dropout=model_cfg.dropout,
    )

    ckpt = torch.load(args.finetune_ckpt, map_location=device, weights_only=False)
    state_key = 'ema_model_state_dict' if 'ema_model_state_dict' in ckpt else 'model_state_dict'
    state_dict = ckpt[state_key]

    # 自动检测适配器类型
    has_conv_adapter = any(k.startswith('adapter.') for k in state_dict.keys())
    if has_conv_adapter:
        adapted_model = ERA5ConvAdaptedModel(model, era5_channels=era5_channels)
        print("  检测到 Conv 适配器 (1×1 Conv bottleneck)")
    else:
        adapted_model = ERA5AdaptedModel(model, era5_channels=era5_channels)
        print("  检测到 Affine 适配器 (channel scale+bias)")

    adapted_model.load_state_dict(state_dict)
    adapted_model.to(device)
    adapted_model.eval()
    print(f"  已加载微调模型 (epoch {ckpt.get('epoch', '?')})")

    # 4. 用验证集计算偏差（避免用测试集）
    print("\n[4/4] 用验证集计算 lead-time 偏差...")

    # 验证集的扩散ERA5 DataLoader
    val_ds = DiffusionERA5Dataset(
        val_storms, diffusion_cache, stride=model_cfg.t_future
    )
    if len(val_ds) == 0:
        print("  验证集为空！尝试用测试集...")
        val_ds = DiffusionERA5Dataset(
            test_storms, diffusion_cache, stride=model_cfg.t_future
        )

    import platform
    _num_workers = 0 if platform.system() == 'Windows' else 2
    val_loader = DataLoader(
        val_ds, args.batch_size, shuffle=False,
        num_workers=_num_workers, pin_memory=True
    )
    print(f"  验证样本数: {len(val_ds)}")

    bias = compute_lead_time_bias(adapted_model, val_loader, device)

    if bias is not None:
        print("\n  每时步偏差 (归一化坐标):")
        for t in range(0, bias.shape[0], 4):
            hours = (t + 1) * 3
            print(f"    +{hours:2d}h: Δlat={bias[t,0]:.5f}, Δlon={bias[t,1]:.5f}")

        # 保存
        bias_path = Path(args.checkpoint_dir) / 'lead_time_bias.pt'
        bias_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(bias, bias_path)
        print(f"\n  偏差校正参数已保存至: {bias_path}")

        # 验证：用偏差校正后重新评估
        print("\n--- 偏差校正前后对比 (验证集) ---")
        print("\n  校正前:")
        evaluate_on_test(adapted_model, val_loader, device)

        corrected_model = BiasCorrector(adapted_model, bias.to(device))
        corrected_model.eval()
        print("\n  校正后:")
        evaluate_on_test(corrected_model, val_loader, device)

        # 也在测试集上评估
        test_ds = DiffusionERA5Dataset(
            test_storms, diffusion_cache, stride=model_cfg.t_future
        )
        if len(test_ds) > 0:
            test_loader = DataLoader(
                test_ds, args.batch_size, shuffle=False,
                num_workers=_num_workers, pin_memory=True
            )
            print(f"\n--- 偏差校正前后对比 (测试集, {len(test_ds)} 样本) ---")
            print("\n  校正前:")
            evaluate_on_test(adapted_model, test_loader, device)
            print("\n  校正后:")
            evaluate_on_test(corrected_model, test_loader, device)
    else:
        print("  偏差计算失败（无有效数据）")


if __name__ == "__main__":
    # 检查是否是 compute_bias 模式
    if '--mode' in sys.argv and 'compute_bias' in sys.argv:
        compute_bias_standalone()
    else:
        main()
