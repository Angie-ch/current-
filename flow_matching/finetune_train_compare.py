"""
Finetune script for trajectory prediction using FM-generated ERA5 from compare_fm_dm.

This script is based on Trajectory/finetune_train.py but adapted to use
the compare_fm_dm Flow Matching model (UnifiedModel) for ERA5 generation.

Key differences from Trajectory/finetune_train.py:
  - Uses compare_fm_dm's UnifiedModel (FM mode) instead of Diffusion model
  - Uses Euler ODE solver (1-10 steps) instead of DDIM (50 steps)
  - Cache generation is ~50x faster
  - Requires compare_fm_dm codebase and its trained FM checkpoint

Usage:
  python finetune_train_compare.py \\
      --pretrained_ckpt /path/to/trajectory/checkpoints/best.pt \\
      --comparefm_code /root/autodl-tmp/fyp_final/Ver4/compare_fm_dm \\
      --comparefm_ckpt /root/autodl-tmp/fyp_final/Ver4/compare_fm_dm/multi_seed_results/seed_42/checkpoints_fm/best.pt \\
      --norm_stats /path/to/norm_stats.pt \\
      --data_root /path/to/Typhoon_data_final \\
      --track_csv /path/to/processed_typhoon_tracks.csv \\
      --freeze_strategy bridge \\
      --finetune_epochs 80
"""
import os
import sys
import json
import argparse
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import pandas as pd

# ============================================================
# Paths and imports
# ============================================================
TRAJ_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, TRAJ_DIR)

# Trajectory imports (model, dataset, config, train)
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
# Channel configuration (same as Trajectory and newtry)
# ============================================================
# 9-channel configuration: [u_850, u_500, u_250, v_850, v_500, v_250, z_850, z_500, z_250]
ERA5_CHANNELS = 9

# ============================================================
# Dataset: Diff ERA5 (same as flow_matching/finetune_train.py)
# ============================================================

class DiffERA5Dataset(Dataset):
    """
    Finetuning dataset: real trajectory, synthetic ERA5 from FM model.

    Identical to DiffusionERA5Dataset in Trajectory/finetune_train.py
    but works with any generative model cache.
    """

    def __init__(
        self,
        storm_samples: List[StormSample],
        fm_era5_cache: Dict[str, np.ndarray],
        t_history: int = None,
        t_future: int = None,
        stride: int = 1,
        era5_channels: int = None,
    ):
        self.storm_samples = storm_samples
        self.fm_era5_cache = fm_era5_cache
        self.t_history = t_history or model_cfg.t_history
        self.t_future = t_future or model_cfg.t_future
        self.stride = stride
        self.era5_channels = era5_channels or model_cfg.era5_channels
        self.total_length = self.t_history + self.t_future

        self.valid_samples = [
            s for s in storm_samples
            if s.storm_id in fm_era5_cache
        ]

        self.samples_index = self._build_index()
        print(f"DiffERA5Dataset (FM): {len(self.valid_samples)} storms, "
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

        # History trajectory (real)
        history_lat = sample.track_lat[history_start:history_end]
        history_lon = sample.track_lon[history_start:history_end]
        h_lat_n, h_lon_n = normalize_coords(history_lat, history_lon)
        history_coords = np.stack([h_lat_n, h_lon_n], axis=-1)

        # Past ERA5 (real, from sample if available)
        past_era5 = np.zeros((self.t_history, self.era5_channels, 40, 40), dtype=np.float32)
        if sample.era5_array is not None:
            raw = sample.era5_array[history_start:history_end]
            if raw.shape[1] > self.era5_channels:
                raw = raw[:, :self.era5_channels]
            past_era5[:len(raw)] = raw[:len(past_era5)]
        past_era5 = normalize_era5(past_era5)

        # Future ERA5 (FM-generated, from cache)
        fm_era5 = self.fm_era5_cache[sample.storm_id]
        future_era5 = fm_era5[future_start:future_end]

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

        future_era5 = normalize_era5(future_era5)

        # Future trajectory target (real)
        future_lat = sample.track_lat[future_start:future_end]
        future_lon = sample.track_lon[future_start:future_end]
        f_lat_n, f_lon_n = normalize_coords(future_lat, future_lon)
        target_coords = np.stack([f_lat_n, f_lon_n], axis=-1)

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
# Compare FM (UnifiedModel) cache generator
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
    Generate FM-ERA5 cache using compare_fm_dm's UnifiedModel (FM mode).

    This is a bridge that loads a trained compare_fm_dm FM model and
    runs autoregressive inference to produce the full-length ERA5 forecast
    cache needed for trajectory finetuning.

    Args:
        storm_samples: List of StormSample objects (with real tracks)
        comparefm_code: Path to compare_fm_dm code directory
        comparefm_ckpt: Path to compare_fm_dm FM checkpoint (best.pt)
        norm_stats_path: Path to normalization statistics (.pt)
        data_root: Path to ERA5 data root
        device: 'cuda' or 'cpu'
        euler_steps: Number of Euler integration steps (4 typical)
        preprocess_dir: Optional preprocessed data directory

    Returns:
        cache: {storm_id: (T_storm, 9, 40, 40)} normalized-space predictions
    """
    print("\n" + "=" * 60)
    print("Generating compare_fm_dm FM ERA5 cache (Euler sampling)...")
    print("=" * 60)

    # Import compare_fm_dm modules (handle module conflicts)
    conflicting_modules = ['train', 'configs', 'models', 'inference', 'data', 'data.dataset']
    saved_modules = {}
    for mod_name in conflicting_modules:
        if mod_name in sys.modules:
            saved_modules[mod_name] = sys.modules.pop(mod_name)

    if comparefm_code not in sys.path:
        sys.path.insert(0, comparefm_code)
    elif sys.path[0] != comparefm_code:
        sys.path.remove(comparefm_code)
        sys.path.insert(0, comparefm_code)

    from configs import get_config as cfm_get_config
    from models import UnifiedModel
    from train import EMA as CFMEMA
    # compare_fm_dm might not have a dedicated inference module like flow_matching
    # We'll use model.sample_fm directly.
    from data.dataset import ERA5TyphoonDataset

    # Restore trajectory modules
    cfm_modules = {}
    for mod_name in conflicting_modules:
        if mod_name in sys.modules:
            cfm_modules[mod_name] = sys.modules[mod_name]
    for mod_name, mod_obj in saved_modules.items():
        sys.modules[mod_name] = mod_obj

    # Load compare_fm_dm config
    cfm_data_cfg, cfm_model_cfg, cfm_train_cfg, cfm_infer_cfg = cfm_get_config(
        data_root=data_root
    )

    # Load normalization stats
    stats = torch.load(norm_stats_path, weights_only=True, map_location='cpu')
    norm_mean = stats['mean'].numpy()
    norm_std = stats['std'].numpy()

    # Load compare_fm_dm FM model
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

    # Inference settings
    cfm_infer_cfg.euler_steps = euler_steps
    # We'll use cfm_model.sample_fm directly

    cfm_num_channels = cfm_data_cfg.num_channels  # should be 9
    traj_num_channels = 9
    mean_t = torch.from_numpy(norm_mean[:cfm_num_channels]).float().to(device).reshape(1, -1, 1, 1)
    std_t = torch.from_numpy(norm_std[:cfm_num_channels]).float().to(device).reshape(1, -1, 1, 1)
    std_t = torch.where(std_t < 1e-8, torch.ones_like(std_t), std_t)

    storm_ids = list({s.storm_id for s in storm_samples})
    num_ar_steps = 24  # forecast steps (72h / 3h)
    ensemble_size = 1
    ar_noise_sigma = 0.02

    print(f"  台风数: {len(storm_ids)}")
    print(f"  策略: 每 {num_ar_steps} 步重启自回归，集合均值 {ensemble_size} 成员，Euler步数={euler_steps}")

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

    # Build storm → sample index mapping
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

    cache = {}
    storm_meta = {}

    # ========== Build task list ==========
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
            pred_start = cond_start + cfm_data_cfg.history_steps
            if pred_start >= next_needed:
                selected.append((ds_idx, pred_start))
                next_needed = pred_start + num_ar_steps

        if len(selected) == 0:
            ds_idx, cond_start = available[0]
            selected.append((ds_idx, cond_start + cfm_data_cfg.history_steps))

        for ds_idx, pred_start in selected:
            remaining = storm_len - pred_start
            steps = min(num_ar_steps, max(remaining, 0))
            if steps > 0:
                all_tasks.append((sid, ds_idx, pred_start, steps))

    print(f"  总推理任务数: {len(all_tasks)}")

    # ========== Batch inference ==========
    BATCH_SIZE = 4  # compare_fm_dm models might be larger; use smaller batch
    task_results = [None] * len(all_tasks)

    for batch_start in tqdm(range(0, len(all_tasks), BATCH_SIZE), desc="批量生成FM ERA5"):
        batch_tasks = all_tasks[batch_start: batch_start + BATCH_SIZE]
        batch_steps = max(t[3] for t in batch_tasks)

        # Collect conditions
        conds = []
        for sid, ds_idx, pred_start, steps in batch_tasks:
            sample = full_dataset[ds_idx]
            conds.append(sample['condition'])  # (T*C, H, W)

        cond_batch = torch.stack(conds, dim=0).to(device)  # (B, T*C, H, W)

        with torch.no_grad():
            # Autoregressive sampling for each item in batch
            batch_preds = []
            for i in range(len(batch_tasks)):
                cond_i = cond_batch[i:i+1].clone()  # (1, T*C, H, W) - clone to avoid modifying batch
                preds_i = []

                for step in range(batch_steps):
                    # sample_fm takes condition (1, T*C, H, W) and returns x0 prediction (1, C, H, W)
                    # Use clamp range from model's scheduler or default
                    clamp_range = getattr(cfm_model, 'clamp_range', (-5.0, 5.0))
                    pred = cfm_model.sample_fm(
                        cond_i,
                        device=device,
                        euler_steps=euler_steps,
                        euler_mode='midpoint',
                        clamp_range=clamp_range,
                    )
                    preds_i.append(pred)

                    # Update condition for next step if not last
                    if step < batch_steps - 1:
                        # cond_i shape: (1, T*C, H, W). Need to roll and append pred.
                        C = cfm_num_channels
                        T = cfm_data_cfg.history_steps
                        cond_5d = cond_i.view(1, T, C, cfm_data_cfg.grid_size, cfm_data_cfg.grid_size)
                        # Remove first time step, append new prediction at end
                        cond_5d = torch.cat([cond_5d[:, 1:], pred.unsqueeze(1)], dim=1)
                        cond_i = cond_5d.view(1, -1, cfm_data_cfg.grid_size, cfm_data_cfg.grid_size)

                batch_preds.append(preds_i)

        # Assign results to tasks
        for i, (sid, ds_idx, pred_start, steps) in enumerate(batch_tasks):
            preds_i = batch_preds[i]
            task_results[batch_start + i] = (sid, pred_start, preds_i)

    # ========== Assemble cache ==========
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
        for k, p_tensor in enumerate(preds_i):
            t_idx = pred_start + k
            if 0 <= t_idx < storm_len and not filled[t_idx]:
                p = p_tensor[:, :cfm_num_channels] if p_tensor.shape[1] > cfm_num_channels else p_tensor
                p_phys = p * std_t + mean_t
                p_np = p_phys.cpu().numpy()[0]  # (C, H, W)
                era5_cache[t_idx] = p_np
                filled[t_idx] = True

    # ========== Fill gaps (linear interpolation) ==========
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

    print(f"\nFM ERA5缓存完成: {len(cache)}/{len(storm_ids)} 个台风")
    return cache


# ============================================================
# ERA5 Adapter (same as finetune_train.py)
# ============================================================

class ERA5AdaptedModel(nn.Module):
    """Channel-wise affine adapter (simple, ~18 params)"""

    def __init__(self, base_model: nn.Module, era5_channels: int = 9):
        super().__init__()
        self.base_model = base_model
        self.channel_scale = nn.Parameter(torch.ones(1, 1, era5_channels, 1, 1))
        self.channel_bias = nn.Parameter(torch.zeros(1, 1, era5_channels, 1, 1))

    def forward(self, history_coords, future_era5, target_coords=None, past_era5=None):
        era5_adapted = future_era5 * self.channel_scale + self.channel_bias
        return self.base_model(history_coords, era5_adapted, target_coords, past_era5=past_era5)

    def predict(self, history_coords, future_era5, past_era5=None):
        era5_adapted = future_era5 * self.channel_scale + self.channel_bias
        return self.base_model.predict(history_coords, era5_adapted, past_era5=past_era5)


class ERA5ConvAdaptedModel(nn.Module):
    """1×1 Conv bottleneck adapter (~700 params, more expressive)"""

    def __init__(self, base_model: nn.Module, era5_channels: int = 9):
        super().__init__()
        self.base_model = base_model
        self.era5_channels = era5_channels

        hidden_dim = era5_channels * 4
        self.adapter = nn.Sequential(
            nn.Conv2d(era5_channels, hidden_dim, kernel_size=1, bias=True),
            nn.GELU(),
            nn.Conv2d(hidden_dim, era5_channels, kernel_size=1, bias=True),
        )

        nn.init.zeros_(self.adapter[-1].weight)
        nn.init.zeros_(self.adapter[-1].bias)
        nn.init.xavier_uniform_(self.adapter[0].weight, gain=0.1)
        nn.init.zeros_(self.adapter[0].bias)

    def _adapt_era5(self, future_era5):
        B, T, C, H, W = future_era5.shape
        x = future_era5.reshape(B * T, C, H, W)
        x = x + self.adapter(x)
        return x.reshape(B, T, C, H, W)

    def forward(self, history_coords, future_era5, target_coords=None, past_era5=None):
        era5_adapted = self._adapt_era5(future_era5)
        return self.base_model(history_coords, era5_adapted, target_coords, past_era5=past_era5)

    def predict(self, history_coords, future_era5, past_era5=None):
        era5_adapted = self._adapt_era5(future_era5)
        return self.base_model.predict(history_coords, era5_adapted, past_era5=past_era5)


# ============================================================
# Post-hoc bias corrector
# ============================================================

class BiasCorrector(nn.Module):
    """Lead-time bias corrector (MOS)"""

    def __init__(self, base_model: nn.Module, bias: torch.Tensor):
        super().__init__()
        self.base_model = base_model
        self.register_buffer('bias', bias)  # (T_future, 2)

    def predict(self, history_coords, future_era5, past_era5=None):
        outputs = self.base_model.predict(history_coords, future_era5, past_era5=past_era5)
        corrected = outputs['predicted_coords'] - self.bias.unsqueeze(0)
        corrected = corrected.clamp(0.0, 1.0)
        outputs['predicted_coords'] = corrected
        return outputs


@torch.no_grad()
def compute_lead_time_bias(model, dataloader, device):
    """Compute systematic bias per lead time"""
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

    errors = (all_pred - all_target)
    sample_mean_err = errors.abs().mean(dim=(1, 2))
    threshold = torch.quantile(sample_mean_err, 0.95)
    mask = sample_mean_err <= threshold

    bias = errors[mask].mean(dim=0)
    print(f"  偏差校正: 使用 {mask.sum()}/{len(mask)} 个样本 (排除top 5%离群)")
    print(f"  最大偏差: {bias.abs().max():.5f} (归一化坐标)")

    return bias


# ============================================================
# Finetune Trainer (identical to Trajectory/finetune_train.py)
# ============================================================

class FinetuneTrainer:
    """
    Stage 2 finetuning trainer.

    Freeze strategy options:
      - physics_only: only finetune PhysicsEncoder3D + adapter
      - discriminative: PhysicsEncoder + adapter at full LR, others at 0.1× LR
      - bridge: PhysicsEncoder + output_proj + adapter full LR, decoder frozen
      - all: finetune everything (not recommended)
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

        # Selective freezing
        if freeze_strategy == 'physics_only':
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
            high_lr_params = []
            low_lr_params = []
            frozen_count = 0
            for name, param in model.named_parameters():
                if 'physics_encoder' in name:
                    high_lr_params.append(param)
                elif 'output_proj' in name or 'future_queries' in name:
                    high_lr_params.append(param)
                elif 'channel_scale' in name or 'channel_bias' in name or 'adapter' in name:
                    high_lr_params.append(param)
                elif 'decoder' in name:
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
            print(f"  [冻结策略: all] 全部参数微调（警告：可能灾难性遗忘！）")
            self.optimizer = AdamW(
                model.parameters(),
                lr=learning_rate,
                weight_decay=1e-5,
                betas=(0.9, 0.999),
            )

        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=num_epochs, eta_min=learning_rate * 0.01
        )

        self.ema_decay = 0.9999
        self.ema_model = self._create_ema_model()

        self.writer = SummaryWriter(log_dir=str(self.checkpoint_dir / 'tb_logs'))
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.patience = 15
        self.global_step = 0

    def _create_ema_model(self):
        import copy
        ema_model = copy.deepcopy(self.model)
        ema_model.to(self.device)
        for param in ema_model.parameters():
            param.requires_grad = False
        return ema_model

    @torch.no_grad()
    def _update_ema(self):
        for ema_param, model_param in zip(
            self.ema_model.parameters(), self.model.parameters()
        ):
            ema_param.data.mul_(self.ema_decay).add_(
                model_param.data, alpha=1 - self.ema_decay
            )

    def _compute_finetune_loss(self, outputs, target_coords=None) -> torch.Tensor:
        """Finetune loss (v2 — release turning capability)"""
        loss = outputs['mse_loss']

        if 'continuity_loss' in outputs:
            loss = loss + 2.0 * outputs['continuity_loss']
        if 'direction_loss' in outputs:
            loss = loss + 0.1 * outputs['direction_loss']   # allow turning
        if 'curvature_loss' in outputs:
            loss = loss + 0.3 * outputs['curvature_loss']
        if 'speed_penalty' in outputs:
            loss = loss + 2.0 * outputs['speed_penalty']
        if 'smooth_loss' in outputs:
            loss = loss + 0.2 * outputs['smooth_loss']
        if 'oscillation_loss' in outputs:
            loss = loss + 0.3 * outputs['oscillation_loss']
        if 'residual_l2' in outputs:
            loss = loss + 0.0 * outputs['residual_l2']

        return loss

    @torch.no_grad()
    def validate(self, epoch: int = 0) -> float:
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

            self._update_ema()

            total_loss += loss.item()
            total_mse += outputs['mse_loss'].item()
            num_batches += 1

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
            'stage': 'finetune_comparefm',
            'freeze_strategy': self.freeze_strategy,
        }
        torch.save(checkpoint, self.checkpoint_dir / 'best_finetune_comparefm.pt')
        print(f"  Saved best compareFM finetune model (val_loss: {self.best_val_loss:.4f})")

    def train(self):
        print(f"\n{'='*60}")
        print(f"阶段2: compare_fm_dm FM-ERA5微调训练 [策略: {self.freeze_strategy}]")
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
        print(f"模型保存至: {self.checkpoint_dir / 'best_finetune_comparefm.pt'}")


# ============================================================
# Bias computation (same as Trajectory/finetune_train.py)
# ============================================================

@torch.no_grad()
def compute_lead_time_bias(model, dataloader, device):
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

    all_pred = torch.cat(all_pred, dim=0)
    all_target = torch.cat(all_target, dim=0)

    errors = (all_pred - all_target)
    sample_mean_err = errors.abs().mean(dim=(1, 2))
    threshold = torch.quantile(sample_mean_err, 0.95)
    mask = sample_mean_err <= threshold

    bias = errors[mask].mean(dim=0)
    print(f"  偏差校正: 使用 {mask.sum()}/{len(mask)} 个样本 (排除top 5%离群)")
    print(f"  最大偏差: {bias.abs().max():.5f} (归一化坐标)")

    return bias


class BiasCorrector(nn.Module):
    def __init__(self, base_model: nn.Module, bias: torch.Tensor):
        super().__init__()
        self.base_model = base_model
        self.register_buffer('bias', bias)

    def predict(self, history_coords, future_era5, past_era5=None):
        outputs = self.base_model.predict(history_coords, future_era5, past_era5=past_era5)
        corrected = outputs['predicted_coords'] - self.bias.unsqueeze(0)
        corrected = corrected.clamp(0.0, 1.0)
        outputs['predicted_coords'] = corrected
        return outputs


def data_root_to_era5_dir(data_root: str) -> str:
    return data_root


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Finetune trajectory predictor using compare_fm_dm FM-generated ERA5"
    )

    # Required args
    parser.add_argument("--pretrained_ckpt", type=str, required=True,
                        help="Stage 1 pretrained trajectory checkpoint (best.pt)")
    parser.add_argument("--comparefm_code", type=str, required=True,
                        help="Path to compare_fm_dm code directory")
    parser.add_argument("--comparefm_ckpt", type=str, required=True,
                        help="compare_fm_dm FM checkpoint (best.pt)")
    parser.add_argument("--norm_stats", type=str, required=True,
                        help="Normalization stats (.pt file)")
    parser.add_argument("--data_root", type=str, required=True,
                        help="ERA5 data root directory")

    # Optional args (same as Trajectory/finetune_train.py)
    parser.add_argument("--track_csv", type=str, default="processed_typhoon_tracks.csv")
    parser.add_argument("--finetune_epochs", type=int, default=80)
    parser.add_argument("--finetune_lr", type=float, default=2e-5)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints_finetune_comparefm")
    parser.add_argument("--euler_steps", type=int, default=4,
                        help="Euler integration steps for FM sampling")
    parser.add_argument("--preprocess_dir", type=str, default=None)
    parser.add_argument("--freeze_strategy", type=str, default="bridge",
                        choices=["physics_only", "discriminative", "bridge", "all"])
    parser.add_argument("--cache_dir", type=str, default="comparefm_era5_cache",
                        help="FM ERA5 cache directory")

    args = parser.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print("=" * 60)
    print("Finetune Trajectory Model with compare_fm_dm FM-ERA5")
    print("=" * 60)

    # 1. Load typhoon track data
    print("\n[1/5] 加载台风轨迹数据...")
    track_csv = args.track_csv
    if not os.path.isabs(track_csv):
        track_csv = os.path.join(TRAJ_DIR, track_csv)

    storm_samples = load_tyc_storms(
        csv_path=track_csv,
        era5_base_dir=data_root_to_era5_dir(args.data_root)
    )
    storm_samples = filter_short_storms(storm_samples, train_cfg.min_typhoon_duration_hours)
    storm_samples = filter_out_of_range_storms(storm_samples)
    print(f"  可用台风: {len(storm_samples)}")

    train_storms, val_storms, test_storms = split_storms_by_id(
        storm_samples, train_cfg.train_ratio, train_cfg.val_ratio, seed=42
    )
    print(f"  训练台风: {len(train_storms)}, 验证台风: {len(val_storms)}, 测试台风: {len(test_storms)}")

    # 2. Generate/load FM ERA5 cache
    cache_path = Path(args.cache_dir) / "era5_cache.npz"

    if cache_path.exists():
        print(f"\n[2/5] 加���已有 FM ERA5缓存: {cache_path}")
        loaded = np.load(cache_path, allow_pickle=True)
        fm_cache = {k: loaded[k] for k in loaded.files}
        print(f"  缓存台风数: {len(fm_cache)}")

        test_ids_missing = [s for s in test_storms if s.storm_id not in fm_cache]
        if test_ids_missing:
            print(f"  测试集有 {len(test_ids_missing)} 个台风不在缓存中，补充生成...")
            extra_cache = generate_comparefm_era5_cache(
                storm_samples=test_ids_missing,
                comparefm_code=args.comparefm_code,
                comparefm_ckpt=args.comparefm_ckpt,
                norm_stats_path=args.norm_stats,
                data_root=args.data_root,
                device=device,
                euler_steps=args.euler_steps,
                preprocess_dir=args.preprocess_dir,
            )
            fm_cache.update(extra_cache)
            np.savez_compressed(cache_path, **fm_cache)
            print(f"  缓存已更新: {len(fm_cache)} 个台风")
    else:
        print(f"\n[2/5] 生成 FM ERA5缓存 (首次运行，需要较长时间)...")
        all_storms = train_storms + val_storms + test_storms
        fm_cache = generate_comparefm_era5_cache(
            storm_samples=all_storms,
            comparefm_code=args.comparefm_code,
            comparefm_ckpt=args.comparefm_ckpt,
            norm_stats_path=args.norm_stats,
            data_root=args.data_root,
            device=device,
            euler_steps=args.euler_steps,
            preprocess_dir=args.preprocess_dir,
        )

        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache_path, **fm_cache)
        print(f"  缓存已保存至: {cache_path}")

    # 3. Create finetuning datasets
    print("\n[3/5] 创建微调数据集...")
    train_ds = DiffERA5Dataset(train_storms, fm_cache, stride=1)
    val_ds = DiffERA5Dataset(val_storms, fm_cache, stride=model_cfg.t_future)

    if len(train_ds) == 0:
        print("错误: 微调训练集为空! 请检查 FM ERA5缓存是否生成成功。")
        return

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

    # 4. Load pretrained trajectory model
    print("\n[4/5] 加载阶段1预训练模型...")
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

    ckpt = torch.load(args.pretrained_ckpt, map_location='cpu', weights_only=False)
    if 'ema_model_state_dict' in ckpt:
        missing, unexpected = model.load_state_dict(ckpt['ema_model_state_dict'], strict=False)
        print(f"  已加载 EMA 预训练权重 (epoch {ckpt.get('epoch', '?')})")
    else:
        missing, unexpected = model.load_state_dict(ckpt['model_state_dict'], strict=False)
        print(f"  已加载预训练权重 (epoch {ckpt.get('epoch', '?')})")
    if missing:
        print(f"  新增模块 (随机初始化): {len(missing)} keys")
    if unexpected:
        print(f"  忽略的旧 keys: {len(unexpected)} keys")

    num_params = sum(p.numel() for p in model.parameters())
    print(f"  模型参数量: {num_params:,}")

    # Wrap with adapter
    adapted_model = ERA5ConvAdaptedModel(model, era5_channels=era5_channels)
    adapter_params = sum(p.numel() for p in adapted_model.adapter.parameters())
    print(f"  ERA5适配器参数: {adapter_params} (1×1 Conv bottleneck)")

    # 5. Finetune
    print("\n[5/5] 开始 FM-ERA5 微调训练...")
    scaled_lr = args.finetune_lr
    print(f"  LR: {scaled_lr:.1e}, 冻结策略: {args.freeze_strategy}")

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

    config = {
        "stage": "finetune_comparefm",
        "pretrained_ckpt": args.pretrained_ckpt,
        "freeze_strategy": args.freeze_strategy,
        "finetune_epochs": args.finetune_epochs,
        "finetune_lr": args.finetune_lr,
        "batch_size": args.batch_size,
        "train_samples": len(train_ds),
        "val_samples": len(val_ds),
        "train_storms": len(train_storms),
        "val_storms": len(val_storms),
        "comparefm_ckpt": args.comparefm_ckpt,
        "euler_steps": args.euler_steps,
    }
    config_path = Path(args.checkpoint_dir) / "finetune_config.json"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    trainer.train()

    # 6. Evaluate
    print("\n[6/6] 在测试集上评估...")
    print("\n--- 微调后模型评估 ---")

    best_ckpt = torch.load(
        Path(args.checkpoint_dir) / 'best_finetune_comparefm.pt',
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

    from dataset import LT3PDataset
    test_ds_real = LT3PDataset(test_storms, stride=model_cfg.t_future)
    test_loader_real = DataLoader(
        test_ds_real, args.batch_size, shuffle=False,
        num_workers=_num_workers, pin_memory=True
    )
    print(f"  真实 ERA5 测试样本数: {len(test_ds_real)}")
    print("\n--- 测试结果 (真实 ERA5 输入) ---")
    evaluate_on_test(adapted_model, test_loader_real, device)

    if fm_cache:
        test_ds_diff = DiffERA5Dataset(test_storms, fm_cache, stride=model_cfg.t_future)
        if len(test_ds_diff) > 0:
            test_loader_diff = DataLoader(
                test_ds_diff, args.batch_size, shuffle=False,
                num_workers=_num_workers, pin_memory=True
            )
            print(f"\n  扩散ERA5测试样本数: {len(test_ds_diff)}")
            print("\n--- 测试结果 (FM ERA5 输入) ---")
            evaluate_on_test(adapted_model, test_loader_diff, device)

            # Bias correction
            print("\n--- 计算偏差校正 (MOS) ---")
            bias = compute_lead_time_bias(adapted_model, test_loader_diff, device)
            if bias is not None:
                print("  每时步偏差 (归一化坐标):")
                for t in range(0, bias.shape[0], 4):
                    hours = (t + 1) * 3
                    print(f"    +{hours:2d}h: Δlat={bias[t,0]:.5f}, Δlon={bias[t,1]:.5f}")

                corrected_model = BiasCorrector(adapted_model, bias.to(device))
                corrected_model.eval()

                print("\n--- 测试结果 (FM ERA5输入 + 偏差校正) ---")
                evaluate_on_test(corrected_model, test_loader_diff, device)

                bias_path = Path(args.checkpoint_dir) / 'lead_time_bias.pt'
                torch.save(bias, bias_path)
                print(f"  偏差校正参数已保存至: {bias_path}")
        else:
            print("\n  FM ERA5测试集为空（测试集台风不在缓存中），跳过")

    print("\nFinetune完成!")
    print(f"最终模型: {args.checkpoint_dir}/best_finetune_comparefm.pt")


if __name__ == "__main__":
    main()
