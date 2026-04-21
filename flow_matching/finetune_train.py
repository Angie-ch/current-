"""
Flow Matching 微调脚本 — 用 CFM 预测的 ERA5 微调轨迹预测模型

核心优势（vs Diffusion 版本 finetune_train.py）:
  - 缓存生成速度: ~50倍提速
    DDIM: 1464台风 × 50步 × 1000步扩散 = ~10小时
    CFM:  1464台风 × 1步(Euler) = ~15分钟
  - 物理一致性: 最优传输路径保证生成场平滑且物理连贯
  - 确定性映射: FM 是确定性 ODE，同一噪声种子 → 唯一的天气场映射

使用方式:
  python finetune_train.py \\
      --pretrained_ckpt checkpoints/best.pt \\
      --cfm_code /path/to/flow_matching \\
      --cfm_ckpt /path/to/flow_matching/checkpoints/best.pt \\
      --norm_stats /path/to/flow_matching/norm_stats.pt \\
      --data_root /path/to/Typhoon_data_final
"""
import os
import sys
import json
import argparse
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import pandas as pd

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
# 通道说明：与 newtry 完全一致（9通道）
# ============================================================
# [0:3]  u_850, u_500, u_250
# [3:6]  v_850, v_500, v_250
# [6:9]  z_850, z_500, z_250
# ============================================================


# ============================================================
# CFM-ERA5 数据集（与 Diffusion 版本接口完全一致）
# ============================================================

class DiffERA5Dataset(Dataset):
    """
    微调数据集：用 CFM 预测的 ERA5 替换真实 ERA5

    与 DiffusionERA5Dataset 接口完全一致，
    唯一区别是 ERA5 来源从扩散模型变为 CFM 流匹配模型
    """

    def __init__(
        self,
        storm_samples: List[StormSample],
        cfm_era5_cache: Dict[str, np.ndarray],
        t_history: int = None,
        t_future: int = None,
        stride: int = 1,
        era5_channels: int = None,
    ):
        self.storm_samples = storm_samples
        self.cfm_era5_cache = cfm_era5_cache
        self.t_history = t_history or model_cfg.t_history
        self.t_future = t_future or model_cfg.t_future
        self.stride = stride
        self.era5_channels = era5_channels or model_cfg.era5_channels
        self.total_length = self.t_history + self.t_future

        self.valid_samples = [
            s for s in storm_samples
            if s.storm_id in cfm_era5_cache
        ]

        self.samples_index = self._build_index()
        print(f"DiffERA5Dataset (CFM): {len(self.valid_samples)} storms, "
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

        history_lat = sample.track_lat[history_start:history_end]
        history_lon = sample.track_lon[history_start:history_end]
        h_lat_n, h_lon_n = normalize_coords(history_lat, history_lon)
        history_coords = np.stack([h_lat_n, h_lon_n], axis=-1)

        cfm_era5 = self.cfm_era5_cache[sample.storm_id]
        future_era5 = cfm_era5[future_start:future_end]

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
# CFM ERA5 缓存生成（核心优化：~50倍提速）
# ============================================================

def generate_cfm_era5_cache(
    storm_samples: List[StormSample],
    cfm_code: str,
    cfm_ckpt: str,
    norm_stats_path: str,
    data_root: str,
    device: str = 'cuda',
    euler_steps: int = 1,
    preprocess_dir: str = None,
) -> Dict[str, np.ndarray]:
    """
    用 CFM 模型为每个台风生成完整时间线的 ERA5 预测缓存

    核心优势:
      - Euler 1步采样，50倍速于 DDIM 50步
      - 最优传输路径，方差最低
      - 确定性采样，同一噪声种子 → 唯一映射

    性能对比:
      DDIM: ~10小时 (1464台风)
      CFM:  ~15分钟 (1464台风)
    """
    print("\n" + "=" * 60)
    print("生成 CFM ERA5 预测缓存 (Euler 1步采样)...")
    print("=" * 60)

    conflicting_modules = ['train', 'configs', 'models', 'inference', 'data', 'data.dataset']
    saved_modules = {}
    for mod_name in conflicting_modules:
        if mod_name in sys.modules:
            saved_modules[mod_name] = sys.modules.pop(mod_name)

    if cfm_code not in sys.path:
        sys.path.insert(0, cfm_code)
    elif sys.path[0] != cfm_code:
        sys.path.remove(cfm_code)
        sys.path.insert(0, cfm_code)

    from configs import get_config as cfm_get_config
    from models import ERA5FlowMatchingModel
    from train import EMA as CFMEMA
    from inference import EulerCFMTranslator
    from data.dataset import ERA5TyphoonDataset

    cfm_modules = {}
    for mod_name in conflicting_modules:
        if mod_name in sys.modules:
            cfm_modules[mod_name] = sys.modules[mod_name]
    for mod_name, mod_obj in saved_modules.items():
        sys.modules[mod_name] = mod_obj

    cfm_data_cfg, cfm_model_cfg, _, cfm_infer_cfg = cfm_get_config(data_root=data_root)
    cfm_history_steps = cfm_data_cfg.history_steps

    stats = torch.load(norm_stats_path, weights_only=True, map_location='cpu')
    norm_mean = stats['mean'].numpy()
    norm_std = stats['std'].numpy()

    cfm_model = ERA5FlowMatchingModel(cfm_model_cfg, cfm_data_cfg).to(device)
    ckpt = torch.load(cfm_ckpt, map_location=device, weights_only=False)

    if 'ema_state_dict' in ckpt:
        ema = CFMEMA(cfm_model, decay=0.9999)
        ema.load_state_dict(ckpt['ema_state_dict'])
        ema.apply_shadow(cfm_model)
        print("  已加载 CFM 模型 EMA 参数")
    else:
        cfm_model.load_state_dict(ckpt['model_state_dict'])
    cfm_model.eval()

    translator = EulerCFMTranslator(cfm_model, cfm_data_cfg)
    translator.euler_steps = euler_steps
    print(f"  CFM 采样步数: {euler_steps} (Euler)")

    cfm_num_channels = cfm_data_cfg.num_channels
    traj_num_channels = 9
    mean_t = torch.from_numpy(norm_mean[:cfm_num_channels]).float().to(device).reshape(1, -1, 1, 1)
    std_t = torch.from_numpy(norm_std[:cfm_num_channels]).float().to(device).reshape(1, -1, 1, 1)
    std_t = torch.where(std_t < 1e-8, torch.ones_like(std_t), std_t)

    storm_ids = list({s.storm_id for s in storm_samples})
    num_ar_steps = 24
    ensemble_size = 1
    ar_noise_sigma = 0.02

    print(f"  台风数: {len(storm_ids)}")
    print(f"  策略: 每 {num_ar_steps} 步重启自回归，集合均值 {ensemble_size} 成员，噪声 σ={ar_noise_sigma}")

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
        print(f"  创建 ERA5 数据集失败: {e}")
        return {}

    print(f"  ERA5 数据集: {len(full_dataset)} 个样本")

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

    for sid in tqdm(storm_ids, desc="生成CFM ERA5"):
        storm_obj = next((s for s in storm_samples if s.storm_id == sid), None)
        if storm_obj is None:
            continue

        storm_len = len(storm_obj)
        if sid not in tid_to_samples or len(tid_to_samples[sid]) == 0:
            print(f"  跳过 {sid}: 无可用条件样本")
            continue

        era5_cache = np.zeros((storm_len, traj_num_channels, 40, 40), dtype=np.float32)
        filled = np.zeros(storm_len, dtype=bool)

        available = tid_to_samples[sid]
        selected: List[Tuple[int, int]] = []
        next_needed = 0

        for ds_idx, cond_start in available:
            pred_start = cond_start + cfm_history_steps
            if pred_start >= next_needed:
                selected.append((ds_idx, pred_start))
                next_needed = pred_start + num_ar_steps

        if len(selected) == 0:
            ds_idx, cond_start = available[0]
            selected.append((ds_idx, cond_start + cfm_history_steps))

        for ds_idx, pred_start in selected:
            remaining = storm_len - pred_start
            steps = min(num_ar_steps, max(remaining, 0))
            if steps <= 0:
                continue

            sample = full_dataset[ds_idx]
            cond = sample['condition'].unsqueeze(0).to(device)

            with torch.no_grad():
                preds = translator.predict_autoregressive(
                    cond,
                    num_steps=steps,
                    noise_sigma=ar_noise_sigma,
                    ensemble_per_step=ensemble_size,
                )

            for k, pred in enumerate(preds):
                t_idx = pred_start + k
                if t_idx < 0 or t_idx >= storm_len:
                    continue
                if filled[t_idx]:
                    continue

                p = pred[:, :cfm_num_channels] if pred.shape[1] > cfm_num_channels else pred
                p_phys = p * std_t + mean_t
                p_np = p_phys.cpu().numpy()[0]

                era5_cache[t_idx] = p_np
                filled[t_idx] = True

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
                elif t == storm_len - 1:
                    for gap_t in range(last_valid + 1, storm_len):
                        era5_cache[gap_t] = era5_cache[last_valid]

            if not filled[storm_len - 1]:
                for gap_t in range(last_valid + 1, storm_len):
                    era5_cache[gap_t] = era5_cache[last_valid]

        cache[sid] = era5_cache

    print(f"\nCFM ERA5 缓存完成: {len(cache)}/{len(storm_ids)} 个台风")
    return cache


# ============================================================
# ERA5 适配器（与 finetune_train.py 完全一致）
# ============================================================

class ERA5ConvAdaptedModel(nn.Module):
    """1×1 Conv Bottleneck + 残差连接"""

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

    def forward(self, history_coords, future_era5, target_coords=None):
        era5_adapted = self._adapt_era5(future_era5)
        return self.base_model(history_coords, era5_adapted, target_coords)

    def predict(self, history_coords, future_era5):
        era5_adapted = self._adapt_era5(future_era5)
        return self.base_model.predict(history_coords, era5_adapted)


class FinetuneTrainer:
    """CFM-ERA5 微调训练器（与 finetune_train.py 完全一致的接口）"""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        era5_channels: int,
        device: str = 'cuda',
        learning_rate: float = 5e-5,
        num_epochs: int = 50,
        checkpoint_dir: str = 'checkpoints_finetune_cfm',
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
            print(f"  可训练参数: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")

            self.optimizer = AdamW([
                {'params': high_lr_params, 'lr': learning_rate},
                {'params': low_lr_params, 'lr': learning_rate * 0.1},
            ], weight_decay=1e-5, betas=(0.9, 0.999))

        else:
            print(f"  [冻结策略: all] 全部参数微调")
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

    def _compute_loss(self, outputs) -> torch.Tensor:
        loss = outputs['mse_loss']
        if 'continuity_loss' in outputs:
            loss = loss + 2.0 * outputs['continuity_loss']
        if 'direction_loss' in outputs:
            loss = loss + 1.0 * outputs['direction_loss']
        if 'curvature_loss' in outputs:
            loss = loss + 1.0 * outputs['curvature_loss']
        if 'speed_penalty' in outputs:
            loss = loss + 2.0 * outputs['speed_penalty']
        if 'smooth_loss' in outputs:
            loss = loss + 0.5 * outputs['smooth_loss']
        if 'oscillation_loss' in outputs:
            loss = loss + 0.3 * outputs['oscillation_loss']
        if 'residual_l2' in outputs:
            loss = loss + 0.05 * outputs['residual_l2']
        return loss

    @torch.no_grad()
    def validate(self, epoch: int = 0) -> float:
        self.ema_model.eval()
        total_loss = 0.0
        num_batches = 0

        for batch in self.val_loader:
            history_coords = batch['history_coords'].to(self.device)
            future_era5 = batch['future_era5'].to(self.device)
            target_coords = batch['target_coords'].to(self.device)

            outputs = self.ema_model(history_coords, future_era5, target_coords)
            val_loss = self._compute_loss(outputs)
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
            future_era5 = batch['future_era5'].to(self.device, non_blocking=True)
            target_coords = batch['target_coords'].to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)

            outputs = self.model(history_coords, future_era5, target_coords)
            loss = self._compute_loss(outputs)

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
            'stage': 'finetune_cfm',
            'freeze_strategy': self.freeze_strategy,
        }
        torch.save(checkpoint, self.checkpoint_dir / 'best_finetune_cfm.pt')
        print(f"  Saved best CFM finetune model (val_loss: {self.best_val_loss:.4f})")

    def train(self):
        print(f"\n{'='*60}")
        print(f"CFM-ERA5 微调训练 [策略: {self.freeze_strategy}]")
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
        print(f"\nCFM 微调完成! Best val_loss: {self.best_val_loss:.4f}")
        print(f"模型保存至: {self.checkpoint_dir / 'best_finetune_cfm.pt'}")


# ============================================================
# 主函数
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="CFM-ERA5 微调轨迹预测模型")

    parser.add_argument("--pretrained_ckpt", type=str, required=True,
                        help="阶段1预训练的 checkpoint 路径")
    parser.add_argument("--cfm_code", type=str, required=True,
                        help="CFM 模型代码目录")
    parser.add_argument("--cfm_ckpt", type=str, required=True,
                        help="CFM 模型 checkpoint")
    parser.add_argument("--norm_stats", type=str, required=True,
                        help="归一化统计文件 (norm_stats.pt)")
    parser.add_argument("--data_root", type=str, required=True,
                        help="ERA5 数据根目录")

    parser.add_argument("--track_csv", type=str, default="processed_typhoon_tracks.csv")
    parser.add_argument("--finetune_epochs", type=int, default=80)
    parser.add_argument("--finetune_lr", type=float, default=2e-5)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints_finetune_cfm")
    parser.add_argument("--euler_steps", type=int, default=1,
                        help="CFM Euler 采样步数 (推荐1)")
    parser.add_argument("--preprocess_dir", type=str, default=None)
    parser.add_argument("--freeze_strategy", type=str, default="bridge",
                        choices=["physics_only", "discriminative", "bridge", "all"])
    parser.add_argument("--cache_dir", type=str, default="cfm_era5_cache",
                        help="CFM ERA5 缓存保存目录")

    args = parser.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print("=" * 60)
    print("CFM-ERA5 微调轨迹预测模型")
    print("=" * 60)

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

    cache_path = Path(args.cache_dir) / "era5_cache.npz"

    if cache_path.exists():
        print(f"\n[2/5] 加载已有 CFM ERA5 缓存: {cache_path}")
        loaded = np.load(cache_path, allow_pickle=True)
        cfm_cache = {k: loaded[k] for k in loaded.files}
        print(f"  缓存台风数: {len(cfm_cache)}")
    else:
        print(f"\n[2/5] 生成 CFM ERA5 缓存 (Euler {args.euler_steps}步采样)...")
        all_storms = train_storms + val_storms + test_storms
        cfm_cache = generate_cfm_era5_cache(
            storm_samples=all_storms,
            cfm_code=args.cfm_code,
            cfm_ckpt=args.cfm_ckpt,
            norm_stats_path=args.norm_stats,
            data_root=args.data_root,
            device=device,
            euler_steps=args.euler_steps,
            preprocess_dir=args.preprocess_dir,
        )

        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache_path, **cfm_cache)
        print(f"  缓存已保存至: {cache_path}")

    print("\n[3/5] 创建微调数据集...")
    train_ds = DiffERA5Dataset(train_storms, cfm_cache, stride=1)
    val_ds = DiffERA5Dataset(val_storms, cfm_cache, stride=model_cfg.t_future)

    if len(train_ds) == 0:
        print("错误: 微调训练集为空! 请检查 CFM ERA5 缓存是否生成成功。")
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
        model.load_state_dict(ckpt['ema_model_state_dict'])
        print(f"  已加载 EMA 预训练权重 (epoch {ckpt.get('epoch', '?')})")
    else:
        model.load_state_dict(ckpt['model_state_dict'])
        print(f"  已加载预训练权重 (epoch {ckpt.get('epoch', '?')})")

    num_params = sum(p.numel() for p in model.parameters())
    print(f"  模型参数量: {num_params:,}")

    adapted_model = ERA5ConvAdaptedModel(model, era5_channels=era5_channels)

    print("\n[5/5] 开始 CFM-ERA5 微调训练...")
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
        "stage": "finetune_cfm",
        "pretrained_ckpt": args.pretrained_ckpt,
        "freeze_strategy": args.freeze_strategy,
        "finetune_epochs": args.finetune_epochs,
        "finetune_lr": args.finetune_lr,
        "batch_size": args.batch_size,
        "train_samples": len(train_ds),
        "val_samples": len(val_ds),
        "cfm_ckpt": args.cfm_ckpt,
        "euler_steps": args.euler_steps,
    }
    config_path = Path(args.checkpoint_dir) / "finetune_config.json"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    trainer.train()

    print("\n[6/6] 在测试集上评估...")
    print("\n--- 微调后模型评估 ---")

    best_ckpt = torch.load(
        Path(args.checkpoint_dir) / 'best_finetune_cfm.pt',
        map_location=device, weights_only=False
    )
    if 'ema_model_state_dict' in best_ckpt:
        adapted_model.load_state_dict(best_ckpt['ema_model_state_dict'])
    else:
        adapted_model.load_state_dict(best_ckpt['model_state_dict'])
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

    if cfm_cache:
        test_ds_diff = DiffERA5Dataset(test_storms, cfm_cache, stride=model_cfg.t_future)
        if len(test_ds_diff) > 0:
            test_loader_diff = DataLoader(
                test_ds_diff, args.batch_size, shuffle=False,
                num_workers=_num_workers, pin_memory=True
            )
            print(f"\n--- 测试结果 (CFM ERA5 输入) ---")
            evaluate_on_test(adapted_model, test_loader_diff, device)

    print("\nCFM 微调完成!")
    print(f"最终模型: {args.checkpoint_dir}/best_finetune_cfm.pt")


def data_root_to_era5_dir(data_root: str) -> str:
    return data_root


if __name__ == "__main__":
    main()
