# -*- coding: utf-8 -*-
#
# 混合训练 v3：基于阶段1最佳模型精调（非从零训练）
#
# 核心发现（v1/v2 的教训）:
#   从零混合训练 → 真实ERA5上的精度被牺牲（300→379km）
#   但真实ERA5才是模型能力的天花板，不应该被牺牲
#
# v3 策略:
#   1. 加载 train.py 训练好的最佳模型（真实ERA5=300km）
#   2. 用 90%真实+10%扩散 做少量epoch精调（30-50 epoch, 极小lr）
#   3. 只微调 PhysicsEncoder3D，冻结 Transformer 解码器
#   4. 目标：真实ERA5保持~300km，扩散ERA5从898km降到~500km
#
# 与 finetune_train.py 的区别：
#   - finetune 用100%扩散ERA5 → 灾难性遗忘（300→620km）
#   - v3 用90%真实+10%扩散 → 防止遗忘，同时适应扩散分布
#   - v3 冻结Transformer解码器 → 保护轨迹预测的核心能力
#
import os
import sys
import json
import copy
import argparse
import platform
from pathlib import Path
from typing import List, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

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

from finetune_train import (
    MixedERA5Dataset,
    DiffusionERA5Dataset,
    generate_diffusion_era5_cache,
    data_root_to_era5_dir,
)


# ============================================================
# 课程式混合数据集
# ============================================================

class CurriculumMixedDataset(MixedERA5Dataset):
    def set_real_ratio(self, new_ratio: float):
        self.real_ratio = max(0.0, min(1.0, new_ratio))


# ============================================================
# 精调训练器 v3
# ============================================================

class FineTuneTrainerV3:
    """
    v3 核心策略：在已训练好的模型上做保守精调

    冻结策略：
      - PhysicsEncoder3D: 可训练（lr=full）→ 适应扩散ERA5分布
      - TrajectoryEncoder/MotionEncoder/traj_proj: 可训练（lr=full/10）→ 轻微适应
      - TrajectoryPredictor (Transformer decoder): 冻结 → 保护轨迹预测能力
    """

    def __init__(
        self,
        model: nn.Module,
        train_dataset: CurriculumMixedDataset,
        val_loader: DataLoader,
        batch_size: int = 128,
        device: str = 'cuda',
        learning_rate: float = 2e-5,
        num_epochs: int = 50,
        checkpoint_dir: str = 'checkpoints_v3',
        warmup_epochs: int = 5,
        patience: int = 15,
        r_start: float = 0.8,
        r_end: float = 0.9,
        r_transition_epoch: int = 20,
        num_workers: int = 0,
    ):
        self.device = device
        self.model = model.to(self.device)
        self.train_dataset = train_dataset
        self.val_loader = val_loader
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.warmup_epochs = warmup_epochs
        self.num_workers = num_workers

        self.r_start = r_start
        self.r_end = r_end
        self.r_transition_epoch = r_transition_epoch

        # ===== 冻结策略 =====
        self._setup_freeze(learning_rate)

        # 学习率调度
        self.scheduler = self._create_scheduler()

        # 日志
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        self.global_step = 0
        self.writer = SummaryWriter(log_dir=str(self.checkpoint_dir / 'tb_logs'))

        # 早停
        self.patience = patience
        self.patience_counter = 0

        # 混合精度
        self.use_amp = train_cfg.use_amp and self.device == 'cuda'
        self.scaler = torch.amp.GradScaler('cuda') if self.use_amp else None

        # EMA
        self.ema_decay = 0.9999
        self.ema_model = self._init_ema()

    def _setup_freeze(self, lr: float):
        """
        分层冻结+差异化学习率

        PhysicsEncoder3D: lr (全速适应扩散ERA5)
        TrajectoryEncoder/MotionEncoder/traj_proj: lr/10 (轻微适应)
        TrajectoryPredictor (output_proj + decoder + queries + lead_time): 冻结
        """
        # 冻结 Transformer 解码器（轨迹预测的核心）
        frozen_count = 0
        trainable_count = 0

        for name, param in self.model.named_parameters():
            if 'trajectory_predictor' in name:
                param.requires_grad = False
                frozen_count += param.numel()
            else:
                param.requires_grad = True
                trainable_count += param.numel()

        print(f"  Frozen params: {frozen_count:,} (trajectory_predictor)")
        print(f"  Trainable params: {trainable_count:,}")

        # 差异化学习率
        physics_params = []
        other_params = []

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if 'physics_encoder' in name:
                physics_params.append(param)
            else:
                other_params.append(param)

        self.optimizer = AdamW([
            {'params': physics_params, 'lr': lr},
            {'params': other_params, 'lr': lr / 10},
        ], weight_decay=1e-5, betas=(0.9, 0.999))

        print(f"  PhysicsEncoder lr: {lr}")
        print(f"  Other trainable lr: {lr/10}")

    def _get_real_ratio(self, epoch: int) -> float:
        if epoch < self.r_transition_epoch:
            progress = epoch / max(1, self.r_transition_epoch)
            return self.r_start + (self.r_end - self.r_start) * progress
        return self.r_end

    def _create_scheduler(self):
        def lr_lambda(epoch):
            if epoch < self.warmup_epochs:
                return (epoch + 1) / self.warmup_epochs
            else:
                progress = (epoch - self.warmup_epochs) / max(1, self.num_epochs - self.warmup_epochs)
                return 0.5 * (1 + np.cos(np.pi * progress))
        return torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

    def _init_ema(self):
        ema_model = copy.deepcopy(self.model)
        ema_model.eval()
        for param in ema_model.parameters():
            param.requires_grad = False
        return ema_model

    @torch.no_grad()
    def _update_ema(self):
        for ema_param, param in zip(self.ema_model.parameters(), self.model.parameters()):
            ema_param.data.mul_(self.ema_decay).add_(param.data, alpha=1 - self.ema_decay)

    def train_epoch(self, epoch: int) -> float:
        self.model.train()
        total_loss = 0.0
        total_mse = 0.0
        num_batches = 0

        current_ratio = self._get_real_ratio(epoch)
        self.train_dataset.set_real_ratio(current_ratio)

        train_loader = DataLoader(
            self.train_dataset, self.batch_size, shuffle=True,
            num_workers=self.num_workers, pin_memory=True, drop_last=True,
        )

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.num_epochs} [r={current_ratio:.2f}]")
        for batch in pbar:
            history_coords = batch['history_coords'].to(self.device, non_blocking=True)
            future_era5 = batch['future_era5'].to(self.device, non_blocking=True)
            target_coords = batch['target_coords'].to(self.device, non_blocking=True)
            sample_weight = batch['sample_weight'].to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast('cuda', enabled=self.use_amp):
                outputs = self.model(history_coords, future_era5, target_coords)
                loss = outputs['loss']
                if train_cfg.use_sample_weights:
                    loss = loss * sample_weight.mean()

            if self.use_amp:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()

            self._update_ema()

            total_loss += loss.item()
            total_mse += outputs['mse_loss'].item()
            num_batches += 1

            if self.global_step % 50 == 0:
                self.writer.add_scalar('step/loss', loss.item(), self.global_step)
            self.global_step += 1

            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'mse': f"{outputs['mse_loss'].item():.4f}",
            })

        avg_loss = total_loss / max(num_batches, 1)
        self.writer.add_scalar('train/loss', avg_loss, epoch)
        self.writer.add_scalar('train/real_ratio', current_ratio, epoch)
        return avg_loss

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
            total_loss += outputs['loss'].item()
            num_batches += 1

        avg_loss = total_loss / max(num_batches, 1)
        if num_batches > 0:
            self.writer.add_scalar('val/loss', avg_loss, epoch)
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
            'stage': 'mixed_finetune_v3',
        }
        torch.save(checkpoint, self.checkpoint_dir / 'best.pt')
        print(f"  Saved best model (val_loss: {self.best_val_loss:.4f})")

    def train(self, resume_from: str = None):
        start_epoch = 0
        if resume_from and os.path.exists(resume_from):
            ckpt = torch.load(resume_from, map_location=self.device, weights_only=False)
            self.model.load_state_dict(ckpt['model_state_dict'])
            if 'ema_model_state_dict' in ckpt:
                self.ema_model.load_state_dict(ckpt['ema_model_state_dict'])
            start_epoch = ckpt.get('epoch', 0) + 1
            print(f"Resumed from epoch {start_epoch}")

        print(f"\n{'='*60}")
        print(f"v3: Mixed Fine-tune on Pretrained Model")
        print(f"  Strategy: freeze trajectory_predictor, train physics_encoder")
        print(f"  real_ratio: {self.r_start:.2f} -> {self.r_end:.2f}")
        print(f"  Epochs: {self.num_epochs}, Patience: {self.patience}")
        print(f"{'='*60}")

        for epoch in range(start_epoch, self.num_epochs):
            train_loss = self.train_epoch(epoch)
            self.train_losses.append(train_loss)

            val_loss = self.validate(epoch)
            self.val_losses.append(val_loss)

            self.scheduler.step()

            current_lr = self.optimizer.param_groups[0]['lr']
            current_ratio = self._get_real_ratio(epoch)

            print(f"Epoch {epoch+1}: Train={train_loss:.4f}, Val={val_loss:.4f}, "
                  f"LR={current_lr:.6f}, r={current_ratio:.2f}")

            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
                self.patience_counter = 0
            else:
                self.patience_counter += 1
            self.save_checkpoint(epoch, is_best)

            if self.patience_counter >= self.patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

        self._save_loss_plot()
        self.writer.close()

    def _save_loss_plot(self):
        if not self.train_losses:
            return
        plt.figure(figsize=(10, 6))
        epochs = range(1, len(self.train_losses) + 1)
        plt.plot(epochs, self.train_losses, 'b-', label='Train', linewidth=2)
        plt.plot(epochs, self.val_losses, 'r-', label='Val', linewidth=2)
        best_epoch = np.argmin(self.val_losses) + 1
        plt.scatter([best_epoch], [min(self.val_losses)], c='green', s=100, zorder=5,
                    label=f'Best (ep {best_epoch})')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('v3: Mixed Fine-tune (freeze decoder)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(self.checkpoint_dir / 'loss_curve.png', dpi=150, bbox_inches='tight')
        plt.close()

    def save_config(self, config_dict: dict):
        config_path = self.checkpoint_dir / 'config.json'
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, indent=2, ensure_ascii=False)
        print(f"Config saved to {config_path}")


# ============================================================
# 主函数
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="v3: Mixed fine-tune on pretrained model")

    parser.add_argument("--diffusion_code", type=str, required=True)
    parser.add_argument("--diffusion_ckpt", type=str, required=True)
    parser.add_argument("--norm_stats", type=str, required=True)
    parser.add_argument("--data_root", type=str, required=True)

    # 阶段1预训练模型（关键！）
    parser.add_argument("--pretrained", type=str, default="checkpoints/best.pt",
                        help="Stage 1 pretrained model (train.py output)")

    parser.add_argument("--track_csv", type=str, default="processed_typhoon_tracks.csv")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints_v3")
    parser.add_argument("--cache_dir", type=str, default="diffusion_era5_cache")
    parser.add_argument("--ddim_steps", type=int, default=50)
    parser.add_argument("--preprocess_dir", type=str, default=None)
    parser.add_argument("--patience", type=int, default=15)

    parser.add_argument("--r_start", type=float, default=0.8)
    parser.add_argument("--r_end", type=float, default=0.9)
    parser.add_argument("--r_transition", type=int, default=20)

    args = parser.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    _num_workers = 0 if platform.system() == 'Windows' else 2

    print("=" * 60)
    print("v3: Mixed Fine-tune on Pretrained Model")
    print(f"  Pretrained: {args.pretrained}")
    print(f"  real_ratio: {args.r_start} -> {args.r_end}")
    print(f"  lr={args.lr}, epochs={args.epochs}, batch_size={args.batch_size}")
    print("=" * 60)

    # ===== 1. 加载数据 =====
    print("\n[1/6] Loading data...")
    track_csv = args.track_csv
    if not os.path.isabs(track_csv):
        track_csv = os.path.join(TRAJ_DIR, track_csv)

    storm_samples = load_tyc_storms(
        csv_path=track_csv,
        era5_base_dir=data_root_to_era5_dir(args.data_root)
    )
    storm_samples = filter_short_storms(storm_samples, train_cfg.min_typhoon_duration_hours)
    storm_samples = filter_out_of_range_storms(storm_samples)

    train_storms, val_storms, test_storms = split_storms_by_id(
        storm_samples, train_cfg.train_ratio, train_cfg.val_ratio, seed=42
    )
    print(f"  Train: {len(train_storms)}, Val: {len(val_storms)}, Test: {len(test_storms)}")

    # ===== 2. 加载扩散 ERA5 缓存 =====
    cache_path = Path(args.cache_dir) / "era5_cache.npz"
    if cache_path.exists():
        print(f"\n[2/6] Loading diffusion ERA5 cache: {cache_path}")
        loaded = np.load(cache_path, allow_pickle=True)
        diffusion_cache = {k: loaded[k] for k in loaded.files}
        print(f"  Cached typhoons: {len(diffusion_cache)}")

        all_storm_ids = {s.storm_id for s in storm_samples}
        missing_ids = all_storm_ids - set(diffusion_cache.keys())
        if missing_ids:
            missing_storms = [s for s in storm_samples if s.storm_id in missing_ids]
            print(f"  {len(missing_storms)} missing, generating...")
            extra = generate_diffusion_era5_cache(
                storm_samples=missing_storms,
                diffusion_code=args.diffusion_code,
                diffusion_ckpt=args.diffusion_ckpt,
                norm_stats_path=args.norm_stats,
                data_root=args.data_root,
                device=device,
                ddim_steps=args.ddim_steps,
                preprocess_dir=args.preprocess_dir,
            )
            diffusion_cache.update(extra)
            np.savez_compressed(cache_path, **diffusion_cache)
    else:
        print(f"\n[2/6] Generating diffusion ERA5 cache...")
        all_storms = train_storms + val_storms + test_storms
        diffusion_cache = generate_diffusion_era5_cache(
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
        np.savez_compressed(cache_path, **diffusion_cache)

    # ===== 3. 创建数据集 =====
    print(f"\n[3/6] Creating datasets...")

    train_ds = CurriculumMixedDataset(
        storm_samples=train_storms,
        diffusion_era5_cache=diffusion_cache,
        real_ratio=args.r_start,
        stride=1,
    )

    val_ds = DiffusionERA5Dataset(
        storm_samples=val_storms,
        diffusion_era5_cache=diffusion_cache,
        stride=model_cfg.t_future,
    )

    if len(train_ds) == 0:
        print("ERROR: Training set empty!")
        return

    print(f"  Train: {len(train_ds)}, Val: {len(val_ds)}")

    val_loader = DataLoader(
        val_ds, args.batch_size, shuffle=False,
        num_workers=_num_workers, pin_memory=True,
    )

    # ===== 4. 加载预训练模型 =====
    print(f"\n[4/6] Loading pretrained model: {args.pretrained}")

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

    # 加载阶段1权重
    pretrained_ckpt = torch.load(args.pretrained, map_location='cpu', weights_only=False)
    if 'ema_model_state_dict' in pretrained_ckpt:
        model.load_state_dict(pretrained_ckpt['ema_model_state_dict'])
        print(f"  Loaded EMA weights (epoch {pretrained_ckpt.get('epoch', '?')})")
    else:
        model.load_state_dict(pretrained_ckpt['model_state_dict'])
        print(f"  Loaded model weights (epoch {pretrained_ckpt.get('epoch', '?')})")

    num_params = sum(p.numel() for p in model.parameters())
    print(f"  Total params: {num_params:,}")

    # ===== 5. 精调训练 =====
    print(f"\n[5/6] Starting mixed fine-tune...")

    trainer = FineTuneTrainerV3(
        model=model,
        train_dataset=train_ds,
        val_loader=val_loader,
        batch_size=args.batch_size,
        device=device,
        learning_rate=args.lr,
        num_epochs=args.epochs,
        checkpoint_dir=args.checkpoint_dir,
        warmup_epochs=5,
        patience=args.patience,
        r_start=args.r_start,
        r_end=args.r_end,
        r_transition_epoch=args.r_transition,
        num_workers=_num_workers,
    )

    config = {
        "stage": "mixed_finetune_v3",
        "pretrained": args.pretrained,
        "strategy": "freeze trajectory_predictor, train physics_encoder(full lr) + others(lr/10)",
        "curriculum": {"r_start": args.r_start, "r_end": args.r_end, "r_transition": args.r_transition},
        "epochs": args.epochs, "lr": args.lr, "batch_size": args.batch_size,
        "model": {"era5_channels": era5_channels, "num_params": num_params},
    }
    trainer.save_config(config)

    trainer.train()

    # ===== 6. 测试 =====
    print(f"\n[6/6] Test Evaluation")
    print("=" * 60)

    best_path = Path(args.checkpoint_dir) / 'best.pt'
    if best_path.exists():
        ckpt = torch.load(best_path, map_location=device, weights_only=False)
        if 'ema_model_state_dict' in ckpt:
            model.load_state_dict(ckpt['ema_model_state_dict'])
            print(f"Loaded best EMA (epoch {ckpt.get('epoch', '?')})")
        else:
            model.load_state_dict(ckpt['model_state_dict'])
    else:
        model = trainer.ema_model

    model.to(device).eval()

    # 真实 ERA5
    print("\n--- Real ERA5 ---")
    test_ds_real = LT3PDataset(test_storms, stride=model_cfg.t_future)
    test_loader_real = DataLoader(
        test_ds_real, args.batch_size, shuffle=False,
        num_workers=_num_workers, pin_memory=True,
    )
    results_real = evaluate_on_test(model, test_loader_real, device)

    # 扩散 ERA5
    print("\n--- Diffusion ERA5 ---")
    test_ds_diff = DiffusionERA5Dataset(
        test_storms, diffusion_cache, stride=model_cfg.t_future,
    )
    if len(test_ds_diff) > 0:
        test_loader_diff = DataLoader(
            test_ds_diff, args.batch_size, shuffle=False,
            num_workers=_num_workers, pin_memory=True,
        )
        results_diff = evaluate_on_test(model, test_loader_diff, device)
    else:
        results_diff = {}

    config["test_results"] = {"real_era5": results_real, "diffusion_era5": results_diff}
    trainer.save_config(config)

    print("\n" + "=" * 60)
    print(f"Done! Model: {args.checkpoint_dir}/best.pt")
    print("=" * 60)


if __name__ == "__main__":
    main()
