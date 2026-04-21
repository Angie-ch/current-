r"""
Plan C: 从头训练轨迹模型，完全使用扩散模型预测的ERA5
解决级联系统的 train-test distribution match 问题

与 Plan B 的区别:
  - Plan B: 预训练(真实ERA5) → 微调(扩散ERA5)
  - Plan C: 从零训练(扩散ERA5)

使用方式:
  python planc_train.py \
      --diffusion_code /root/autodl-tmp/newtry \
      --diffusion_ckpt /root/autodl-tmp/newtry/checkpoints/best.pt \
      --norm_stats /root/autodl-tmp/newtry/norm_stats.pt \
      --data_root /root/autodl-tmp/Typhoon_data_final \
      --track_csv processed_typhoon_tracks.csv \
      --preprocess_dir /root/autodl-tmp/preprocessed_npy \
      --epochs 200 \
      --batch_size 32
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
    normalize_coords, denormalize_coords,
    filter_short_storms, filter_out_of_range_storms, split_storms_by_id,
)
from data_processing import load_tyc_storms
from data_structures import StormSample
from train import evaluate_on_test

# 复用 finetune_train.py 的数据集和缓存生成函数
from finetune_train import DiffusionERA5Dataset, generate_diffusion_era5_cache


# ============================================================
# Plan C 训练器（从零训练，非微调）
# ============================================================

class PlanCTrainer:
    """
    Plan C 训练器：从零开始用扩散ERA5训练轨迹模型
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        era5_channels: int,
        device: str = 'cuda',
        learning_rate: float = 1e-4,
        num_epochs: int = 200,
        checkpoint_dir: str = 'checkpoints_planc',
        warmup_epochs: int = 10,
    ):
        self.device = device
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.num_epochs = num_epochs
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.era5_channels = era5_channels

        # 优化器（与 train.py 保持一致）
        self.optimizer = AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=1e-4,
            betas=(0.9, 0.999),
        )

        # 学习率调度: Warmup + Cosine Annealing
        self.warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            self.optimizer,
            start_factor=0.01,
            end_factor=1.0,
            total_iters=warmup_epochs,
        )
        self.cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=num_epochs - warmup_epochs,
            eta_min=learning_rate * 0.01,
        )
        self.scheduler = torch.optim.lr_scheduler.SequentialLR(
            self.optimizer,
            schedulers=[self.warmup_scheduler, self.cosine_scheduler],
            milestones=[warmup_epochs],
        )

        # EMA（指数移动平均，与 train.py 一致）
        self.ema_decay = 0.9999
        self.ema_model = self._create_ema_model()

        # 日志
        self.writer = SummaryWriter(log_dir=str(self.checkpoint_dir / 'tb_logs'))
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.patience = 25  # Early stopping patience
        self.global_step = 0

    def _create_ema_model(self):
        """创建 EMA 模型副本"""
        ema_model = type(self.model)(
            coord_dim=model_cfg.coord_dim,
            output_dim=model_cfg.output_dim,
            era5_channels=self.era5_channels,
            t_history=model_cfg.t_history,
            t_future=model_cfg.t_future,
            d_model=model_cfg.transformer_dim,
            n_heads=model_cfg.transformer_heads,
            n_layers=model_cfg.transformer_layers,
            ff_dim=model_cfg.transformer_ff_dim,
            dropout=model_cfg.dropout,
        ).to(self.device)
        ema_model.load_state_dict(self.model.state_dict())
        for param in ema_model.parameters():
            param.requires_grad = False
        return ema_model

    @torch.no_grad()
    def _update_ema(self):
        """更新 EMA 参数"""
        for ema_param, model_param in zip(self.ema_model.parameters(), self.model.parameters()):
            ema_param.data.mul_(self.ema_decay).add_(model_param.data, alpha=1 - self.ema_decay)

    @torch.no_grad()
    def validate(self, epoch: int = 0, use_ema: bool = True) -> float:
        """验证（使用 EMA 模型）"""
        eval_model = self.ema_model if use_ema else self.model
        eval_model.eval()

        total_loss = 0.0
        num_batches = 0

        for batch in self.val_loader:
            history_coords = batch['history_coords'].to(self.device)
            future_era5 = batch['future_era5'].to(self.device)
            target_coords = batch['target_coords'].to(self.device)

            outputs = eval_model(history_coords, future_era5, target_coords)
            total_loss += outputs['loss'].item()
            num_batches += 1

        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        self.writer.add_scalar('planc_val/loss', avg_loss, epoch)
        return avg_loss

    def train_epoch(self, epoch: int) -> float:
        """训练一个 epoch"""
        self.model.train()
        total_loss = 0.0
        total_mse = 0.0
        num_batches = 0

        pbar = tqdm(self.train_loader, desc=f"Plan C Epoch {epoch+1}/{self.num_epochs}")
        for batch in pbar:
            history_coords = batch['history_coords'].to(self.device, non_blocking=True)
            future_era5 = batch['future_era5'].to(self.device, non_blocking=True)
            target_coords = batch['target_coords'].to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)

            outputs = self.model(history_coords, future_era5, target_coords)
            loss = outputs['loss']

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
                self.writer.add_scalar('planc_step/loss', loss.item(), self.global_step)
                self.writer.add_scalar('planc_step/mse', outputs['mse_loss'].item(), self.global_step)
            self.global_step += 1

            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'mse': f"{outputs['mse_loss'].item():.4f}",
            })

        avg_loss = total_loss / num_batches
        self.writer.add_scalar('planc_train/loss', avg_loss, epoch)
        self.writer.add_scalar('planc_train/mse', total_mse / num_batches, epoch)
        return avg_loss

    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """保存 checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'ema_model_state_dict': self.ema_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'stage': 'planc',
        }

        # 保存最新
        torch.save(checkpoint, self.checkpoint_dir / 'latest.pt')

        # 保存最佳
        if is_best:
            torch.save(checkpoint, self.checkpoint_dir / 'best.pt')
            print(f"  ✓ 保存最佳模型 (val_loss: {self.best_val_loss:.4f})")

    def train(self):
        """主训练循环"""
        print(f"\n{'='*60}")
        print("Plan C: 从零训练（完全使用扩散ERA5）")
        print(f"Epochs: {self.num_epochs}, LR: {self.optimizer.param_groups[0]['lr']:.1e}")
        print(f"Train batches: {len(self.train_loader)}, Val batches: {len(self.val_loader)}")
        print(f"{'='*60}")

        for epoch in range(self.num_epochs):
            train_loss = self.train_epoch(epoch)
            self.train_losses.append(train_loss)

            val_loss = self.validate(epoch, use_ema=True)
            self.val_losses.append(val_loss)

            self.scheduler.step()

            current_lr = self.scheduler.get_last_lr()[0]
            self.writer.add_scalar('planc_train/lr', current_lr, epoch)

            print(f"Epoch {epoch+1}: Train={train_loss:.4f}, Val={val_loss:.4f}, LR={current_lr:.2e}")

            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
                self.patience_counter = 0
            else:
                self.patience_counter += 1

            self.save_checkpoint(epoch, is_best)

            if self.patience_counter >= self.patience:
                print(f"Early stopping! 连续 {self.patience} 个 epoch 无改善")
                break

        self.writer.close()
        print(f"\n训练完成! Best val_loss: {self.best_val_loss:.4f}")
        print(f"模型保存至: {self.checkpoint_dir / 'best.pt'}")


# ============================================================
# 主函数
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Plan C: 从零训练轨迹模型（完全使用扩散ERA5）")

    # 必需参数
    parser.add_argument("--diffusion_code", type=str, required=True,
                        help="扩散模型代码目录")
    parser.add_argument("--diffusion_ckpt", type=str, required=True,
                        help="扩散模型checkpoint")
    parser.add_argument("--norm_stats", type=str, required=True,
                        help="扩散模型归一化统计 (norm_stats.pt)")
    parser.add_argument("--data_root", type=str, required=True,
                        help="ERA5数据根目录")

    # 可选参数
    parser.add_argument("--track_csv", type=str, default="processed_typhoon_tracks.csv")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints_planc")
    parser.add_argument("--ddim_steps", type=int, default=50)
    parser.add_argument("--preprocess_dir", type=str, default=None,
                        help="扩散模型预处理NPY目录")
    parser.add_argument("--cache_dir", type=str, default="diffusion_era5_cache",
                        help="扩散ERA5缓存保存目录")
    parser.add_argument("--warmup_epochs", type=int, default=10)

    args = parser.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print("=" * 60)
    print("Plan C: 从零训练（完全使用扩散ERA5）")
    print("=" * 60)

    # ===== 1. 加载台风轨迹数据 =====
    print("\n[1/4] 加载台风轨迹数据...")
    track_csv = args.track_csv
    if not os.path.isabs(track_csv):
        track_csv = os.path.join(TRAJ_DIR, track_csv)

    storm_samples = load_tyc_storms(
        csv_path=track_csv,
        era5_base_dir=args.data_root
    )
    storm_samples = filter_short_storms(storm_samples, train_cfg.min_typhoon_duration_hours)
    storm_samples = filter_out_of_range_storms(storm_samples)
    print(f"  可用台风: {len(storm_samples)}")

    # 划分数据集
    train_storms, val_storms, test_storms = split_storms_by_id(
        storm_samples, train_cfg.train_ratio, train_cfg.val_ratio, seed=42
    )
    print(f"  训练台风: {len(train_storms)}, 验证台风: {len(val_storms)}, 测试台风: {len(test_storms)}")

    # ===== 2. 生成/加载扩散ERA5缓存 =====
    cache_path = Path(args.cache_dir) / "era5_cache.npz"

    if cache_path.exists():
        print(f"\n[2/4] 加载已有扩散ERA5缓存: {cache_path}")
        loaded = np.load(cache_path, allow_pickle=True)
        diffusion_cache = {k: loaded[k] for k in loaded.files}
        print(f"  缓存台风数: {len(diffusion_cache)}")

        # 检查是否需要补充生成
        all_storm_ids = {s.storm_id for s in storm_samples}
        missing_ids = [s for s in storm_samples if s.storm_id not in diffusion_cache]

        if missing_ids:
            print(f"  有 {len(missing_ids)} 个台风不在缓存中，补充生成...")
            extra_cache = generate_diffusion_era5_cache(
                storm_samples=missing_ids,
                diffusion_code=args.diffusion_code,
                diffusion_ckpt=args.diffusion_ckpt,
                norm_stats_path=args.norm_stats,
                data_root=args.data_root,
                device=device,
                ddim_steps=args.ddim_steps,
                preprocess_dir=args.preprocess_dir,
            )
            diffusion_cache.update(extra_cache)
            # 更新缓存文件
            np.savez_compressed(cache_path, **diffusion_cache)
            print(f"  缓存已更新: {len(diffusion_cache)} 个台风")
    else:
        print(f"\n[2/4] 生成扩散ERA5缓存 (首次运行，需要较长时间)...")
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

        # 保存缓存
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache_path, **diffusion_cache)
        print(f"  缓存已保存至: {cache_path}")

    # ===== 3. 创建数据集 =====
    print("\n[3/4] 创建训练数据集...")
    train_ds = DiffusionERA5Dataset(
        train_storms, diffusion_cache, stride=1
    )
    val_ds = DiffusionERA5Dataset(
        val_storms, diffusion_cache, stride=model_cfg.t_future
    )

    if len(train_ds) == 0:
        print("错误: 训练集为空! 请检查扩散ERA5缓存是否生成成功。")
        return

    train_loader = DataLoader(
        train_ds, args.batch_size, shuffle=True,
        num_workers=4, pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        val_ds, args.batch_size, shuffle=False,
        num_workers=4, pin_memory=True
    )

    print(f"  训练样本: {len(train_ds)}, 验证样本: {len(val_ds)}")

    # ===== 4. 创建模型并训练 =====
    print("\n[4/4] 创建模型并开始训练...")

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

    num_params = sum(p.numel() for p in model.parameters())
    print(f"  模型参数量: {num_params:,}")

    # 训练
    trainer = PlanCTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        era5_channels=era5_channels,
        device=device,
        learning_rate=args.learning_rate,
        num_epochs=args.epochs,
        checkpoint_dir=args.checkpoint_dir,
        warmup_epochs=args.warmup_epochs,
    )

    # 保存配置
    config = {
        "stage": "planc",
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
        "train_samples": len(train_ds),
        "val_samples": len(val_ds),
        "train_storms": len(train_storms),
        "val_storms": len(val_storms),
        "test_storms": len(test_storms),
        "diffusion_ckpt": args.diffusion_ckpt,
        "ddim_steps": args.ddim_steps,
    }
    config_path = Path(args.checkpoint_dir) / "planc_config.json"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    trainer.train()

    # ===== 5. 测试集评估 =====
    print("\n[5/5] 在测试集上评估...")

    # 加载最佳权重
    best_ckpt = torch.load(
        Path(args.checkpoint_dir) / 'best.pt',
        map_location=device, weights_only=False
    )
    model.load_state_dict(best_ckpt['ema_model_state_dict'])
    model.to(device)
    model.eval()
    print(f"  已加载最佳模型 (epoch {best_ckpt.get('epoch', '?')})")

    # 用扩散 ERA5 测试
    test_ds = DiffusionERA5Dataset(
        test_storms, diffusion_cache, stride=model_cfg.t_future
    )

    if len(test_ds) > 0:
        test_loader = DataLoader(
            test_ds, args.batch_size, shuffle=False,
            num_workers=4, pin_memory=True
        )
        print(f"  测试样本数: {len(test_ds)}")
        print("\n--- Plan C 测试结果 (扩散ERA5输入) ---")
        evaluate_on_test(model, test_loader, device)
    else:
        print("  测试集为空，跳过评估")

    print("\nPlan C 训练完成!")
    print(f"最终模型: {args.checkpoint_dir}/best.pt")


if __name__ == "__main__":
    main()
