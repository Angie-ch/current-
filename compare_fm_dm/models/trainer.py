"""
统一训练模块 — Flow Matching和Diffusion双模式

FM模式:
  - 训练目标: MSE(v_pred, x_1 - x_0)
  - 采样: Euler ODE求解器

DM模式:
  - 训练目标: MSE(eps_pred, noise) 或 MSE(v_pred, velocity)
  - 采样: DDIM采样器
"""
import os
import copy
import math
import logging
import argparse
from typing import Optional, Tuple
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

logger = logging.getLogger(__name__)


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
# DataPrefetcher
# ============================================================

class DataPrefetcher:
    """GPU数据预取器"""

    def __init__(self, loader: DataLoader, device: torch.device):
        self.loader = loader
        self.device = device
        self.stream = torch.cuda.Stream(device=device) if device.type == "cuda" else None

    def __iter__(self):
        self._iter = iter(self.loader)
        self._preload()
        return self

    def _preload(self):
        try:
            self._next_batch = next(self._iter)
        except StopIteration:
            self._next_batch = None
            return

        if self.stream is not None:
            with torch.cuda.stream(self.stream):
                for key in self._next_batch:
                    if isinstance(self._next_batch[key], torch.Tensor):
                        self._next_batch[key] = self._next_batch[key].to(
                            self.device, non_blocking=True
                        )

    def __next__(self):
        if self.stream is not None:
            torch.cuda.current_stream(self.device).wait_stream(self.stream)

        batch = self._next_batch
        if batch is None:
            raise StopIteration

        self._preload()
        return batch

    def __len__(self):
        return len(self.loader)


# ============================================================
# 统一训练器
# ============================================================

class UnifiedTrainer:
    """
    统一训练器 — 支持FM和DM模式

    两种模式的区别:
    - FM: forward_fm() — 线性路径, 预测v = x_1 - x_0
    - DM: forward_dm() — 扩散路径, 预测噪声ε

    共同的优化器、学习率调度、物理损失完全相同
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        train_cfg,
        data_cfg,
        work_dir: str = ".",
        method: str = "fm",
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.cfg = train_cfg
        self.data_cfg = data_cfg
        self.work_dir = work_dir
        self.method = method  # "fm" or "dm"

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

        # 混合精度
        self.amp_dtype = (
            torch.bfloat16 if train_cfg.amp_dtype == "bfloat16" else torch.float16
        )
        self.use_amp = train_cfg.use_amp and self.device.type == "cuda"
        self.scaler = (
            torch.amp.GradScaler("cuda", enabled=self.use_amp)
            if self.use_amp
            else None
        )

        # EMA
        self.ema = EMA(self.model, decay=train_cfg.ema_decay)

        # TensorBoard
        self.writer = None
        if train_cfg.use_tensorboard:
            log_dir = os.path.join(work_dir, f"logs_{method}")
            os.makedirs(log_dir, exist_ok=True)
            self.writer = SummaryWriter(log_dir)

        # Checkpoint目录
        self.ckpt_dir = os.path.join(work_dir, f"{train_cfg.checkpoint_dir}_{method}")
        os.makedirs(self.ckpt_dir, exist_ok=True)

        # 训练状态
        self.global_step = 0
        self.optim_step = 0
        self.epoch = 0
        self.best_val_loss = float("inf")
        self.patience_counter = 0

        if train_cfg.resume_from:
            self._load_checkpoint(train_cfg.resume_from)

    def _compute_physics_weights(self) -> Tuple[float, float, float]:
        """Symmetric physics weights — same schedule for FM and DM."""
        warmup = getattr(self.cfg, 'physics_warmup_steps', 0)
        target = getattr(self.cfg, 'physics_target_weight', 0.1)
        geo_weight = getattr(self.cfg, 'geostrophic_weight', 0.0)

        if warmup == 0 or self.optim_step >= warmup:
            return target, target, geo_weight

        progress = min(1.0, self.optim_step / warmup)
        weight = target * progress
        return weight, weight, geo_weight

    def train(self):
        """主训练循环"""
        logger.info("=" * 60)
        logger.info(f"训练模式: {'Flow Matching (CFM)' if self.method == 'fm' else 'Diffusion (DDPM)'}")
        logger.info("=" * 60)
        logger.info(f"max_epochs={self.cfg.max_epochs}")
        logger.info(f"batch_size={self.cfg.batch_size}")

        epoch_pbar = tqdm(
            range(self.epoch, self.cfg.max_epochs),
            desc="训练进度",
            unit="epoch",
            initial=self.epoch,
            total=self.cfg.max_epochs,
        )
        for epoch in epoch_pbar:
            self.epoch = epoch
            train_loss = self._train_one_epoch()

            epoch_pbar.set_postfix(
                loss=f"{train_loss:.4f}",
                lr=f"{self.optimizer.param_groups[0]['lr']:.2e}",
                best=f"{self.best_val_loss:.4f}",
            )
            logger.info(
                f"Epoch {epoch+1}/{self.cfg.max_epochs} | "
                f"train_loss={train_loss:.6f} | "
                f"lr={self.optimizer.param_groups[0]['lr']:.2e}"
            )

            if (epoch + 1) % self.cfg.eval_every == 0:
                val_loss = self._validate()
                logger.info(f"  验证 loss={val_loss:.6f} (best={self.best_val_loss:.6f})")

                if self.writer:
                    self.writer.add_scalar("val/loss", val_loss, epoch + 1)

                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.patience_counter = 0
                    self._save_checkpoint(f"best.pt", is_best=True)
                    logger.info(f"  ✓ 新的最佳模型已保存 ({self.method.upper()})")
                else:
                    self.patience_counter += 1
                    logger.info(
                        f"  ✗ 无改善 ({self.patience_counter}/{self.cfg.early_stopping_patience})"
                    )

                if self.patience_counter >= self.cfg.early_stopping_patience:
                    logger.info(f"Early Stopping: 连续 {self.patience_counter} 次验证无改善")
                    break

            if (epoch + 1) % (self.cfg.eval_every * 2) == 0:
                self._save_checkpoint("latest.pt")

        self._save_checkpoint("final.pt")
        if self.writer:
            self.writer.close()
        logger.info(f"{self.method.upper()} 训练完成!")

    def _train_one_epoch(self) -> float:
        """训练一个epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        prefetcher = DataPrefetcher(self.train_loader, self.device)
        self.optimizer.zero_grad(set_to_none=True)
        accum_loss = 0.0

        batch_pbar = tqdm(
            enumerate(prefetcher),
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

            # 条件噪声增强 (applied in-place on condition)
            if self.cfg.condition_noise_sigma > 0:
                rampup = min(1.0, self.epoch / max(self.cfg.condition_noise_rampup_epochs, 1))
                noise_sigma = self.cfg.condition_noise_sigma * rampup
                noise_mask = (
                    torch.rand(condition.shape[0], 1, 1, 1, device=condition.device)
                    < self.cfg.condition_noise_prob
                ).float()

                white_noise = torch.randn_like(condition)
                if getattr(self.cfg, 'condition_noise_spatial_smooth', False):
                    ks = getattr(self.cfg, 'condition_noise_smooth_kernel', 5)
                    pad = ks // 2
                    smooth_noise = F.avg_pool2d(
                        white_noise, kernel_size=ks, stride=1, padding=pad
                    )
                    smooth_std = smooth_noise.std(dim=(2, 3), keepdim=True).clamp(min=1e-6)
                    smooth_noise = smooth_noise / smooth_std
                    noise = 0.5 * white_noise + 0.5 * smooth_noise
                else:
                    noise = white_noise

                condition = condition + noise_mask * noise_sigma * noise

            # Sync noisy condition back to batch dict for model forward
            batch["condition"] = condition

            # Move center_lats to GPU if present (for GeostrophicBalanceLoss)
            if "center_lats" in batch:
                cl = batch["center_lats"]
                if not cl.is_cuda:
                    batch["center_lats"] = cl.to(self.device)

            # 物理损失权重
            w_div, w_curl, w_geo = self._compute_physics_weights()

            # ZPredictor weight (Option 2)
            w_z_pred = getattr(self.cfg, 'z_predictor_weight', 0.0)

            # 前向传播 — pass batch for center_lats (geo loss)
            with torch.amp.autocast("cuda", dtype=self.amp_dtype, enabled=self.use_amp):
                outputs = self.model(condition, target, batch=batch)
                loss_geo = outputs.get("loss_geo", torch.tensor(0.0, device=condition.device))
                loss_z_pred = outputs.get("loss_z_pred", torch.tensor(0.0, device=condition.device))
                # Unified aux loss key — both FM (loss_x1) and DM (loss_x1) use same key
                aux_loss = outputs.get("loss_x1", torch.tensor(0.0, device=condition.device))
                aux_w = getattr(self.cfg, 'aux_loss_weight', 0.5)
                loss = (
                    outputs["loss_mse"]
                    + aux_w * aux_loss
                    + w_div * outputs["loss_div"]
                    + w_div * outputs["loss_sol"]
                    + w_curl * outputs["loss_curl"]
                    + w_geo * loss_geo
                    + w_z_pred * loss_z_pred
                )
                loss = loss / self.cfg.gradient_accumulation_steps

            # 安全检测
            loss_val = loss.item()
            if not torch.isfinite(loss) or loss_val > 10.0:
                logger.warning(f"[Step {self.global_step}] 异常 loss={loss_val:.4f}, 跳过")
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
                self.lr_scheduler.step()
                self.ema.update(self.model)
                self.optim_step += 1

                total_loss += accum_loss
                num_batches += 1

                if self.optim_step % self.cfg.log_every == 0:
                    lr = self.optimizer.param_groups[0]["lr"]
                    mse_val = outputs["loss_mse"].item()
                    x1_val = outputs["loss_x1"].item()
                    div_val = outputs["loss_div"].item()
                    sol_val = outputs["loss_sol"].item()
                    curl_val = outputs["loss_curl"].item()
                    z_pred_val = outputs.get("loss_z_pred", torch.tensor(0.0)).item()
                    batch_pbar.set_postfix(
                        loss=f"{accum_loss:.4f}",
                        mse=f"{mse_val:.4f}",
                        x1=f"{x1_val:.4f}",
                        div=f"{div_val:.4f}",
                        sol=f"{sol_val:.4f}",
                        zpred=f"{z_pred_val:.4f}",
                        lr=f"{lr:.2e}",
                    )
                    if self.writer:
                        self.writer.add_scalar("train/loss_total", accum_loss, self.optim_step)
                        self.writer.add_scalar("train/loss_mse", mse_val, self.optim_step)
                        self.writer.add_scalar("train/loss_x1", x1_val, self.optim_step)
                        self.writer.add_scalar("train/loss_div", div_val, self.optim_step)
                        self.writer.add_scalar("train/loss_sol", sol_val, self.optim_step)
                        self.writer.add_scalar("train/loss_curl", curl_val, self.optim_step)
                        self.writer.add_scalar("train/loss_z_pred", z_pred_val, self.optim_step)
                        self.writer.add_scalar("train/lr", lr, self.optim_step)

                    logger.info(
                        f"  step={self.optim_step} | "
                        f"loss={accum_loss:.6f} | mse={mse_val:.6f} | "
                        f"x1={x1_val:.6f} | div={div_val:.6f} | sol={sol_val:.6f} | "
                        f"zpred={z_pred_val:.6f} | lr={lr:.2e}"
                    )

                accum_loss = 0.0

        return total_loss / max(num_batches, 1)

    @torch.no_grad()
    def _validate(self) -> float:
        """验证（使用EMA参数）"""
        self.ema.apply_shadow(self.model)
        self.model.eval()

        total_loss = 0.0
        num_batches = 0

        val_pbar = tqdm(self.val_loader, desc="验证中", unit="batch", leave=False)
        w_div, w_curl, w_geo = self._compute_physics_weights()
        w_z_pred = getattr(self.cfg, 'z_predictor_weight', 0.0)
        for batch in val_pbar:
            condition = batch["condition"].to(self.device)
            target = batch["target"].to(self.device)
            if "center_lats" in batch:
                cl = batch["center_lats"]
                batch["center_lats"] = cl.to(self.device) if not cl.is_cuda else cl

            with torch.amp.autocast("cuda", dtype=self.amp_dtype, enabled=self.use_amp):
                outputs = self.model(condition, target, batch=batch)
                loss_geo = outputs.get("loss_geo", torch.tensor(0.0, device=condition.device))
                loss_z_pred = outputs.get("loss_z_pred", torch.tensor(0.0, device=condition.device))
                aux_loss = outputs.get("loss_x1", torch.tensor(0.0, device=condition.device))
                aux_w = getattr(self.cfg, 'aux_loss_weight', 0.5)
                loss = (
                    outputs["loss_mse"]
                    + aux_w * aux_loss
                    + w_div * outputs["loss_div"]
                    + w_div * outputs["loss_sol"]
                    + w_curl * outputs["loss_curl"]
                    + w_geo * loss_geo
                    + w_z_pred * loss_z_pred
                )

            total_loss += loss.item()
            num_batches += 1

        self.ema.restore(self.model)
        return total_loss / max(num_batches, 1)

    def _save_checkpoint(self, filename: str, is_best: bool = False):
        """保存checkpoint"""
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
            "method": self.method,
        }
        torch.save(state, path)
        logger.info(f"Checkpoint 已保存: {path}")

    def _load_checkpoint(self, path: str):
        """加载checkpoint"""
        logger.info(f"加载 checkpoint: {path}")
        state = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(state["model_state_dict"], strict=False)
        self.optimizer.load_state_dict(state["optimizer_state_dict"])
        self.ema.load_state_dict(state["ema_state_dict"])
        self.epoch = state["epoch"] + 1
        self.global_step = state["global_step"]
        self.optim_step = state["optim_step"]
        self.best_val_loss = state["best_val_loss"]
        self.patience_counter = state["patience_counter"]


def train_model(
    model,
    train_loader,
    val_loader,
    train_cfg,
    data_cfg,
    work_dir: str,
    method: str,
):
    """训练入口函数"""
    trainer = UnifiedTrainer(
        model, train_loader, val_loader, train_cfg, data_cfg, work_dir, method
    )
    trainer.train()
    return trainer
