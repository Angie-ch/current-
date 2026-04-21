"""
Flow Matching 训练脚本 — 替代 ERA5-Diffusion 的 DDPM 训练

核心变化（vs newtry/train.py）:
  - 无扩散调度器: 移除 DiffusionScheduler, 不需要 betas/alphas
  - 连续时间: t ∈ [0, 1] 均匀采样 (而非离散 t ∈ {0,...,999})
  - 预测速度: MSE(v_pred, x_1 - x_0) (而非 MSE(eps_pred, noise))
  - 欧拉采样: 自回归推理用 Euler 1步 (而非 DDIM 50步)

训练流程:
  1. 采样 t ~ U[0,1]
  2. 采样 x_1 ~ N(0,I)
  3. 计算 x_t = (1-t)*x_0 + t*x_1
  4. 计算目标速度 v = x_1 - x_0
  5. 最小化 MSE(v_θ(x_t,t,c), v)

参考:
  - torchcfm: https://github.com/atong01/conditional-flow-matching
  - Lipman et al. ICLR 2023
"""
import os
import sys
import copy
import time
import math
import logging
import argparse
from pathlib import Path
from collections import OrderedDict
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from configs import DataConfig, ModelConfig, TrainConfig, InferenceConfig, get_config

FLOW_MATCHING_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(FLOW_MATCHING_DIR)
sys.path.insert(0, PARENT_DIR)

from newtry.data.dataset import ERA5TyphoonDataset, split_typhoon_ids, build_dataloaders

from models.flow_matching_model import ERA5FlowMatchingModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
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
    """GPU 数据预取器"""

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

        # 学习率调度: Warmup + Cosine Annealing
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
            torch.amp.GradScaler("cuda")
            if self.use_amp and self.amp_dtype == torch.float16
            else None
        )

        # EMA
        self.ema = EMA(self.model, decay=train_cfg.ema_decay)

        # TensorBoard
        self.writer = None
        if train_cfg.use_tensorboard:
            log_dir = os.path.join(work_dir, "logs_cfm")
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

        # P2: 物理损失退火调度的初始值
        self.physics_weight_div = 0.0
        self.physics_weight_curl = 0.0

        if train_cfg.resume_from:
            self._load_checkpoint(train_cfg.resume_from)

    def _compute_physics_weights(self) -> Tuple[float, float]:
        """
        P2: 计算当前步的物理损失权重（退火调度）

        策略:
          - 前 warmup_steps 步: 权重为 0（避免误导性梯度）
          - 之后: 线性或余弦增长至目标权重
        """
        warmup = getattr(self.cfg, 'physics_warmup_steps', 10000)
        target = getattr(self.cfg, 'physics_target_weight', 1.0)
        sched_type = getattr(self.cfg, 'physics_warmup_type', 'linear')

        if self.optim_step < warmup:
            # warmup阶段: 权重为0
            return 0.0, 0.0

        # 从 warmup 步开始增长，经过 warmup 步后达到目标值
        progress = min(1.0, (self.optim_step - warmup) / warmup)

        if sched_type == "cosine":
            # 余弦退火: 从 0 增长到 target
            weight = target * (1 - 0.5 * math.cos(math.pi * progress))
        else:
            # 线性增长: 从 0 增长到 target
            weight = target * progress

        return weight, weight

    def train(self):
        """主训练循环"""
        logger.info("=" * 60)
        logger.info("Flow Matching 训练 — 经典流匹配 (CFM)")
        logger.info("预测目标: 速度场 v = x_1 - x_0")
        logger.info("采样方法: 欧拉积分 (1~4步)")
        logger.info("=" * 60)
        logger.info(f"max_epochs={self.cfg.max_epochs}")
        logger.info(
            f"batch_size={self.cfg.batch_size} × "
            f"grad_accum={self.cfg.gradient_accumulation_steps} = "
            f"effective_batch={self.cfg.batch_size * self.cfg.gradient_accumulation_steps}"
        )
        if self.cfg.condition_noise_sigma > 0:
            logger.info(
                f"条件噪声增强: sigma={self.cfg.condition_noise_sigma}, "
                f"prob={self.cfg.condition_noise_prob}, "
                f"rampup={self.cfg.condition_noise_rampup_epochs} epochs"
            )

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
                    logger.info(f"  ✓ 新的最佳模型已保存 (CFM)")
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
        logger.info("Flow Matching 训练完成!")

    def _train_one_epoch(self) -> float:
        """训练一个 epoch"""
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

            # 条件噪声增强 (与 Diffusion 版本兼容)
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

            # P2: 计算物理损失权重（退火调度）
            physics_w_div, physics_w_curl = self._compute_physics_weights()

            # 混合精度前向
            with torch.amp.autocast("cuda", dtype=self.amp_dtype, enabled=self.use_amp):
                outputs = self.model(condition, target)
                loss = (
                    outputs["loss_mse"]
                    + physics_w_div * outputs["loss_div"]
                    + physics_w_curl * outputs["loss_curl"]
                )
                loss = loss / self.cfg.gradient_accumulation_steps

            # 保存当前物理权重用于日志记录
            self.physics_weight_div = physics_w_div
            self.physics_weight_curl = physics_w_curl

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
                self.lr_scheduler.step()
                self.ema.update(self.model)
                self.optim_step += 1

                total_loss += accum_loss
                num_batches += 1

                if self.optim_step % self.cfg.log_every == 0:
                    lr = self.optimizer.param_groups[0]["lr"]
                    mse_val = outputs["loss_mse"].item()
                    div_val = outputs["loss_div"].item()
                    curl_val = outputs["loss_curl"].item()
                    batch_pbar.set_postfix(
                        loss=f"{accum_loss:.4f}",
                        mse=f"{mse_val:.4f}",
                        div=f"{div_val:.4f}",
                        curl=f"{curl_val:.4f}",
                        lr=f"{lr:.2e}",
                    )
                    if self.writer:
                        self.writer.add_scalar("train/loss_total", accum_loss, self.optim_step)
                        self.writer.add_scalar("train/loss_mse", mse_val, self.optim_step)
                        self.writer.add_scalar("train/loss_div", div_val, self.optim_step)
                        self.writer.add_scalar("train/loss_curl", curl_val, self.optim_step)
                        self.writer.add_scalar("train/lr", lr, self.optim_step)
                        # P2: 记录物理损失权重
                        self.writer.add_scalar("train/physics_weight_div", self.physics_weight_div, self.optim_step)
                        self.writer.add_scalar("train/physics_weight_curl", self.physics_weight_curl, self.optim_step)

                        if self.optim_step % (self.cfg.log_every * 10) == 0:
                            with torch.no_grad():
                                v_pred = outputs["v_pred"]
                                v_true = outputs["v_true"]
                                C = v_pred.shape[1]
                                per_ch_mse = ((v_pred - v_true) ** 2).mean(dim=(0, 2, 3))
                                ch_names = []
                                for var in self.data_cfg.pressure_level_vars:
                                    for lev in self.data_cfg.pressure_levels:
                                        ch_names.append(f"{var}_{lev}")
                                for ci in range(min(C, len(ch_names))):
                                    self.writer.add_scalar(
                                        f"velocity_channel_mse/{ch_names[ci]}",
                                        per_ch_mse[ci].item(),
                                        self.optim_step,
                                    )

                    logger.info(
                        f"  step={self.optim_step} | "
                        f"loss={accum_loss:.6f} | "
                        f"mse={mse_val:.6f} | div={div_val:.6f} | curl={curl_val:.6f} | "
                        f"lr={lr:.2e}"
                    )

                accum_loss = 0.0

        return total_loss / max(num_batches, 1)

    @torch.no_grad()
    def _validate(self) -> float:
        """验证（使用 EMA 参数）"""
        self.ema.apply_shadow(self.model)
        self.model.eval()

        total_loss = 0.0
        num_batches = 0

        # P2: 验证时使用目标物理权重（不进行退火）
        val_physics_w_div = getattr(self.cfg, 'physics_target_weight', 1.0)
        val_physics_w_curl = getattr(self.cfg, 'physics_target_weight', 1.0)

        val_pbar = tqdm(self.val_loader, desc="验证中", unit="batch", leave=False)
        for batch in val_pbar:
            condition = batch["condition"].to(self.device)
            target = batch["target"].to(self.device)

            with torch.amp.autocast("cuda", dtype=self.amp_dtype, enabled=self.use_amp):
                outputs = self.model(condition, target)
                loss = (
                    outputs["loss_mse"]
                    + val_physics_w_div * outputs["loss_div"]
                    + val_physics_w_curl * outputs["loss_curl"]
                )

            total_loss += loss.item()
            num_batches += 1

        self.ema.restore(self.model)
        return total_loss / max(num_batches, 1)

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
            "method": "flow_matching_cfm",
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
            logger.warning(f"⚠️ 模型中有 {len(missing)} 个 key 未从 checkpoint 加载:")
            for k in missing:
                logger.warning(f"  MISSING: {k}")
        if unexpected:
            logger.warning(f"⚠️ checkpoint 中有 {len(unexpected)} 个 key 在当前模型中不存在:")
            for k in unexpected:
                logger.warning(f"  UNEXPECTED: {k}")
        if not missing and not unexpected:
            logger.info("✅ 模型权重完全匹配，全部加载成功")
        self.optimizer.load_state_dict(state["optimizer_state_dict"])
        self.ema.load_state_dict(state["ema_state_dict"])
        self.epoch = state["epoch"] + 1
        self.global_step = state["global_step"]
        self.optim_step = state["optim_step"]
        self.best_val_loss = state["best_val_loss"]
        self.patience_counter = state["patience_counter"]


# ============================================================
# 测试集评估
# ============================================================

def _find_ar_sequences(test_dataset, ar_steps: int, num_samples: int):
    """在测试数据集中找到足够长的连续序列"""
    if len(test_dataset) == 0:
        return []

    sequences = []
    current_tid = None
    current_start = -1
    current_len = 0

    for idx in range(len(test_dataset)):
        sample_info = test_dataset.samples[idx]
        tid = sample_info[0]

        if tid == current_tid:
            current_len += 1
        else:
            if current_tid is not None and current_len >= ar_steps:
                sequences.append((current_start, current_len, current_tid))
            current_tid = tid
            current_start = idx
            current_len = 1

    if current_tid is not None and current_len >= ar_steps:
        sequences.append((current_start, current_len, current_tid))

    valid_starts = []
    for seg_start, seg_len, tid in sequences:
        max_start = seg_start + seg_len - ar_steps
        for idx in range(seg_start, max_start + 1):
            valid_starts.append(idx)
            if len(valid_starts) >= num_samples:
                return valid_starts

    return valid_starts


@torch.no_grad()
def evaluate_on_test(
    model: ERA5FlowMatchingModel,
    test_dataset,
    data_cfg,
    infer_cfg,
    norm_mean: np.ndarray,
    norm_std: np.ndarray,
    device: torch.device,
    num_samples: int = 5,
):
    """
    在测试集上做自回归评估 (24步 → 72h)
    使用 CFM 欧拉采样
    """
    from inference import EulerCFMTranslator, evaluate_predictions

    translator = EulerCFMTranslator(model, data_cfg)
    translator.euler_steps = infer_cfg.euler_steps
    translator.z_clamp_range = infer_cfg.z_clamp_range

    ar_steps = infer_cfg.autoregressive_steps
    noise_sigma = infer_cfg.autoregressive_noise_sigma
    C = data_cfg.num_channels

    effective_num = num_samples if num_samples > 0 else len(test_dataset)
    valid_starts = _find_ar_sequences(test_dataset, ar_steps, effective_num)
    n_samples = len(valid_starts)

    if n_samples == 0:
        logger.warning("测试集中无足够长的连续序列，跳过自回归评估")
        return

    logger.info(f"找到 {n_samples} 个有效评估起始点")

    all_preds = [[] for _ in range(ar_steps)]
    all_gts = [[] for _ in range(ar_steps)]

    for sample_idx, start_idx in enumerate(
        tqdm(valid_starts, desc="测试集自回归推理 (CFM)", unit="样本")
    ):
        sample = test_dataset[start_idx]
        cond = sample["condition"].unsqueeze(0).to(device)

        preds = translator.predict_autoregressive(
            cond, num_steps=ar_steps, noise_sigma=noise_sigma,
        )

        for t in range(ar_steps):
            all_preds[t].append(preds[t].cpu())
            gt_sample = test_dataset[start_idx + t]
            gt_step = gt_sample["target"][:C]
            all_gts[t].append(gt_step.unsqueeze(0))

    valid_preds = [torch.cat(all_preds[t], dim=0) for t in range(ar_steps)]
    valid_gts = [torch.cat(all_gts[t], dim=0) for t in range(ar_steps)]

    var_names = []
    for var in data_cfg.pressure_level_vars:
        for lev in data_cfg.pressure_levels:
            var_names.append(f"{var}_{lev}")

    results = evaluate_predictions(
        valid_preds, valid_gts, norm_mean, norm_std, var_names,
    )
    rmse = results["rmse"]

    n_leads = rmse.shape[0]
    n_vars = rmse.shape[1]

    header = f"{'时效':>10}"
    for v in var_names:
        header += f"  {v:>8}"
    logger.info(header)
    logger.info("=" * (10 + 10 * n_vars))

    for t in range(n_leads):
        lead_h = (t + 1) * data_cfg.time_interval_hours
        row = f"+ {lead_h:>3}h    "
        for c in range(n_vars):
            row += f"  {rmse[t, c]:>8.2f}"
        logger.info(row)

    mean_row = f"{'平均':>10}"
    for c in range(n_vars):
        mean_row += f"  {rmse[:, c].mean():>8.2f}"
    logger.info(mean_row)


# ============================================================
# 入口
# ============================================================

def main():
    torch.set_float32_matmul_precision("high")

    parser = argparse.ArgumentParser(description="Flow Matching Training (CFM)")
    parser.add_argument("--data_root", type=str, default="/Volumes/T7 Shield/Typhoon_data_final")
    parser.add_argument("--work_dir", type=str, default=".")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--preprocess_dir", type=str, default=None)
    parser.add_argument("--test_samples", type=int, default=0)
    parser.add_argument("--norm_stats", type=str, default=None,
                        help="归一化统计文件路径 (与 newtry 共享)")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    data_cfg, model_cfg, train_cfg, infer_cfg = get_config(data_root=args.data_root)

    if args.batch_size:
        train_cfg.batch_size = args.batch_size
    if args.epochs:
        train_cfg.max_epochs = args.epochs
    if args.lr:
        train_cfg.learning_rate = args.lr
    if args.resume:
        train_cfg.resume_from = args.resume
    train_cfg.seed = args.seed

    if args.preprocess_dir:
        data_cfg.preprocessed_dir = args.preprocess_dir

    # 归一化统计: 如果指定了 norm_stats，优先使用；否则在训练时计算
    if args.norm_stats:
        data_cfg.norm_stats_path = args.norm_stats
    else:
        data_cfg.norm_stats_path = os.path.join(args.work_dir, "norm_stats.pt")

    logger.info("构建数据加载器...")
    train_loader, val_loader, test_loader, norm_mean, norm_std = build_dataloaders(
        data_cfg, train_cfg
    )
    logger.info(f"训练集: {len(train_loader.dataset)} 样本, {len(train_loader)} batches")
    logger.info(f"验证集: {len(val_loader.dataset)} 样本")

    logger.info("构建 CFM 模型...")
    model = ERA5FlowMatchingModel(model_cfg, data_cfg, train_cfg)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"模型参数: {total_params/1e6:.2f}M (可训练: {trainable_params/1e6:.2f}M)")

    trainer = CFMTrainer(model, train_loader, val_loader, train_cfg, data_cfg, args.work_dir)
    trainer.train()

    logger.info("=" * 60)
    logger.info("在测试集上评估 (自回归 24步 → 72h, CFM Euler采样)...")

    best_ckpt_path = os.path.join(args.work_dir, train_cfg.checkpoint_dir, "best.pt")
    if not os.path.exists(best_ckpt_path):
        logger.warning(f"找不到 best checkpoint: {best_ckpt_path}, 跳过测试集评估")
    else:
        device = trainer.device
        eval_model = ERA5FlowMatchingModel(model_cfg, data_cfg, train_cfg).to(device)
        ckpt = torch.load(best_ckpt_path, map_location=device, weights_only=False)

        if "ema_state_dict" in ckpt:
            eval_model.load_state_dict(ckpt["model_state_dict"])
            ema = EMA(eval_model, decay=train_cfg.ema_decay)
            ema.load_state_dict(ckpt["ema_state_dict"])
            ema.apply_shadow(eval_model)
            logger.info("已加载 best checkpoint 的 EMA 参数")
        else:
            eval_model.load_state_dict(ckpt["model_state_dict"])
            logger.info("已加载 best checkpoint (无 EMA)")

        eval_model.eval()

        test_dataset = test_loader.dataset
        evaluate_on_test(
            eval_model, test_dataset, data_cfg, infer_cfg,
            norm_mean, norm_std, device, num_samples=args.test_samples,
        )
        logger.info("测试集评估完成!")


if __name__ == "__main__":
    main()
