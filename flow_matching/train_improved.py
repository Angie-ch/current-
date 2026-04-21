"""
Flow Matching 改进版训练脚本
基于 improved_config.py 的配置，针对 RMSE 优化

主要改进:
1. x0_loss_weight: 0.1 → 0.5
2. physics_target_weight: 0.05 → 0.05 (降低，避免干扰主损失)
3. 启用地转平衡损失 (geostrophic_weight=0.05)
4. UV/Z通道分离权重
5. 更长的物理损失 warmup (epoch 80 → 200)
"""
import argparse
import logging
import os
import sys
import numpy as np
import torch
from tqdm import tqdm

# 添加项目路径
ver3_dir = "/root/autodl-tmp/fyp_final/VER3_original/VER3"
fm_dir = ver3_dir + "/flow_matching"
sys.path.insert(0, ver3_dir)
sys.path.insert(0, fm_dir)
sys.path.insert(0, fm_dir + "/configs")
sys.path.insert(0, fm_dir + "/models")

from flow_matching.configs.config import DataConfig, ModelConfig, TrainConfig
from flow_matching.models.flow_matching_model import ERA5FlowMatchingModel
from flow_matching.train_preprocessed import EMA, PhysicalConsistencyLoss, create_preprocessed_dataloaders, CFMTrainer
from flow_matching.models.improved_losses import GeostrophicBalanceLoss

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def apply_improved_config(train_cfg: TrainConfig) -> TrainConfig:
    """将改进配置应用到 TrainConfig"""
    improvements = {
        'x0_loss_weight': 0.5,
        'physics_target_weight': 0.05,  # 降低，避免干扰主MSE损失
        'use_geostrophic_physics': True,
        'geostrophic_weight': 0.05,  # 降低地转平衡权重
        'divergence_weight': 0.02,
        'vorticity_weight': 0.05,
        'physics_warmup_start_epoch': 80,  # 延长warmup，让模型先充分学习MSE
        'physics_warmup_end_epoch': 200,  # 延长到训练结束
        'channel_weights': (
            2.0, 2.0, 3.0,
            2.0, 2.0, 3.0,
            1.0, 1.2, 1.5,
        ),
        'pressure_level_weights': (1.0, 1.2, 1.5),
        'condition_noise_sigma': 0.35,
        'condition_noise_prob': 0.6,
        'checkpoint_dir': 'checkpoints/improved_v1',
        'checkpoint_weights': (0.8, 0.2),
    }
    
    for key, value in improvements.items():
        if hasattr(train_cfg, key):
            setattr(train_cfg, key, value)
        else:
            logger.warning(f"属性 {key} 不存在于 TrainConfig")
    
    return train_cfg


class ImprovedCFMTrainer(CFMTrainer):
    """
    改进版 CFM Trainer

    在原有 CFMTrainer 基础上应用:
    1. x0_loss_weight: 0.1 → 0.5
    2. physics_target_weight: 0.05 (降低，避免loss跳升)
    3. 启用地转平衡损失 (geostrophic_weight=0.05)
    4. 更长的物理损失 warmup (epoch 80 → 200)
    """
    
    def __init__(self, model, train_loader, val_loader, train_cfg, data_cfg, work_dir):
        # 先调用父类初始化
        super().__init__(model, train_loader, val_loader, train_cfg, data_cfg, work_dir)
        
        # 覆盖为改进配置
        self.use_geostrophic_physics = train_cfg.use_geostrophic_physics
        self.geostrophic_weight = train_cfg.geostrophic_weight
        self.divergence_weight = train_cfg.divergence_weight
        self.vorticity_weight = train_cfg.vorticity_weight
        self.x0_loss_weight = train_cfg.x0_loss_weight
        self.physics_target_weight = train_cfg.physics_target_weight
        
        # 初始化改进版地转平衡损失
        if self.use_geostrophic_physics:
            lat_deg = np.linspace(data_cfg.lat_range[1], data_cfg.lat_range[0], data_cfg.grid_size)
            lon_deg = np.linspace(data_cfg.lon_range[0], data_cfg.lon_range[1], data_cfg.grid_size)
            lat_rad = np.deg2rad(lat_deg)[:, None]
            lon_grid = np.deg2rad(lon_deg)[None, :]
            lat_grid = np.repeat(lat_rad, data_cfg.grid_size, axis=1)
            lat_grid_tensor = torch.tensor(lat_grid, dtype=torch.float32, device=self.device)
            
            self.improved_geo_loss = GeostrophicBalanceLoss(
                lat_grid=lat_grid_tensor,
                lon_res=0.25,
                lat_res=0.25,
                weight=1.0,
            ).to(self.device)
            
            logger.info(f"地转平衡损失已启用 (weight={self.geostrophic_weight})")
        else:
            self.improved_geo_loss = None
        
        logger.info(f"改进版损失权重: x0={self.x0_loss_weight}, physics={self.physics_target_weight}")
        logger.info(f"物理损失启动: epoch {train_cfg.physics_warmup_start_epoch} → {train_cfg.physics_warmup_end_epoch}")

    def _train_one_epoch(self):
        """重写训练一个epoch，应用改进版损失"""
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

            # 计算物理权重
            physics_weight = self._compute_physics_weight()
            phase = self._get_training_phase()

            with torch.amp.autocast("cuda", dtype=self.amp_dtype, enabled=self.use_amp):
                outputs = self.model(condition, target)

                # 基础损失 - 改进版权重
                loss_mse = outputs["loss_mse"]
                loss_x0 = outputs["loss_x0"]
                loss_div = outputs["loss_div"]
                loss_sol = outputs["loss_sol"]
                loss_curl = outputs["loss_curl"]
                
                # 改进版损失组合
                base_loss = loss_mse + self.x0_loss_weight * loss_x0 + physics_weight * (loss_div + loss_sol + loss_curl)
                
                # 地转平衡损失
                geo_loss = 0.0
                if self.use_geostrophic_physics and self.improved_geo_loss is not None and physics_weight > 0:
                    pred = outputs.get('predicted', None)
                    if pred is not None:
                        u_pred = pred[:, 0, :, :]
                        v_pred = pred[:, 3, :, :]
                        z_pred = pred[:, 6, :, :]
                        
                        geo_loss = self.improved_geo_loss(u_pred, v_pred, z_pred)
                        base_loss = base_loss + self.geostrophic_weight * geo_loss
                
                loss = base_loss / self.cfg.gradient_accumulation_steps

            # 安全检测
            loss_val = loss.item()
            if not torch.isfinite(loss) or loss_val > 10.0:
                logger.warning(f"[Step {self.global_step}] 异常 loss={loss_val:.4f}, 跳过此 batch")
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
                    geo_val = geo_loss.item() if geo_loss > 0 else 0.0
                    batch_pbar.set_postfix(
                        loss=f"{accum_loss:.4f}",
                        mse=f"{mse_val:.4f}",
                        x0=f"{x0_val:.4f}",
                        geo=f"{geo_val:.4f}",
                        phys_w=f"{physics_weight:.3f}",
                        phase=phase,
                        lr=f"{lr:.2e}",
                    )
                    if self.writer:
                        self.writer.add_scalar("train/loss_total", accum_loss, self.optim_step)
                        self.writer.add_scalar("train/loss_mse", mse_val, self.optim_step)
                        self.writer.add_scalar("train/loss_x0", x0_val, self.optim_step)
                        self.writer.add_scalar("train/loss_geo", geo_val, self.optim_step)
                        self.writer.add_scalar("train/physics_weight", physics_weight, self.optim_step)
                        self.writer.add_scalar("train/lr", lr, self.optim_step)
                        self.writer.add_scalar("train/phase", phase, self.optim_step)

                    logger.info(
                        f"  step={self.optim_step} | "
                        f"loss={accum_loss:.6f} | "
                        f"mse={mse_val:.6f} | x0={x0_val:.6f} | "
                        f"geo={geo_val:.6f} | phys_w={physics_weight:.3f} | "
                        f"phase={phase} | lr={lr:.2e}"
                    )

                accum_loss = 0.0

        return total_loss / max(num_batches, 1)


def main():
    torch.set_float32_matmul_precision("high")

    parser = argparse.ArgumentParser(description="Flow Matching 改进版训练")
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

    # 获取基础配置
    data_cfg = DataConfig()
    model_cfg = ModelConfig()
    train_cfg = TrainConfig()
    
    # 应用改进配置
    train_cfg = apply_improved_config(train_cfg)
    
    # 覆盖命令行参数
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

    logger.info("=" * 70)
    logger.info("Flow Matching 改进版训练 (RMSE优化)")
    logger.info("=" * 70)
    logger.info("主要改进:")
    logger.info(f"  x0_loss_weight: 0.1 → {train_cfg.x0_loss_weight}")
    logger.info(f"  physics_target_weight: → {train_cfg.physics_target_weight}")
    logger.info(f"  use_geostrophic_physics: False → {train_cfg.use_geostrophic_physics}")
    logger.info(f"  geostrophic_weight: 0.0 → {train_cfg.geostrophic_weight}")
    logger.info(f"  physics_warmup_start: → {train_cfg.physics_warmup_start_epoch}")
    logger.info(f"  physics_warmup_end: → {train_cfg.physics_warmup_end_epoch}")
    logger.info("=" * 70)

    logger.info("构建数据加载器...")
    train_loader, val_loader, test_loader, norm_mean, norm_std = create_preprocessed_dataloaders(
        data_cfg, train_cfg
    )
    logger.info(f"训练集: {len(train_loader.dataset)} 样本")
    logger.info(f"验证集: {len(val_loader.dataset)} 样本")

    # 检测通道数
    sample_batch = next(iter(train_loader))
    cond_shape = sample_batch['condition'].shape
    if len(cond_shape) == 4:
        era5_channels = cond_shape[1]
    elif len(cond_shape) == 5:
        era5_channels = cond_shape[2]
    else:
        raise ValueError(f"未知 condition shape: {cond_shape}")
    logger.info(f"ERA5 channels: {era5_channels}")

    logger.info("构建 CFM 模型...")
    model_cfg.in_channels = era5_channels
    model_cfg.cond_channels = era5_channels

    model = ERA5FlowMatchingModel(model_cfg, data_cfg, train_cfg)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"模型参数: {total_params/1e6:.2f}M (可训练: {trainable_params/1e6:.2f}M)")

    # 创建改进版 trainer
    trainer = ImprovedCFMTrainer(
        model, train_loader, val_loader, train_cfg, data_cfg, args.work_dir
    )
    trainer.train()

    logger.info("改进版训练完成!")


if __name__ == "__main__":
    main()
