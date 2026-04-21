"""
Flow Matching Z通道专项优化训练脚本
针对 z_850 表现最差问题，三管齐下:

方案F (通道权重优化):
  - z_850 权重从 1.0 → 3.0 (RMSE最高的通道给最高权重)
  - z_500 权重从 1.2 → 2.5
  - z_250 权重保持 2.0
  - UV通道适当降低: u_250/v_250 从 3.0 → 2.5

方案G (提前物理损失 warmup):
  - physics_warmup_start_epoch: 100 → 0 (立即启动)
  - physics_warmup_end_epoch: 160 → 100 (前100个epoch线性增长)
  - 让Z通道从训练一开始就受物理约束

方案I (Z通道数据增强):
  - 训练时对 Z 通道 (channels 6,7,8) 加入轻微高斯噪声 (σ=0.01)
  - 提升模型对 Z 通道的鲁棒性

额外改进:
  - 训练轮数: 200 epoch (原120)
  - z_phys_weight: 2.0 (Z通道专属物理约束权重)
  - 学习率调度: 前160 epoch保持2e-4，后40 epoch余弦退火至1e-5
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
from flow_matching.train_preprocessed import EMA, create_preprocessed_dataloaders, CFMTrainer
from flow_matching.models.improved_losses import GeostrophicBalanceLoss

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def apply_z_optimized_config(train_cfg: TrainConfig) -> TrainConfig:
    """应用Z通道优化配置 (方案F+G+I)"""
    improvements = {
        # ===== 方案F: Z通道权重大幅提升 =====
        # 顺序: u_850,u_500,u_250, v_850,v_500,v_250, z_850,z_500,z_250
        'channel_weights': (
            2.0, 2.0, 2.0,   # u_850, u_500, u_250 (降低2.5→2.0)
            2.0, 2.0, 2.0,   # v_850, v_500, v_250 (降低2.5→2.0)
            3.0, 2.5, 2.0,   # z_850↑(1.0→3.0), z_500↑(1.2→2.5), z_250(保持2.0)
        ),
        # ===== 方案G: 物理损失提前启动 =====
        'physics_warmup_start_epoch': 0,   # 立即启动物理损失
        'physics_warmup_end_epoch': 100,   # 前100个epoch线性增长到目标权重
        'physics_target_weight': 0.05,    # 物理损失目标权重
        # ===== 新增: Z通道专属物理约束权重 =====
        'z_phys_weight': 2.0,  # Z通道的物理约束放大2倍
        # ===== 方案I: 条件噪声增强 (对Z通道更有效) =====
        'condition_noise_sigma': 0.35,
        'condition_noise_prob': 0.6,
        # ===== 其他优化 =====
        'use_geostrophic_physics': True,  # 启用地转平衡
        'geostrophic_weight': 0.05,
        'divergence_weight': 0.02,
        'vorticity_weight': 0.05,
        'x0_loss_weight': 0.5,
        'max_epochs': 200,  # 完整训练200 epoch (使用max_epochs而非epochs)
        'checkpoint_dir': 'checkpoints/z_optimized',
        'checkpoint_weights': (0.8, 0.2),
    }

    for key, value in improvements.items():
        if hasattr(train_cfg, key):
            setattr(train_cfg, key, value)
        else:
            logger.warning(f"属性 {key} 不存在于 TrainConfig")

    return train_cfg


def add_z_channel_noise(condition: torch.Tensor, sigma: float = 0.01, prob: float = 0.5) -> torch.Tensor:
    """
    对 Z 通道 (channels 6,7,8) 加入高斯噪声 (方案I)

    Args:
        condition: (B, T, C, H, W) 归一化后的条件序列
        sigma: 噪声标准差 (相对于归一化尺度)
        prob: 应用噪声的概率

    Returns:
        添加噪声后的 condition
    """
    if torch.rand(1).item() > prob:
        return condition

    # Z 通道索引 (假设通道顺序: u,v,z 各3层)
    # condition shape: (B, T, 9, H, W)
    z_indices = [6, 7, 8]
    noise = torch.randn_like(condition[:, :, z_indices])
    condition = condition.clone()
    condition[:, :, z_indices] += sigma * noise
    return condition


class ZOptimizedCFMTrainer(CFMTrainer):
    """Z通道优化版 Trainer"""

    def __init__(self, model, train_loader, val_loader, train_cfg, data_cfg, work_dir):
        # 在父类初始化前，先修正 train_cfg 中的 warmup 字段
        # 这样父类初始化时就会使用正确的值
        train_cfg.physics_warmup_start_epoch = 0
        train_cfg.physics_warmup_end_epoch = 100

        # 调用父类初始化（会使用修正后的 train_cfg）
        super().__init__(model, train_loader, val_loader, train_cfg, data_cfg, work_dir)

        # 覆盖为改进配置
        self.z_phys_weight = getattr(train_cfg, 'z_phys_weight', 2.0)
        self.geostrophic_weight = train_cfg.geostrophic_weight
        self.divergence_weight = train_cfg.divergence_weight
        self.vorticity_weight = train_cfg.vorticity_weight
        self.x0_loss_weight = train_cfg.x0_loss_weight
        self.physics_target_weight = train_cfg.physics_target_weight

        # 初始化地转平衡损失
        if train_cfg.use_geostrophic_physics and train_cfg.geostrophic_weight > 0:
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

        logger.info(f"Z优化配置: z_phys_weight={self.z_phys_weight}")
        logger.info(f"物理损失 warmup: epoch 0 → 100")

    def _train_one_epoch(self):
        """重写训练一个epoch，应用Z通道优化"""
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

            # ===== 方案I: Z通道噪声增强 =====
            condition = add_z_channel_noise(condition, sigma=0.01, prob=0.5)

            # 计算物理权重 (方案G: 从epoch 0开始warmup)
            physics_weight = self._compute_physics_weight()

            with torch.amp.autocast("cuda", dtype=self.amp_dtype, enabled=self.use_amp):
                outputs = self.model(condition, target)

                # 基础损失
                loss_mse = outputs["loss_mse"]
                loss_x0 = outputs["loss_x0"]
                loss_div = outputs["loss_div"]
                loss_sol = outputs["loss_sol"]
                loss_curl = outputs["loss_curl"]

                # 基础损失组合
                base_loss = loss_mse + self.x0_loss_weight * loss_x0 + physics_weight * (loss_div + loss_sol + loss_curl)

                # ===== 地转平衡损失 + Z通道专属权重 =====
                geo_loss = 0.0
                if self.improved_geo_loss is not None and physics_weight > 0:
                    pred = outputs.get('predicted', None)
                    if pred is not None:
                        u_pred = pred[:, 0, :, :]
                        v_pred = pred[:, 3, :, :]
                        z_pred = pred[:, 6, :, :]

                        geo_loss = self.improved_geo_loss(u_pred, v_pred, z_pred)
                        # Z通道专属权重放大
                        base_loss = base_loss + self.geostrophic_weight * geo_loss * self.z_phys_weight

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
                phase = self._get_training_phase()
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
                        z_phys_w=f"{self.z_phys_weight}",
                        phase=phase,
                        lr=f"{lr:.2e}",
                    )
                    if self.writer:
                        self.writer.add_scalar("train/loss_total", accum_loss, self.optim_step)
                        self.writer.add_scalar("train/loss_mse", mse_val, self.optim_step)
                        self.writer.add_scalar("train/loss_x0", x0_val, self.optim_step)
                        self.writer.add_scalar("train/loss_geo", geo_val, self.optim_step)
                        self.writer.add_scalar("train/z_phys_weight", self.z_phys_weight, self.optim_step)
                        self.writer.add_scalar("train/physics_weight", physics_weight, self.optim_step)
                        self.writer.add_scalar("train/lr", lr, self.optim_step)

                    logger.info(
                        f"  step={self.optim_step} | "
                        f"loss={accum_loss:.6f} | mse={mse_val:.6f} | x0={x0_val:.6f} | "
                        f"geo={geo_val:.6f} | phys_w={physics_weight:.3f} | z_phys_w={self.z_phys_weight} | "
                        f"phase={self._get_training_phase()} | lr={lr:.2e}"
                    )

                accum_loss = 0.0

        return total_loss / max(num_batches, 1)


def main():
    torch.set_float32_matmul_precision("high")

    parser = argparse.ArgumentParser(description="Flow Matching Z通道优化训练")
    parser.add_argument("--era5_dir", type=str,
                        default="/root/autodl-tmp/fyp_final/preprocessed_9ch_40x40",
                        help="预处理数据目录")
    parser.add_argument("--csv_path", type=str,
                        default="/root/autodl-tmp/fyp_final/VER3_original/VER3/Trajectory/processed_typhoon_tracks.csv",
                        help="轨迹CSV文件路径")
    parser.add_argument("--work_dir", type=str, default=".")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--max_epochs", type=int, default=None, help="覆盖TrainConfig.max_epochs")
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

    # 应用Z通道优化配置
    train_cfg = apply_z_optimized_config(train_cfg)

    # 覆盖命令行参数
    if args.csv_path:
        data_cfg.csv_path = args.csv_path
    if args.batch_size:
        train_cfg.batch_size = args.batch_size
    if args.max_epochs:
        train_cfg.max_epochs = args.max_epochs
    if args.lr:
        train_cfg.learning_rate = args.lr
    if args.resume:
        train_cfg.resume_from = args.resume
    train_cfg.seed = args.seed

    logger.info("=" * 70)
    logger.info("Flow Matching Z通道优化训练 (方案F+G+I)")
    logger.info("=" * 70)
    logger.info("主要改进:")
    logger.info(f"  方案F - Z通道权重: z_850=3.0↑, z_500=2.5↑, z_250=2.0 (原:1.0,1.2,2.0)")
    logger.info(f"  方案G - 物理损失 warmup: epoch 0 → 100 (原:100→160)")
    logger.info(f"  方案I - Z通道噪声增强: σ=0.01, prob=0.5")
    logger.info(f"  z_phys_weight: {train_cfg.z_phys_weight} (Z专属物理约束放大)")
    logger.info(f"  总训练轮数: {train_cfg.max_epochs} epoch")
    logger.info("=" * 70)

    logger.info("构建数据加载器...")
    train_loader, val_loader, test_loader, norm_mean, norm_std = create_preprocessed_dataloaders(
        data_cfg, train_cfg
    )
    logger.info(f"训练集: {len(train_loader.dataset)} 样本")
    logger.info(f"验证集: {len(val_loader.dataset)} 样本")
    logger.info(f"测试集: {len(test_loader.dataset)} 样本")

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

    # 创建Z优化版 trainer
    trainer = ZOptimizedCFMTrainer(
        model, train_loader, val_loader, train_cfg, data_cfg, args.work_dir
    )
    trainer.train()

    logger.info("Z通道优化训练完成!")


if __name__ == "__main__":
    main()
