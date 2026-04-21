"""
Flow Matching 多步自回归预测测试脚本

使用方法:
    # 单步预测
    python inference_test.py --mode single --checkpoint checkpoints/best.pt
    
    # 自回归24步预测
    python inference_test.py --mode autoregressive --checkpoint checkpoints/best.pt --num_steps 24
    
    # 带评估
    python inference_test.py --mode autoregressive --checkpoint checkpoints/best.pt --eval
"""
import os
import sys
import argparse
import logging
from pathlib import Path

import numpy as np
import torch

# 添加 flow_matching 目录到 path
FLOW_MATCHING_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(FLOW_MATCHING_DIR)
sys.path.insert(0, PARENT_DIR)

from flow_matching.configs.config import DataConfig, ModelConfig, InferenceConfig
from flow_matching.models.flow_matching_model import ERA5FlowMatchingModel
from flow_matching.inference import CFMInferencer, CFMPredictor, compute_channel_rmse

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def load_model(checkpoint_path: str, device: torch.device):
    """加载 Flow Matching 模型"""
    logger.info(f"加载模型: {checkpoint_path}")
    
    # 加载配置
    data_cfg = DataConfig()
    model_cfg = ModelConfig()
    train_cfg = None  # 训练配置不影响推理
    
    # 创建模型
    model = ERA5FlowMatchingModel(model_cfg, data_cfg, train_cfg)
    
    # 加载 checkpoint
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # 尝试加载 EMA 参数
    if 'ema_state_dict' in ckpt:
        from flow_matching.train_preprocessed import EMA
        ema = EMA(model, decay=0.9999)
        ema.load_state_dict(ckpt['ema_state_dict'])
        ema.apply_shadow(model)
        logger.info(f"已加载 EMA 参数 (epoch {ckpt.get('epoch', '?')})")
    elif 'model_state_dict' in ckpt:
        model.load_state_dict(ckpt['model_state_dict'])
        logger.info(f"已加载模型参数 (epoch {ckpt.get('epoch', '?')})")
    else:
        model.load_state_dict(ckpt)
    
    model.eval()
    return model, data_cfg


def load_test_data(data_cfg: DataConfig, num_samples: int = 4):
    """加载测试数据"""
    logger.info("加载测试数据...")
    
    # 归一化统计
    norm_stats_path = data_cfg.norm_stats_path
    if os.path.exists(norm_stats_path):
        stats = torch.load(norm_stats_path, weights_only=True, map_location='cpu')
        norm_mean = stats['mean'].numpy()
        norm_std = stats['std'].numpy()
    else:
        logger.warning(f"未找到归一化统计: {norm_stats_path}")
        norm_mean = None
        norm_std = None
    
    # 使用训练数据加载器
    sys.path.insert(0, FLOW_MATCHING_DIR)
    from train_preprocessed import create_preprocessed_dataloaders, PreprocessedCFMData
    
    # 创建临时配置
    from dataclasses import dataclass
    
    @dataclass
    class TempTrainConfig:
        batch_size: int = 4
        gradient_accumulation_steps: int = 1
        max_epochs: int = 1
        learning_rate: float = 1e-4
        weight_decay: float = 0.01
        betas: tuple = (0.9, 0.999)
        warmup_steps: int = 100
        warmup_start_lr: float = 1e-6
        min_lr: float = 1e-6
        use_amp: bool = False
        amp_dtype: str = "float16"
        ema_decay: float = 0.999
        ema_start_step: int = 0
        max_grad_norm: float = 1.0
        physics_loss_weight: float = 0.0
        vorticity_loss_weight: float = 0.0
        use_channel_weights: bool = True
        channel_weights: tuple = (1.0,) * 9
        condition_noise_sigma: float = 0.0
        condition_noise_rampup_epochs: int = 100
        condition_noise_prob: float = 0.0
        condition_noise_spatial_smooth: bool = False
        condition_noise_smooth_kernel: int = 5
        eval_every: int = 10
        early_stopping_patience: int = 50
        log_every: int = 20
        use_tensorboard: bool = False
        checkpoint_dir: str = "checkpoints"
        resume_from: str = None
        use_compile: bool = False
        cudnn_benchmark: bool = True
        seed: int = 42
        velocity_loss_scale: float = 1.0
        velocity_clamp: tuple = None
        physics_warmup_steps: int = 10000
        physics_warmup_type: str = "linear"
        physics_target_weight: float = 1.0
        x0_loss_weight: float = 0.0
        path_perturb_prob: float = 0.0
        path_perturb_sigma: float = 0.0
        physics_warmup_start_epoch: int = 20
        physics_warmup_end_epoch: int = 80
        phase3_lr_start_epoch: int = 80
    
    train_cfg = TempTrainConfig()
    
    try:
        train_loader, val_loader, _, _ = create_preprocessed_dataloaders(data_cfg, train_cfg)
        
        # 获取样本
        samples = []
        for i, batch in enumerate(train_loader):
            samples.append(batch)
            if i >= 0:  # 只取第一个 batch
                break
        
        if samples:
            batch = samples[0]
            condition = batch['condition'][:num_samples]
            target = batch['target'][:num_samples]
            logger.info(f"加载了 {condition.shape[0]} 个样本")
            return condition, target, norm_mean, norm_std
    except Exception as e:
        logger.error(f"加载数据失败: {e}")
    
    # Fallback: 生成随机数据
    logger.warning("使用随机测试数据")
    condition = torch.randn(num_samples, 16, 9, 40, 40)
    target = torch.randn(num_samples, 9, 40, 40)
    return condition, target, norm_mean, norm_std


def test_single_step(model, data_cfg, condition, target, device):
    """测试单步预测"""
    logger.info("\n" + "="*60)
    logger.info("测试: 单步预测")
    logger.info("="*60)
    
    infer_cfg = InferenceConfig()
    inferencer = CFMInferencer(model, data_cfg, infer_cfg, device=device)
    
    # 预测
    pred = inferencer.predict_single(condition.to(device))
    
    # 计算 RMSE
    pred = pred.cpu()
    target = target.cpu()
    
    total_rmse = torch.sqrt(((pred - target) ** 2).mean()).item()
    logger.info(f"单步预测 RMSE: {total_rmse:.4f}")
    
    # 各通道 RMSE
    channel_rmses = compute_channel_rmse(pred, target)
    logger.info("各通道 RMSE:")
    for ch, rmse in channel_rmses.items():
        logger.info(f"  {ch}: {rmse:.4f}")
    
    return pred, total_rmse


def test_autoregressive(model, data_cfg, condition, target, num_steps, device):
    """测试自回归多步预测"""
    logger.info("\n" + "="*60)
    logger.info(f"测试: 自回归 {num_steps} 步预测")
    logger.info("="*60)
    
    infer_cfg = InferenceConfig()
    infer_cfg.autoregressive_steps = num_steps
    infer_cfg.euler_steps = 4  # 每步用 4 步 Euler
    infer_cfg.euler_mode = 'midpoint'
    
    inferencer = CFMInferencer(model, data_cfg, infer_cfg, device=device)
    
    # 预测
    preds = inferencer.predict_autoregressive(
        condition.to(device), 
        num_steps=num_steps,
        noise_sigma=0.0
    )
    
    # Stack
    pred_seq = torch.stack(preds, dim=1)  # (B, num_steps, C, H, W)
    
    # 计算 RMSE (只评估第一步，因为后续没有对应 target)
    target_first = target.cpu()[:, :pred_seq.shape[2], :, :] if target.ndim == 4 else target.cpu()
    
    rmse_per_step = []
    for step in range(min(num_steps, pred_seq.shape[1])):
        step_pred = pred_seq[:, step].cpu()
        step_target = target_first[:, step] if step < target_first.shape[1] else step_pred
        rmse = torch.sqrt(((step_pred - step_target) ** 2).mean()).item()
        rmse_per_step.append(rmse)
    
    logger.info(f"\n各步 RMSE:")
    for step, rmse in enumerate(rmse_per_step):
        hours = (step + 1) * 3
        logger.info(f"  +{hours:2d}h: {rmse:.4f}")
    
    mean_rmse = np.mean(rmse_per_step)
    logger.info(f"\n平均 RMSE: {mean_rmse:.4f}")
    
    return pred_seq, rmse_per_step


def main():
    parser = argparse.ArgumentParser(description="Flow Matching 推理测试")
    parser.add_argument("--mode", choices=["single", "autoregressive"], default="autoregressive",
                        help="预测模式")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="模型 checkpoint 路径")
    parser.add_argument("--num_steps", type=int, default=24,
                        help="自回归步数")
    parser.add_argument("--euler_steps", type=int, default=4,
                        help="每步 Euler 积分步数")
    parser.add_argument("--euler_mode", type=str, default="midpoint",
                        choices=["euler", "midpoint", "heun"],
                        help="Euler 模式")
    parser.add_argument("--num_samples", type=int, default=4,
                        help="测试样本数")
    parser.add_argument("--eval", action="store_true",
                        help="与真值评估")
    parser.add_argument("--output_dir", type=str, default="inference_outputs",
                        help="输出目录")
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"设备: {device}")
    
    # 加载模型
    model, data_cfg = load_model(args.checkpoint, device)
    
    # 加载测试数据
    condition, target, norm_mean, norm_std = load_test_data(data_cfg, args.num_samples)
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.mode == "single":
        pred, rmse = test_single_step(model, data_cfg, condition, target, device)
    else:
        pred, rmses = test_autoregressive(
            model, data_cfg, condition, target, args.num_steps, device
        )
    
    logger.info("\n" + "="*60)
    logger.info("测试完成!")
    logger.info("="*60)


if __name__ == "__main__":
    main()
