"""
Flow Matching 自回归评估脚本
与 Diffusion 的 evaluate_multi.py 对应的评估脚本

使用方法:
    python eval_flow_matching.py \
        --checkpoint checkpoints/best.pt \
        --era5_dir ../preprocessed_9ch_40x40 \
        --output_dir eval_results_fm
"""
import os
import sys
import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch

# 添加项目路径
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(SCRIPT_DIR))

from flow_matching.configs.config import DataConfig, ModelConfig, InferenceConfig
from flow_matching.models.flow_matching_model import ERA5FlowMatchingModel
from flow_matching.inference import CFMInferencer, CFMPredictor

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# 变量名称定义
VAR_NAMES = [
    'u_850', 'u_500', 'u_250',
    'v_850', 'v_500', 'v_250',
    'z_850', 'z_500', 'z_250',
]


def load_model(checkpoint_path: str, device: torch.device):
    """加载 Flow Matching 模型"""
    logger.info(f"加载模型: {checkpoint_path}")

    data_cfg = DataConfig()
    model_cfg = ModelConfig()

    model = ERA5FlowMatchingModel(model_cfg, data_cfg, None)

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    if 'ema_state_dict' in ckpt:
        from flow_matching.train_preprocessed import EMA
        ema = EMA(model, decay=0.9999)
        ema.load_state_dict(ckpt['ema_state_dict'])
        ema.apply_shadow(model)
        logger.info(f"已加载 EMA 参数")
    elif 'model_state_dict' in ckpt:
        model.load_state_dict(ckpt['model_state_dict'])

    model.eval()
    return model, data_cfg


def load_preprocessed_data(data_cfg: DataConfig):
    """加载预处理数据"""
    logger.info("加载预处理数据...")

    # 归一化统计
    norm_stats_path = data_cfg.norm_stats_path
    if os.path.exists(norm_stats_path):
        stats = torch.load(norm_stats_path, weights_only=True, map_location='cpu')
        norm_mean = stats['mean'].numpy()
        norm_std = stats['std'].numpy()
    else:
        logger.warning(f"未找到归一化统计: {norm_stats_path}")
        norm_mean = np.zeros(9)
        norm_std = np.ones(9)

    # 导入数据加载模块
    sys.path.insert(0, SCRIPT_DIR)
    from train_preprocessed import PreprocessedCFMData, create_preprocessed_dataloaders

    # 创建临时训练配置
    from dataclasses import dataclass
    @dataclass
    class TempTrainConfig:
        batch_size: int = 16
        gradient_accumulation_steps: int = 1
        max_epochs: int = 120           # 三阶段需要完整训练
        learning_rate: float = 2e-4     # Phase 1-2 基础学习率
        weight_decay: float = 0.01
        betas: tuple = (0.9, 0.999)
        warmup_steps: int = 200        # 热启动步数
        warmup_start_lr: float = 1e-6
        min_lr: float = 1e-5            # Phase 3 最低学习率
        use_amp: bool = True
        amp_dtype: str = "float16"
        ema_decay: float = 0.999
        ema_start_step: int = 0
        max_grad_norm: float = 1.0
        physics_loss_weight: float = 1.0       # 启用物理损失调度
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
        velocity_clamp: tuple = (-5.0, 5.0)   # 限制速度场范围
        physics_warmup_steps: int = 10000
        physics_warmup_type: str = "linear"    # linear | cosine
        physics_target_weight: float = 0.1    # 最终物理权重 (0.1 而非 1.0)
        x0_loss_weight: float = 0.5          # 直接 x0 监督
        path_perturb_prob: float = 0.3         # Phase 2 路径扰动概率
        path_perturb_sigma: float = 0.05      # 扰动强度
        physics_warmup_start_epoch: int = 20  # Phase 2 开始引入物理损失
        physics_warmup_end_epoch: int = 80    # Phase 3 锁定物理权重
        phase3_lr_start_epoch: int = 80       # Phase 3 学习率退火起点

    train_cfg = TempTrainConfig()

    try:
        train_loader, val_loader, test_loader, _ = create_preprocessed_dataloaders(data_cfg, train_cfg)
        logger.info(f"数据加载成功: train={len(train_loader.dataset)}, val={len(val_loader.dataset)}, test={len(test_loader.dataset)}")
        return train_loader, val_loader, test_loader, norm_mean, norm_std
    except Exception as e:
        logger.error(f"加载数据失败: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None, norm_mean, norm_std


def denormalize_field(data_norm, mean, std):
    """(C, H, W) numpy 反归一化"""
    std = np.where(std < 1e-8, 1.0, std)
    return data_norm * std.reshape(-1, 1, 1) + mean.reshape(-1, 1, 1)


def evaluate_autoregressive(model, data_cfg, test_loader, device, num_steps=24, num_typhoons=20,
                            use_noise_reset=True, euler_steps=8, ensemble_k=3):
    """
    自回归评估

    参数:
        model: Flow Matching 模型
        data_cfg: 数据配置
        test_loader: 测试数据加载器
        device: 设备
        num_steps: 自回归步数
        num_typhoons: 评估的台风数量
        use_noise_reset: 是否使用 noise-reset 策略 (避免 x_t 分布偏移)
        euler_steps: Euler 采样步数 (推荐 8，从默认的 4 提升)
        ensemble_k: 每步集成采样次数

    返回:
        all_rmse: List of (T, C) arrays
        all_tids: List of typhoon ids
    """
    infer_cfg = InferenceConfig()
    infer_cfg.autoregressive_steps = num_steps
    infer_cfg.euler_steps = euler_steps
    infer_cfg.euler_mode = 'midpoint'

    inferencer = CFMInferencer(model, data_cfg, infer_cfg, device=device)

    # 归一化参数
    norm_stats_path = data_cfg.norm_stats_path
    if os.path.exists(norm_stats_path):
        stats = torch.load(norm_stats_path, weights_only=True, map_location=device)
        norm_mean = stats['mean'].numpy()
        norm_std = stats['std'].numpy()
    else:
        norm_mean = np.zeros(9)
        norm_std = np.ones(9)

    C = data_cfg.era5_channels  # 9
    T_hist = data_cfg.history_steps  # 16

    # 收集台风 ID
    typhoon_samples = {}
    for i, batch in enumerate(test_loader):
        if i >= num_typhoons:
            break

        condition = batch['condition']  # (B, T, C, H, W)
        target = batch['target']  # (B, T, C, H, W)
        typhoon_ids = batch.get('typhoon_id', [f'typoon_{i}'] * condition.shape[0])

        for j in range(min(condition.shape[0], 3)):  # 每个 typhoon 取几个样本
            tid = typhoon_ids[j] if j < len(typhoon_ids) else f'typoon_{i}'
            if tid not in typhoon_samples:
                typhoon_samples[tid] = []
            typhoon_samples[tid].append({
                'condition': condition[j].cpu(),
                'target': target[j].cpu(),
                'start_idx': i * test_loader.batch_size + j
            })

    logger.info(f"评估 {len(typhoon_samples)} 个台风")
    logger.info(f"推理配置: noise_reset={use_noise_reset}, euler_steps={euler_steps}, ensemble_k={ensemble_k}")

    # 自回归推理
    all_rmse = []
    all_tids = []

    for tid, samples in list(typhoon_samples.items())[:num_typhoons]:
        for sample_idx, sample in enumerate(samples):
            cond = sample['condition'].unsqueeze(0).to(device)  # (1, T, C, H, W)

            logger.info(f"推理: {tid} (样本 {sample_idx})")

            try:
                if use_noise_reset:
                    # ===== Noise-Reset 策略 =====
                    # 避免 x_t 分布偏移导致预测发散
                    # 每步从 N(0,I) 采样，配合集成平均
                    current_condition = cond.clone()
                    B, T, C, H, W = current_condition.shape

                    preds = []
                    for step in range(num_steps):
                        step_preds = []
                        for k in range(ensemble_k):
                            torch.manual_seed(42 + step * 100 + k)
                            x_start = torch.randn(B, C, H, W, device=device)
                            x0 = inferencer._sample_single(x_start, current_condition)
                            x0 = torch.clamp(x0, -5.0, 5.0)

                            # z delta clamp
                            if step > 0 and inferencer.z_channel_indices:
                                z_new = x0[:, inferencer.z_channel_indices]
                                z_prev = preds[-1][:, inferencer.z_channel_indices] if preds else torch.zeros_like(z_new)
                                z_delta = z_new - z_prev
                                z_delta_clamped = z_delta.clamp(-0.5, 0.5)
                                x0 = x0.clone()
                                x0[:, inferencer.z_channel_indices] = z_prev + z_delta_clamped
                                x0[:, inferencer.z_channel_indices] = x0[:, inferencer.z_channel_indices].clamp(
                                    *inferencer.z_clamp_range)

                            step_preds.append(x0)

                        # 集成平均
                        x0_ensemble = torch.stack(step_preds).mean(0)
                        preds.append(x0_ensemble[0])  # List of (C, H, W)

                        # 滑动窗口更新
                        if T > 1:
                            current_condition = torch.cat([
                                current_condition[:, 1:, :, :, :],
                                x0_ensemble.unsqueeze(1)
                            ], dim=1)
                else:
                    with torch.no_grad():
                        preds = inferencer.predict_autoregressive(
                            cond, num_steps=num_steps, noise_sigma=0.0
                        )
            except Exception as e:
                logger.error(f"推理失败: {e}")
                import traceback
                traceback.print_exc()
                continue

            # 收集真值并计算 RMSE
            rmse = np.full((num_steps, C), np.nan)
            n_valid = 0

            target_seq = sample['target']  # (T, C, H, W)

            for t in range(num_steps):
                gt_idx = t
                if gt_idx >= target_seq.shape[0]:
                    break

                pred_phys = denormalize_field(preds[t][0].cpu().numpy()[:C], norm_mean[:C], norm_std[:C])
                gt_phys = denormalize_field(target_seq[gt_idx][:C].numpy(), norm_mean[:C], norm_std[:C])

                for v in range(C):
                    rmse[t, v] = np.sqrt(np.mean((pred_phys[v] - gt_phys[v]) ** 2))
                n_valid += 1

            if n_valid > 0:
                all_rmse.append(rmse)
                all_tids.append(f"{tid}_{sample_idx}")
                logger.info(f"  有效步: {n_valid}/{num_steps}, "
                            f"3h RMSE u850={rmse[0,0]:.2f}, {n_valid*3}h RMSE u850={rmse[min(num_steps-1, n_valid-1),0]:.2f}")

    return all_rmse, all_tids


def main():
    parser = argparse.ArgumentParser(description="Flow Matching 自回归评估")
    parser.add_argument("--checkpoint", type=str, required=True, help="模型 checkpoint 路径")
    parser.add_argument("--era5_dir", type=str, default="../preprocessed_9ch_40x40",
                        help="ERA5 预处理数据目录")
    parser.add_argument("--csv_path", type=str,
                        default="../VER3_original/VER3/Trajectory/processed_typhoon_tracks.csv",
                        help="台风轨迹 CSV 路径")
    parser.add_argument("--output_dir", type=str, default="eval_results_fm",
                        help="输出目录")
    parser.add_argument("--num_typhoons", type=int, default=20, help="评估的台风数量")
    parser.add_argument("--num_ar_steps", type=int, default=24, help="自回归步数")
    parser.add_argument("--use_noise_reset", action="store_true", default=True,
                        help="使用 noise-reset 策略 (避免 x_t 分布偏移)")
    parser.add_argument("--no_noise_reset", action="store_true",
                        help="禁用 noise-reset，使用原始 AR")
    parser.add_argument("--euler_steps", type=int, default=8,
                        help="Euler/Midpoint 采样步数 (默认 8)")
    parser.add_argument("--ensemble_k", type=int, default=3,
                        help="每步集成采样次数 (默认 3)")
    parser.add_argument("--force_three_stage", action="store_true",
                        help="强制使用三阶段训练配置 (physics + x0 + path perturb)")
    args = parser.parse_args()

    # noise_reset: 命令行 --no_noise_reset 优先
    use_noise_reset = not args.no_noise_reset

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    # 1. 加载模型
    model, data_cfg = load_model(args.checkpoint, device)

    # 更新配置路径
    if not os.path.exists(data_cfg.norm_stats_path):
        data_cfg.norm_stats_path = os.path.join(args.era5_dir, "norm_stats.pt")
    if not os.path.exists(data_cfg.csv_path):
        data_cfg.csv_path = args.csv_path
    if not os.path.exists(data_cfg.era5_dir):
        data_cfg.era5_dir = args.era5_dir

    # 2. 加载数据
    _, _, test_loader, norm_mean, norm_std = load_preprocessed_data(data_cfg)

    if test_loader is None:
        logger.error("无法加载测试数据")
        return

    # 3. 评估
    logger.info(f"开始自回归评估: {args.num_ar_steps} 步")
    all_rmse, all_tids = evaluate_autoregressive(
        model, data_cfg, test_loader, device,
        num_steps=args.num_ar_steps, num_typhoons=args.num_typhoons,
        use_noise_reset=use_noise_reset,
        euler_steps=args.euler_steps,
        ensemble_k=args.ensemble_k,
    )

    if not all_rmse:
        logger.error("没有有效结果!")
        return

    # 4. 汇总统计
    stacked = np.stack(all_rmse, axis=0)  # (N, T, C)
    mean_rmse = np.nanmean(stacked, axis=0)  # (T, C)
    median_rmse = np.nanmedian(stacked, axis=0)  # (T, C)
    count = np.sum(~np.isnan(stacked[:, :, 0]), axis=0)  # (T,)

    hours = np.array([(t+1)*3 for t in range(args.num_ar_steps)])

    # 打印 Mean RMSE
    print(f"\n{'='*100}")
    print(f"  [Flow Matching Mean RMSE] {len(all_rmse)} typhoons, {args.num_ar_steps} steps")
    print(f"{'='*100}")
    print(f"{'时效':<8}", end="")
    for vn in VAR_NAMES:
        print(f" {vn:>8}", end="")
    print()
    print("-" * (8 + 9 * 9))

    for t in range(args.num_ar_steps):
        if count[t] == 0:
            continue
        print(f"+{(t+1)*3:>3}h    ", end="")
        for v in range(9):
            print(f" {mean_rmse[t, v]:>8.2f}", end="")
        print()

    print(f"\n{'平均':>7} ", end="")
    for v in range(9):
        print(f" {np.nanmean(mean_rmse[:, v]):>8.2f}", end="")
    print()

    # 打印 Median RMSE
    print(f"\n{'='*100}")
    print(f"  [Flow Matching Median RMSE]")
    print(f"{'='*100}")
    print(f"{'时效':<8}", end="")
    for vn in VAR_NAMES:
        print(f" {vn:>8}", end="")
    print()
    print("-" * (8 + 9 * 9))

    for t in range(args.num_ar_steps):
        if count[t] == 0:
            continue
        print(f"+{(t+1)*3:>3}h    ", end="")
        for v in range(9):
            print(f" {median_rmse[t, v]:>8.2f}", end="")
        print()

    # 5. 保存 CSV
    csv_path_mean = os.path.join(args.output_dir, "rmse_mean.csv")
    with open(csv_path_mean, 'w', encoding='utf-8') as f:
        f.write("lead_time_h," + ",".join(VAR_NAMES) + "\n")
        for t in range(args.num_ar_steps):
            if count[t] == 0:
                continue
            f.write(f"{(t+1)*3}")
            for v in range(9):
                f.write(f",{mean_rmse[t, v]:.4f}")
            f.write("\n")
    logger.info(f"Mean CSV: {csv_path_mean}")

    csv_path_median = os.path.join(args.output_dir, "rmse_median.csv")
    with open(csv_path_median, 'w', encoding='utf-8') as f:
        f.write("lead_time_h," + ",".join(VAR_NAMES) + "\n")
        for t in range(args.num_ar_steps):
            if count[t] == 0:
                continue
            f.write(f"{(t+1)*3}")
            for v in range(9):
                f.write(f",{median_rmse[t, v]:.4f}")
            f.write("\n")
    logger.info(f"Median CSV: {csv_path_median}")

    # 6. 绘图
    valid = count > 0

    # 图1: RMSE vs Lead Time
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for v in range(6):
        axes[0].plot(hours[valid], mean_rmse[valid, v], marker='o', markersize=3,
                     label=VAR_NAMES[v], linewidth=1.5)
    axes[0].set_xlabel('Lead time (hours)')
    axes[0].set_ylabel('RMSE (m/s)')
    axes[0].set_title('Flow Matching - Wind RMSE')
    axes[0].legend(fontsize=7, ncol=2)
    axes[0].grid(True, alpha=0.3)

    for v in range(6, 9):
        axes[1].plot(hours[valid], mean_rmse[valid, v], marker='s', markersize=3,
                     label=VAR_NAMES[v], linewidth=1.5)
    axes[1].set_xlabel('Lead time (hours)')
    axes[1].set_ylabel(u'RMSE (m²/s²)')
    axes[1].set_title('Flow Matching - Geopotential RMSE')
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(args.output_dir, "rmse_vs_leadtime.png")
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved: {save_path}")

    logger.info(f"\n评估完成! 结果保存在: {args.output_dir}/")


if __name__ == "__main__":
    main()
