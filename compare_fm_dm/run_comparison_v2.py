"""
统一FM/DM对比实验主脚本

修复包导入问题后，此脚本可以正常运行
"""
import os
import sys

# 确保包能被正确导入
current_file = os.path.abspath(__file__)
package_dir = os.path.dirname(current_file)
root_dir = os.path.dirname(package_dir)

# 将项目根目录添加到路径
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)
if package_dir not in sys.path:
    sys.path.insert(0, package_dir)

# 现在可以安全导入
import json
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from configs import (
    DataConfig, ModelConfig, TrainConfig, InferenceConfig,
    ComparisonConfig, get_config, get_comparison_config
)
from data.dataset import ERA5Dataset, build_dataloaders
from models.unified_model import UnifiedModel, create_model
from models.trainer import UnifiedTrainer, EMA
from models.adapter import load_newtry_checkpoint, AdaptedDiffusionModel
from evaluation.metrics import (
    ComparisonEvaluator,
    compute_rmse, compute_mae, compute_lat_weighted_rmse,
    compute_acc, compute_channel_bias,
    compute_2d_psd, compute_kinetic_energy_spectrum,
    compute_spectral_slope,
    compute_divergence, compute_divergence_rmse,
    compute_geostrophic_balance, compute_vorticity,
    compute_temporal_coherence,
    compute_nfe_efficiency,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ============================================================
# 辅助函数
# ============================================================

def load_checkpoint(
    model: UnifiedModel,
    checkpoint_path: str,
    device: torch.device,
    use_ema: bool = True,
) -> Tuple[UnifiedModel, bool]:
    """加载模型checkpoint"""
    if not os.path.exists(checkpoint_path):
        return model, False

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = ckpt["model_state_dict"]

    # Handle dit. prefix mismatch between checkpoint and model
    # Try direct load first, then try adding/removing dit. prefix
    missing_keys = []
    unexpected_keys = []

    try:
        model.load_state_dict(state_dict, strict=False)
        logger.info(f"已加载模型参数: {checkpoint_path}")
    except Exception as e:
        logger.warning(f"直接加载失败，尝试添加dit.前缀: {e}")
        new_state_dict = {}
        for k, v in state_dict.items():
            if not k.startswith("dit."):
                new_key = "dit." + k
            else:
                new_key = k
            new_state_dict[new_key] = v
        try:
            model.load_state_dict(new_state_dict, strict=False)
            logger.info(f"已加载模型参数 (添加dit.前缀): {checkpoint_path}")
        except Exception as e2:
            logger.warning(f"添加dit.前缀也失败，尝试移除dit.前缀: {e2}")
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith("dit."):
                    new_key = k[4:]
                else:
                    new_key = k
                new_state_dict[new_key] = v
            model.load_state_dict(new_state_dict, strict=False)
            logger.info(f"已加载模型参数 (移除dit.前缀): {checkpoint_path}")

    if use_ema and "ema_state_dict" in ckpt:
        ema = EMA(model, decay=0.999)
        ema_state = ckpt["ema_state_dict"]
        if "shadow" in ema_state:
            shadow_dict = ema_state["shadow"]
            # Add dit. prefix if needed
            new_shadow = {}
            for k, v in shadow_dict.items():
                if not k.startswith("dit."):
                    new_key = "dit." + k
                else:
                    new_key = k
                new_shadow[new_key] = v
            ema_state = {"shadow": new_shadow}
        try:
            ema.load_state_dict(ema_state)
            ema.apply_shadow(model)
            logger.info(f"已加载 EMA 参数: {checkpoint_path}")
        except Exception as e:
            logger.warning(f"EMA加载失败，跳过EMA: {e}")

    return model, True


def find_ar_sequences(
    dataset: ERA5Dataset,
    ar_steps: int,
    min_lead_time: int = 1,
    max_lead_time: int = 24,
    skip_step: int = 1,
) -> List[List[int]]:
    """在测试集中查找自回归序列"""
    sequences = []
    test_typhoons = list(set([tid for tid, _ in dataset.samples]))

    for tid in test_typhoons:
        indices = [i for i, (t, _) in enumerate(dataset.samples) if t == tid]
        if len(indices) < ar_steps * skip_step:
            continue

        for start_idx in range(0, len(indices) - (ar_steps - 1) * skip_step):
            seq = [indices[start_idx + i * skip_step] for i in range(ar_steps)]
            sequences.append(seq)

    logger.info(f"找到 {len(sequences)} 个自回归序列 (来自 {len(test_typhoons)} 个台风)")
    return sequences


# ============================================================
# 主程序
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="FM vs DM 对比实验")

    # 数据
    parser.add_argument("--data_root", type=str, default="/root/autodl-tmp/fyp_final/Ver4/Trajectory/preprocessed_era5")
    parser.add_argument("--preprocess_dir", type=str, default="/root/autodl-tmp/fyp_final/Ver4/Trajectory/preprocessed_era5")
    parser.add_argument("--norm_stats", type=str, default="/root/autodl-tmp/fyp_final/Ver4/Trajectory/preprocessed_era5/norm_stats.pt")

    # FM 模型
    parser.add_argument("--fm_ckpt", type=str, default="./multi_seed_results/seed_42/checkpoints_fm/best.pt",
                        help="FM checkpoint 路径")

    # DM 模型
    parser.add_argument("--dm_ckpt", type=str, default=None,
                        help="DM checkpoint 路径 (若为空则使用默认训练)")
    parser.add_argument("--external_dm_ckpt", type=str, default=None,
                        help="外部 DM checkpoint (如 newtry 的 best_eps.pt)")

    # 评估
    parser.add_argument("--eval_samples", type=int, default=100)
    parser.add_argument("--ar_steps", type=int, default=12)
    parser.add_argument("--ddim_steps", type=int, default=50)

    # 控制
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--run_fm", action="store_true", default=True)
    parser.add_argument("--run_dm", action="store_true", default=True)
    parser.add_argument("--device", type=str, default="cuda")

    # 输出
    parser.add_argument("--work_dir", type=str, default="./results_newtry_comparison")
    parser.add_argument("--output_dir", type=str, default="./results_newtry_comparison/figures")

    args = parser.parse_args()
    os.makedirs(args.work_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logger.info(f"使用设备: {device}")

    # ========== 配置 ==========
    data_cfg = DataConfig(
        data_root=args.data_root,
        era5_dir=args.data_root,
        preprocessed_dir=args.preprocess_dir,
        norm_stats_path=args.norm_stats,
        history_steps=5,
        forecast_steps=1,
        grid_size=40,
        pressure_level_vars=["u", "v", "z"],
        pressure_levels=[850, 500, 250],
        surface_vars=[],
        num_workers=4,
        pin_memory=True,
    )

    model_cfg = ModelConfig(
        in_channels=data_cfg.num_channels,
        cond_channels=data_cfg.condition_channels,
        d_model=384,
        n_heads=6,
        n_dit_layers=12,
        n_cond_layers=3,
        ff_mult=4,
        patch_size=4,
        dropout=0.1,
        prediction_type="eps",
    )

    train_cfg = TrainConfig(
        batch_size=16,
        eval_every=10,
        seed=42,
        use_channel_weights=False,
    )

    infer_cfg = InferenceConfig(
        ddim_steps=args.ddim_steps,
        clamp_range=None,  # was (-5.0, 5.0) — kills z variance
        autoregressive_steps=args.ar_steps,
        autoregressive_noise_sigma=0.05,
        ensemble_size=5,
    )

    # ========== 数据 ==========
    logger.info("构建数据集...")
    _, _, test_loader, norm_mean, norm_std = build_dataloaders(
        data_cfg, train_cfg, norm_mean=None, norm_std=None
    )
    test_dataset = test_loader.dataset
    logger.info(f"测试集大小: {len(test_dataset)}")

    # ========== 加载 FM 模型 ==========
    logger.info(f"加载 FM 模型: {args.fm_ckpt}")
    fm_model = UnifiedModel(model_cfg, data_cfg, train_cfg, method="fm").to(device)
    fm_model, _ = load_checkpoint(fm_model, args.fm_ckpt, device, use_ema=True)

    # ========== 加载 DM 模型 ==========
    if args.external_dm_ckpt:
        logger.info(f"加载外部 DM 模型 (newtry): {args.external_dm_ckpt}")
        dm_model = load_newtry_checkpoint(
            args.external_dm_ckpt, data_cfg, model_cfg, train_cfg, device
        )
    else:
        logger.info(f"加载 DM 模型 (默认): {args.dm_ckpt or '不加载'}")
        dm_model = UnifiedModel(model_cfg, data_cfg, train_cfg, method="dm").to(device)
        if args.dm_ckpt:
            dm_model, _ = load_checkpoint(dm_model, args.dm_ckpt, device)

    # ========== 评估 ==========
    evaluator = ComparisonEvaluator(
        data_cfg=data_cfg,
        norm_mean=norm_mean,
        norm_std=norm_std,
    )

    # 确定性评估
    logger.info("开始确定性评估...")

    # 收集 FM 预测 (使用 Euler 采样)
    fm_predictions = []
    fm_ground_truth = []

    logger.info("生成 FM 预测...")
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="FM 推理", unit="batch", total=min(args.eval_samples, len(test_loader))):
            condition = batch["condition"].to(device)
            target = batch["target"].to(device)

            if condition.shape[0] > 4:
                condition = condition[:4]
                target = target[:4]

            pred = fm_model.sample_fm(condition, device, euler_steps=args.ddim_steps)  # FM uses euler_steps
            fm_predictions.append(pred.cpu())
            fm_ground_truth.append(target.cpu())

            if len(fm_predictions) * pred.shape[0] >= args.eval_samples:
                break

    # 收集 DM 预测 (使用 DDIM 采样)
    dm_predictions = []
    dm_ground_truth = []

    logger.info("生成 DM 预测...")
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="DM 推理", unit="batch", total=min(args.eval_samples, len(test_loader))):
            condition = batch["condition"].to(device)
            target = batch["target"].to(device)

            if condition.shape[0] > 4:
                condition = condition[:4]
                target = target[:4]

            if args.external_dm_ckpt:
                # AdaptedDiffusionModel uses sample()
                pred = dm_model.sample(condition, device, ddim_steps=args.ddim_steps)
            else:
                # UnifiedModel uses sample_dm()
                pred = dm_model.sample_dm(condition, device, ddim_steps=args.ddim_steps)

            dm_predictions.append(pred.cpu())
            dm_ground_truth.append(target.cpu())

            if len(dm_predictions) * pred.shape[0] >= args.eval_samples:
                break

    # 评估
    fm_results = evaluator.evaluate_single(fm_predictions, fm_ground_truth, method_name="FM")
    dm_results = evaluator.evaluate_single(dm_predictions, dm_ground_truth, method_name="DM")

    # 打印对比
    logger.info("=" * 70)
    logger.info("对比结果:")
    logger.info("-" * 70)

    for metric in ["rmse_mean", "lat_weighted_rmse_mean", "mae_mean"]:
        fm_val = fm_results.get(metric, 0)
        dm_val = dm_results.get(metric, 0)
        logger.info(f"{metric:30s}: FM={fm_val:.4f}, DM={dm_val:.4f}, 差值={fm_val-dm_val:+.4f}")

    # 保存结果
    results = {
        "fm": fm_results,
        "dm": dm_results,
        "args": vars(args),
    }
    output_path = os.path.join(args.work_dir, "deterministic_results.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"结果已保存到: {output_path}")

    logger.info("=" * 60)
    logger.info("✅ 对比实验完成!")


if __name__ == "__main__":
    main()
