"""
综合对比实验运行脚本 — Flow Matching vs Diffusion

功能:
1. 训练FM模型和DM模型 (共享骨干)
2. 运行全面对比评估:
   - 确定性精度 (RMSE, ACC, Bias)
   - 频谱分析 (PSD, 动能谱, 谱斜率) — 论文核心
   - 物理一致性 (散度, 地转平衡)
   - 推理效率 (NFE曲线)
   - 台风个例研究
3. 生成论文可视化图表

使用方法:
    python run_comparison.py \
        --data_root /path/to/era5_data \
        --work_dir /path/to/output \
        --run_fm \
        --run_dm \
        --epochs 200 \
        --batch_size 64
"""
import os
import sys
import json
import copy
import time
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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
    model.load_state_dict(ckpt["model_state_dict"], strict=False)

    if use_ema and "ema_state_dict" in ckpt:
        ema = EMA(model, decay=0.999)
        ema.load_state_dict(ckpt["ema_state_dict"])
        ema.apply_shadow(model)
        logger.info(f"已加载 EMA 参数: {checkpoint_path}")
    else:
        logger.info(f"已加载模型参数: {checkpoint_path}")

    return model, True


def find_ar_sequences(
    dataset: ERA5Dataset,
    ar_steps: int,
    num_samples: int,
) -> List[int]:
    """在数据集中找到足够长的连续序列起始点"""
    if len(dataset) == 0:
        return []

    sequences = []
    current_tid = None
    current_start = -1
    current_len = 0

    for idx in range(len(dataset)):
        sample_info = dataset.samples[idx]
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
    for seg_start, seg_len, _ in sequences:
        max_start = seg_start + seg_len - ar_steps
        for idx in range(seg_start, max_start + 1):
            valid_starts.append(idx)
            if len(valid_starts) >= num_samples:
                return valid_starts

    return valid_starts


# ============================================================
# 自回归推理
# ============================================================

@torch.no_grad()
def autoregressive_inference(
    model: UnifiedModel,
    dataset: ERA5Dataset,
    start_indices: List[int],
    num_steps: int,
    device: torch.device,
    method: str,
            clamp_range: Optional[Tuple[float, float]] = None,  # was (-5.0, 5.0)
            z_clamp_range: Optional[Tuple[float, float]] = None,  # was (-1.0, 1.0)
    euler_steps: int = 4,
    ddim_steps: int = 50,
    noise_sigma: float = 0.05,
    infer_cfg = None,
) -> Tuple[List[List[torch.Tensor]], List[List[torch.Tensor]]]:
    """
    自回归多步推理

    Returns:
        (all_preds, all_gts): 每步的预测和真值列表
    """
    if infer_cfg is not None:
        euler_steps = getattr(infer_cfg, 'euler_steps', euler_steps)
        ddim_steps = getattr(infer_cfg, 'ddim_steps', ddim_steps)
        z_clamp_range = getattr(infer_cfg, 'z_clamp_range', z_clamp_range)

    all_preds = [[] for _ in range(num_steps)]
    all_gts = [[] for _ in range(num_steps)]

    z_channel_indices = model.z_channel_indices

    for sample_idx, start_idx in enumerate(start_indices):
        sample = dataset[start_idx]
        cond = sample["condition"].unsqueeze(0).to(device)

        # cond: (1, T*C, H, W) — 时间维已展平
        # 从中提取元数据用于后续计算
        B, TC, H, W = cond.shape
        C = model.data_cfg.num_channels
        T = model.data_cfg.history_steps
        # 重塑为 (1, T, C, H, W) 用于AR滚动
        window = cond.view(1, T, C, H, W)

        x_t = torch.randn(1, C, H, W, device=device)
        z_prev = None

        for step in range(num_steps):
            # 采样
            if method == "fm":
                pred = model.sample_fm(
                    window.view(1, -1, H, W),
                    device,
                    euler_steps=euler_steps,
                    euler_mode="midpoint",
                    clamp_range=clamp_range,
                    z_clamp_range=z_clamp_range,
                )
            else:
                pred = model.sample_dm(
                    window.view(1, -1, H, W),
                    device,
                    ddim_steps=ddim_steps,
                    clamp_range=clamp_range,
                    z_clamp_range=z_clamp_range,
                )

            pred = pred[:, :C]

            # Z通道Delta Clamp
            if z_channel_indices and z_prev is not None:
                z_new = pred[:, z_channel_indices]
                z_delta = z_new - z_prev
                z_delta_clamped = z_delta.clamp(-0.5, 0.5)
                pred[:, z_channel_indices] = z_prev + z_delta_clamped

            if z_channel_indices:
                z_prev = pred[:, z_channel_indices].clone()

            # 收集预测
            all_preds[step].append(pred.cpu())

            # 收集真值
            gt_sample = dataset[start_idx + step]
            gt = gt_sample["target"][:C].unsqueeze(0)
            all_gts[step].append(gt)

            # 滚动条件窗口
            if T > 1:
                pred_noise = pred
                if noise_sigma > 0:
                    pred_noise = pred + torch.randn_like(pred) * noise_sigma

                window = torch.cat([
                    window[:, 1:],
                    pred_noise.unsqueeze(1),
                ], dim=1)

        if (sample_idx + 1) % 20 == 0:
            logger.info(f"  [{method.upper()}] 推理进度: {sample_idx + 1}/{len(start_indices)}")

    # 合并batch
    all_preds = [
        torch.cat(step_preds, dim=0) for step_preds in all_preds
    ]
    all_gts = [
        torch.cat(step_gts, dim=0) for step_gts in all_gts
    ]

    return all_preds, all_gts


# ============================================================
# 主对比实验
# ============================================================

class ComparisonExperiment:
    """
    FM vs DM 对比实验主类
    """

    def __init__(
        self,
        data_cfg: DataConfig,
        model_cfg: ModelConfig,
        train_cfg: TrainConfig,
        infer_cfg: InferenceConfig,
        comp_cfg: ComparisonConfig,
        work_dir: str,
    ):
        self.data_cfg = data_cfg
        self.model_cfg = model_cfg
        self.train_cfg = train_cfg
        self.infer_cfg = infer_cfg
        self.comp_cfg = comp_cfg
        self.work_dir = work_dir

        self.device = torch.device(comp_cfg.device if torch.cuda.is_available() else "cpu")
        logger.info(f"设备: {self.device}")

        # 加载数据
        self.train_loader, self.val_loader, self.test_loader, self.norm_mean, self.norm_std = build_dataloaders(
            data_cfg, train_cfg
        )
        logger.info(f"训练集: {len(self.train_loader.dataset)}, 验证集: {len(self.val_loader.dataset)}, 测试集: {len(self.test_loader.dataset)}")

        self.test_dataset = self.test_loader.dataset
        self.data_cfg.norm_stats_path = data_cfg.norm_stats_path or os.path.join(work_dir, "norm_stats.pt")

        # 评估器
        self.evaluator = ComparisonEvaluator(
            data_cfg=data_cfg,
            device=self.device,
            norm_mean=self.norm_mean,
            norm_std=self.norm_std,
        )

        # 模型
        self.fm_model = None
        self.dm_model = None

    def train_models(self):
        """训练FM和DM模型"""
        logger.info("=" * 70)
        logger.info("开始训练模型")
        logger.info("=" * 70)

        # 训练Flow Matching模型
        if self.comp_cfg.run_fm:
            logger.info("-" * 50)
            logger.info("训练 Flow Matching 模型")
            logger.info("-" * 50)

            fm_ckpt = os.path.join(self.work_dir, "checkpoints_fm", "best.pt")
            if os.path.exists(fm_ckpt):
                logger.info(f"找到FM checkpoint: {fm_ckpt}, 跳过训练")
                self.fm_model = create_model(self.model_cfg, self.data_cfg, self.train_cfg, method="fm")
                self.fm_model = self.fm_model.to(self.device)
                self.fm_model, _ = load_checkpoint(self.fm_model, fm_ckpt, self.device, use_ema=True)
            else:
                self.fm_model = create_model(self.model_cfg, self.data_cfg, self.train_cfg, method="fm")
                self.fm_model = self.fm_model.to(self.device)

                trainer = UnifiedTrainer(
                    self.fm_model,
                    self.train_loader,
                    self.val_loader,
                    self.train_cfg,
                    self.data_cfg,
                    work_dir=self.work_dir,
                    method="fm",
                )
                trainer.train()

                # 加载best checkpoint
                best_ckpt = os.path.join(self.work_dir, "checkpoints_fm", "best.pt")
                if os.path.exists(best_ckpt):
                    self.fm_model, _ = load_checkpoint(self.fm_model, best_ckpt, self.device, use_ema=True)

            self.fm_model.eval()

        # 训练Diffusion模型
        if self.comp_cfg.run_dm:
            logger.info("-" * 50)
            logger.info("训练 Diffusion 模型")
            logger.info("-" * 50)

            dm_ckpt = os.path.join(self.work_dir, "checkpoints_dm", "best.pt")
            if os.path.exists(dm_ckpt):
                logger.info(f"找到DM checkpoint: {dm_ckpt}, 跳过训练")
                self.dm_model = create_model(self.model_cfg, self.data_cfg, self.train_cfg, method="dm")
                self.dm_model = self.dm_model.to(self.device)
                self.dm_model, _ = load_checkpoint(self.dm_model, dm_ckpt, self.device, use_ema=True)
            else:
                self.dm_model = create_model(self.model_cfg, self.data_cfg, self.train_cfg, method="dm")
                self.dm_model = self.dm_model.to(self.device)

                trainer = UnifiedTrainer(
                    self.dm_model,
                    self.train_loader,
                    self.val_loader,
                    self.train_cfg,
                    self.data_cfg,
                    work_dir=self.work_dir,
                    method="dm",
                )
                trainer.train()

                # 加载best checkpoint
                best_ckpt = os.path.join(self.work_dir, "checkpoints_dm", "best.pt")
                if os.path.exists(best_ckpt):
                    self.dm_model, _ = load_checkpoint(self.dm_model, best_ckpt, self.device, use_ema=True)

            self.dm_model.eval()

    def run_evaluation(self):
        """运行全面对比评估"""
        logger.info("=" * 70)
        logger.info("开始对比评估")
        logger.info("=" * 70)

        # 准备评估样本
        num_samples = self.comp_cfg.num_eval_samples
        ar_steps = self.infer_cfg.autoregressive_steps

        start_indices = find_ar_sequences(self.test_dataset, ar_steps, num_samples)
        if len(start_indices) == 0:
            logger.warning("找不到足够的连续序列，限制评估样本数")
            num_samples = min(num_samples, len(self.test_dataset))
            start_indices = list(range(num_samples))

        logger.info(f"评估样本数: {len(start_indices)}, 自回归步数: {ar_steps}")

        all_results = {}

        # ===========================
        # 1. Flow Matching 评估
        # ===========================
        if self.fm_model is not None and self.comp_cfg.run_fm:
            logger.info("-" * 50)
            logger.info("评估 Flow Matching")
            logger.info("-" * 50)

            fm_preds, fm_gts = autoregressive_inference(
                self.fm_model, self.test_dataset, start_indices, ar_steps,
                self.device, method="fm",
                clamp_range=self.infer_cfg.clamp_range,
                z_clamp_range=self.infer_cfg.z_clamp_range,
                euler_steps=self.infer_cfg.euler_steps,
                ddim_steps=self.infer_cfg.ddim_steps,
                noise_sigma=self.infer_cfg.autoregressive_noise_sigma,
                infer_cfg=self.infer_cfg,
            )

            fm_results = self.evaluator.evaluate_single(fm_preds, fm_gts, method_name="FM")
            all_results["FM"] = fm_results

            if self.comp_cfg.compute_psd:
                fm_spectral = self.evaluator.evaluate_spectral(fm_preds, fm_gts, method_name="FM")
                all_results["FM_spectral"] = fm_spectral

        # ===========================
        # 2. Diffusion 评估
        # ===========================
        if self.dm_model is not None and self.comp_cfg.run_dm:
            logger.info("-" * 50)
            logger.info("评估 Diffusion")
            logger.info("-" * 50)

            dm_preds, dm_gts = autoregressive_inference(
                self.dm_model, self.test_dataset, start_indices, ar_steps,
                self.device, method="dm",
                clamp_range=self.infer_cfg.clamp_range,
                z_clamp_range=self.infer_cfg.z_clamp_range,
                euler_steps=self.infer_cfg.euler_steps,
                ddim_steps=self.infer_cfg.ddim_steps,
                noise_sigma=self.infer_cfg.autoregressive_noise_sigma,
                infer_cfg=self.infer_cfg,
            )

            dm_results = self.evaluator.evaluate_single(dm_preds, dm_gts, method_name="DM")
            all_results["DM"] = dm_results

            if self.comp_cfg.compute_psd:
                dm_spectral = self.evaluator.evaluate_spectral(dm_preds, dm_gts, method_name="DM")
                all_results["DM_spectral"] = dm_spectral

        # ===========================
        # 3. NFE效率对比 (论文核心)
        # ===========================
        if self.comp_cfg.compute_rmse and self.fm_model is not None and self.dm_model is not None:
            logger.info("-" * 50)
            logger.info("评估 NFE 效率曲线")
            logger.info("-" * 50)

            # 使用较少样本和步数做NFE测试
            nfe_start_indices = start_indices[:min(30, len(start_indices))]
            # NFE测试: 单步预测精度 vs NFE步数
            # condition = 全历史 (T*C, H, W) flatten → reshape to (T, C, H, W) for sample_fm
            nfe_targets = [
                self.test_dataset[i + 1]["target"].unsqueeze(0)
                for i in nfe_start_indices
            ]
            nfe_conditions = [
                self.test_dataset[i]["condition"]  # (T*C, H, W)
                for i in nfe_start_indices
            ]

            # FM NFE测试 (1-8步)
            fm_nfe_results = {}
            fm_steps = [s for s in self.infer_cfg.nfe_steps_list if s <= 16]
            if fm_steps:
                fm_nfe_results = compute_nfe_efficiency(
                    self.fm_model, nfe_conditions, nfe_targets,
                    fm_steps, method="fm", device=self.device,
                    clamp_range=self.infer_cfg.clamp_range,
                    z_clamp_range=self.infer_cfg.z_clamp_range,
                    z_channel_indices=self.data_cfg.z_channel_indices,
                )

            # DM NFE测试 (5-50步)
            dm_nfe_results = {}
            dm_steps = [s for s in self.infer_cfg.nfe_steps_list if s >= 5]
            if dm_steps:
                dm_nfe_results = compute_nfe_efficiency(
                    self.dm_model, nfe_conditions, nfe_targets,
                    dm_steps, method="dm", device=self.device,
                    clamp_range=self.infer_cfg.clamp_range,
                    z_clamp_range=self.infer_cfg.z_clamp_range,
                    z_channel_indices=self.data_cfg.z_channel_indices,
                )

            all_results["NFE_efficiency"] = {
                "FM": fm_nfe_results,
                "DM": dm_nfe_results,
            }

            self._print_nfe_table(fm_nfe_results, dm_nfe_results)

        # 保存结果
        results_path = os.path.join(self.comp_cfg.output_dir, "comparison_results.json")
        self.evaluator.save_results(all_results, results_path)

        # 打印最终对比总结
        self._print_final_comparison(all_results)

        return all_results

    def _print_nfe_table(self, fm_results: Dict, dm_results: Dict):
        """打印NFE效率表格"""
        logger.info("")
        logger.info("=" * 70)
        logger.info("NFE 效率对比 (推理步数 vs RMSE)")
        logger.info("=" * 70)

        header = f"{'NFE':>8}"
        if fm_results:
            header += f"  {'FM RMSE':>12}"
        if dm_results:
            header += f"  {'DM RMSE':>12}"
        logger.info(header)
        logger.info("-" * 40)

        all_steps = sorted(set(list(fm_results.keys()) + list(dm_results.keys())))
        for step in all_steps:
            row = f"{step:>8}"
            if step in fm_results:
                row += f"  {fm_results[step]['rmse_mean']:>12.4f}"
            else:
                row += f"  {'--':>12}"
            if step in dm_results:
                row += f"  {dm_results[step]['rmse_mean']:>12.4f}"
            else:
                row += f"  {'--':>12}"
            logger.info(row)

        logger.info("")

    def _print_final_comparison(self, results: Dict):
        """打印最终对比总结"""
        logger.info("")
        logger.info("=" * 70)
        logger.info("最终对比总结")
        logger.info("=" * 70)

        fm_rmse = results.get("FM", {}).get("rmse_mean", 0)
        dm_rmse = results.get("DM", {}).get("rmse_mean", 0)
        fm_lat_rmse = results.get("FM", {}).get("lat_weighted_rmse_mean", 0)
        dm_lat_rmse = results.get("DM", {}).get("lat_weighted_rmse_mean", 0)

        logger.info(f"平均RMSE:     FM={fm_rmse:.4f}  DM={dm_rmse:.4f}  差异={fm_rmse-dm_rmse:+.4f}")
        logger.info(f"纬度加权RMSE: FM={fm_lat_rmse:.4f}  DM={dm_lat_rmse:.4f}  差异={fm_lat_rmse-dm_lat_rmse:+.4f}")

        if "FM_spectral" in results and "DM_spectral" in results:
            fm_slope = results["FM_spectral"].get("spectral_slope_pred", 0)
            dm_slope = results["DM_spectral"].get("spectral_slope_pred", 0)
            gt_slope = results["FM_spectral"].get("spectral_slope_gt", 0)
            logger.info(f"谱斜率:       FM={fm_slope:.3f}  DM={dm_slope:.3f}  真值={gt_slope:.3f}")

        if "NFE_efficiency" in results:
            nfe = results["NFE_efficiency"]
            fm_4step = nfe.get("FM", {}).get(4, {}).get("rmse_mean", None)
            dm_50step = nfe.get("DM", {}).get(50, {}).get("rmse_mean", None)
            if fm_4step and dm_50step:
                logger.info(f"效率对比:      FM 4步={fm_4step:.4f}  DM 50步={dm_50step:.4f}")
                logger.info(f"加速比:        FM比DM推理速度快 {50/4:.1f}x")

        logger.info("=" * 70)


# ============================================================
# 入口
# ============================================================

def main():
    torch.set_float32_matmul_precision("high")

    parser = argparse.ArgumentParser(description="FM vs DM 对比实验")

    # 数据
    parser.add_argument("--data_root", type=str, default="/root/autodl-tmp/fyp_final/preprocessed_9ch_40x40")
    parser.add_argument("--preprocess_dir", type=str, default=None)
    parser.add_argument("--norm_stats", type=str, default=None)

    # 实验模式
    parser.add_argument("--run_fm", action="store_true", default=True)
    parser.add_argument("--run_dm", action="store_true", default=True)
    parser.add_argument("--skip_train", action="store_true", default=False)

    # 训练参数
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--seed", type=int, default=42)

    # 推理参数
    parser.add_argument("--ar_steps", type=int, default=24)
    parser.add_argument("--euler_steps", type=int, default=4)
    parser.add_argument("--ddim_steps", type=int, default=50)
    parser.add_argument("--eval_samples", type=int, default=100)

    # 输出
    parser.add_argument("--work_dir", type=str, default=".")
    parser.add_argument("--output_dir", type=str, default="comparison_results")

    # 设备
    parser.add_argument("--device", type=str, default="cuda")

    # 新增: 外部 DM checkpoint (如 newtry 的 best_eps.pt)
    parser.add_argument("--external_dm_ckpt", type=str, default=None,
                        help="加载外部 DM checkpoint 路径 (例如 newtry/checkpoints/best_eps.pt)")

    args = parser.parse_args()

    # 设置随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # 配置
    data_cfg, model_cfg, train_cfg, infer_cfg = get_config()
    if args.norm_stats:
        data_cfg.norm_stats_path = args.norm_stats
    if args.preprocess_dir:
        data_cfg.preprocessed_dir = args.preprocess_dir

    train_cfg.max_epochs = args.epochs
    train_cfg.batch_size = args.batch_size
    train_cfg.learning_rate = args.lr
    train_cfg.seed = args.seed

    infer_cfg.autoregressive_steps = args.ar_steps
    infer_cfg.euler_steps = args.euler_steps
    infer_cfg.ddim_steps = args.ddim_steps

    comp_cfg = get_comparison_config()
    comp_cfg.run_fm = args.run_fm
    comp_cfg.run_dm = args.run_dm
    comp_cfg.num_eval_samples = args.eval_samples
    comp_cfg.output_dir = args.output_dir
    comp_cfg.device = args.device

    data_cfg.data_root = args.data_root
    data_cfg.era5_dir = args.data_root

    # 输出目录
    os.makedirs(args.work_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

    logger.info(f"数据目录: {args.data_root}")
    logger.info(f"输出目录: {args.work_dir}")
    logger.info(f"评估目录: {args.output_dir}")
    logger.info(f"训练epochs: {args.epochs}, batch_size: {args.batch_size}")

    # 创建实验
    experiment = ComparisonExperiment(
        data_cfg, model_cfg, train_cfg, infer_cfg, comp_cfg, args.work_dir
    )

    # 训练
    if not args.skip_train:
        experiment.train_models()
    else:
        # 仅加载模型
        logger.info("跳过训练，仅加载模型")
        fm_ckpt = os.path.join(args.work_dir, "checkpoints_fm", "best.pt")
        dm_ckpt = os.path.join(args.work_dir, "checkpoints_dm", "best.pt")

        # FM 模型 (使用 UnifiedModel)
        if args.run_fm and os.path.exists(fm_ckpt):
            logger.info(f"加载 FM 模型: {fm_ckpt}")
            experiment.fm_model = create_model(model_cfg, data_cfg, train_cfg, method="fm")
            experiment.fm_model = experiment.fm_model.to(experiment.device)
            experiment.fm_model, _ = load_checkpoint(experiment.fm_model, fm_ckpt, experiment.device)
            experiment.fm_model.eval()
        elif args.run_fm:
            logger.warning("FM 模型未找到，跳过")

        # DM 模型: 使用外部 checkpoint (如 newtry) 或默认训练 checkpoint
        if args.run_dm:
            if args.external_dm_ckpt and os.path.exists(args.external_dm_ckpt):
                # 加载外部 checkpoint (如 newtry/best_eps.pt)
                logger.info(f"加载外部 DM 模型: {args.external_dm_ckpt}")
                from models.adapter import load_newtry_checkpoint
                experiment.dm_model = load_newtry_checkpoint(
                    args.external_dm_ckpt,
                    data_cfg,
                    model_cfg,
                    train_cfg,
                    experiment.device,
                )
                experiment.dm_model.eval()
            elif os.path.exists(dm_ckpt):
                logger.info(f"加载 DM 模型: {dm_ckpt}")
                experiment.dm_model = create_model(model_cfg, data_cfg, train_cfg, method="dm")
                experiment.dm_model = experiment.dm_model.to(experiment.device)
                experiment.dm_model, _ = load_checkpoint(experiment.dm_model, dm_ckpt, experiment.device)
                experiment.dm_model.eval()
            else:
                logger.warning("DM 模型未找到，跳过")

    # 评估
    results = experiment.run_evaluation()

    logger.info("")
    logger.info("=" * 70)
    logger.info("对比实验完成!")
    logger.info(f"结果保存在: {args.output_dir}")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
