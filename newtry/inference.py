"""
ERA5-Diffusion 推理脚本

功能:
  1. 单步预测: DDIM 50 步采样 -> 未来 9h (3步) 气象场
  2. 自回归多步预测: 24 轮滚动 -> 72h 预测
  3. 集合预报: 多次采样 -> 集合均值 + 不确定性估计
  4. 评估: RMSE / MAE / ACC / SSIM 逐变量逐时效

使用方式:
  python inference.py --checkpoint checkpoints/best.pt --data_root /path/to/data
"""
import os
import argparse
import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from configs import DataConfig, ModelConfig, InferenceConfig, get_config
from data.dataset import ERA5TyphoonDataset, split_typhoon_ids
from models import ERA5DiffusionModel
from train import EMA

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


class ERA5Predictor:
    """ERA5 气象场预测器"""

    def __init__(
        self,
        model: ERA5DiffusionModel,
        data_cfg: DataConfig,
        infer_cfg: InferenceConfig,
        norm_mean: np.ndarray,
        norm_std: np.ndarray,
        device: torch.device,
    ):
        self.model = model
        self.data_cfg = data_cfg
        self.infer_cfg = infer_cfg
        self.norm_mean = norm_mean  # (C,)
        self.norm_std = norm_std    # (C,)
        self.device = device

        self.model.eval()
        self.model.scheduler.ddim_steps = infer_cfg.ddim_steps
        self.model.scheduler.clamp_range = infer_cfg.clamp_range
        self.model.scheduler.to(device)

        # z 通道逐通道 clamp 参数
        self.z_clamp_range = infer_cfg.z_clamp_range
        self.z_channel_indices = self.model.z_channel_indices

    def denormalize(self, data: torch.Tensor, is_condition: bool = False) -> torch.Tensor:
        """
        反归一化: x = x_norm * std + mean

        data: (B, C*T, H, W)
        """
        C = self.data_cfg.num_channels
        T = self.data_cfg.history_steps if is_condition else self.data_cfg.forecast_steps

        mean = torch.from_numpy(self.norm_mean).float().to(data.device)
        std = torch.from_numpy(self.norm_std).float().to(data.device)
        std = torch.where(std < 1e-8, torch.ones_like(std), std)

        # 展开为 (C*T,) 并 broadcast
        mean_expanded = mean.repeat(T)  # (C*T,)
        std_expanded = std.repeat(T)
        mean_expanded = mean_expanded.reshape(1, -1, 1, 1)
        std_expanded = std_expanded.reshape(1, -1, 1, 1)

        return data * std_expanded + mean_expanded

    @torch.no_grad()
    def predict_single(self, condition: torch.Tensor) -> torch.Tensor:
        """
        单步预测

        condition: (B, 510, 40, 40) 归一化后的条件
        返回: (B, 102, 40, 40) 归一化空间的预测
        """
        return self.model.sample(condition, self.device, z_clamp_range=self.z_clamp_range)

    @torch.no_grad()
    def predict_autoregressive(
        self,
        initial_condition: torch.Tensor,
        num_steps: int = 24,
        noise_sigma: float = 0.02,
        ensemble_per_step: int = 1,
    ) -> List[torch.Tensor]:
        """
        自回归多步预测（支持逐步集合平均 + z 通道 Delta Clamp 防护）

        initial_condition: (B, C*T_hist, 40, 40) 初始条件（归一化后）
        num_steps: 自回归轮数 (默认 24 轮 -> 72h)
        noise_sigma: 自回归噪声注入的标准差
        ensemble_per_step: 每步 DDIM 采样次数，>1 时取平均以降低方差

        返回: [pred_step_1, pred_step_2, ..., pred_step_N]
               每个 shape = (B, C, 40, 40)

        滚动规则:
          每轮预测 forecast_steps 步，仅取第 1 步拼入条件窗口
          丢弃最早 1 步，保持窗口始终为 history_steps 步
        """
        B = initial_condition.shape[0]
        C = self.data_cfg.num_channels  # 9
        H = W = self.data_cfg.grid_size  # 40

        # 将条件从 (B, C*T_hist, H, W) 拆为 (B, T_hist, C, H, W)
        window = initial_condition.reshape(B, self.data_cfg.history_steps, C, H, W)

        predictions = []

        # ---- z 场累积漂移检测 + 持续性回退 (双轨策略) ----
        # 检测方式: 不看步间 delta，而看 z 相对初始条件的累积漂移量
        # z 爆炸本质是渐进式漂移 (每步 ~0.1-0.3 归一化空间)，步间 delta 正常，但累积后 RMSE 爆炸
        # 所以用初始 z 的均值和标准差建立锚点，超出 drift_sigma 倍标准差即判定异常
        z_drift_sigma = 3.0      # 超出初始 z 分布 3σ 视为异常漂移
        z_fallback_count = 0

        if self.z_channel_indices:
            z_window = window[:, :, self.z_channel_indices]  # (B, T_hist, n_z, H, W)
            # 初始 z 的锚点: per-sample, per-channel 的均值和标准差
            z_anchor_mean = z_window.mean(dim=(1, 3, 4))  # (B, n_z)
            z_anchor_std = z_window.std(dim=(1, 3, 4)).clamp(min=0.1)  # (B, n_z), 最小 0.1 防止除零
            z_prev_output = z_window[:, -1].clone()  # (B, n_z, H, W)
        else:
            z_anchor_mean = None
            z_prev_output = None

        desc = f"自回归预测 (ensemble={ensemble_per_step})" if ensemble_per_step > 1 else "自回归预测"
        ar_pbar = tqdm(range(num_steps), desc=desc, unit="步")
        for step_idx in ar_pbar:
            cond = window.reshape(B, -1, H, W)

            if ensemble_per_step > 1:
                K = ensemble_per_step
                cond_repeated = cond.repeat(K, 1, 1, 1)
                pred_all = self.predict_single(cond_repeated)
                pred_all = pred_all.reshape(K, B, self.data_cfg.forecast_steps, C, H, W)
                pred_first = pred_all[:, :, 0].mean(dim=0)
            else:
                pred_full = self.predict_single(cond)
                pred_split = pred_full.reshape(B, self.data_cfg.forecast_steps, C, H, W)
                pred_first = pred_split[:, 0]

            # ---- 双轨策略: 累积漂移检测 ----
            pred_output = pred_first

            if self.z_channel_indices and z_anchor_mean is not None:
                z_new = pred_first[:, self.z_channel_indices]  # (B, n_z, H, W)
                # 计算当前 z 均值相对锚点的漂移 (per-sample, per-channel)
                z_new_mean = z_new.mean(dim=(2, 3))  # (B, n_z)
                z_drift = ((z_new_mean - z_anchor_mean) / z_anchor_std).abs()  # (B, n_z)
                # 任一 z 通道漂移超过阈值即判定该样本异常
                is_anomaly = z_drift.max(dim=1).values > z_drift_sigma  # (B,) bool

                if is_anomaly.any():
                    z_fallback_count += is_anomaly.sum().item()
                    pred_output = pred_first.clone()
                    mask = is_anomaly.reshape(B, 1, 1, 1).expand_as(z_new)
                    pred_output[:, self.z_channel_indices] = torch.where(mask, z_prev_output, z_new)

            # 更新输出轨道 z_prev
            if self.z_channel_indices:
                z_prev_output = pred_output[:, self.z_channel_indices].clone()

            predictions.append(pred_output)

            # ---- 反馈轨道: 始终用模型原始预测 ----
            feedback = pred_first
            if step_idx < num_steps - 1 and noise_sigma > 0:
                feedback = feedback + noise_sigma * torch.randn_like(feedback)
            if self.z_channel_indices and self.z_clamp_range:
                feedback = feedback.clone()
                feedback[:, self.z_channel_indices] = feedback[:, self.z_channel_indices].clamp(
                    *self.z_clamp_range
                )

            window = torch.cat([
                window[:, 1:],
                feedback.unsqueeze(1),
            ], dim=1)

            ar_pbar.set_postfix(lead=f"+{(step_idx+1)*3}h")

        if z_fallback_count > 0:
            logger.info(f"z 场漂移回退触发: {z_fallback_count} 次 (共 {B}x{num_steps}={B*num_steps} 步)")

        return predictions

    @torch.no_grad()
    def predict_ensemble(
        self,
        condition: torch.Tensor,
        ensemble_size: int = 10,
        autoregressive: bool = False,
        num_ar_steps: int = 24,
    ) -> Dict[str, torch.Tensor]:
        """
        集合预报

        condition: (B, 510, 40, 40)
        ensemble_size: 集合成员数

        返回: {
            'mean': 集合均值,
            'std': 集合标准差 (不确定性估计),
            'members': 所有成员的预测,
        }
        """
        if autoregressive:
            # 自回归集合
            all_members = []
            for m in tqdm(range(ensemble_size), desc="集合预报", unit="成员"):
                preds = self.predict_autoregressive(
                    condition, num_ar_steps,
                    noise_sigma=self.infer_cfg.autoregressive_noise_sigma,
                    ensemble_per_step=self.infer_cfg.ar_ensemble_per_step,
                )
                # preds: list of (B, 34, H, W), len = num_ar_steps
                member = torch.stack(preds, dim=1)  # (B, T, 34, H, W)
                all_members.append(member)

            members = torch.stack(all_members, dim=0)  # (E, B, T, 34, H, W)
            return {
                "mean": members.mean(dim=0),
                "std": members.std(dim=0),
                "members": members,
            }
        else:
            # 单步集合
            all_preds = []
            for m in tqdm(range(ensemble_size), desc="集合采样", unit="成员"):
                pred = self.predict_single(condition)
                all_preds.append(pred)

            members = torch.stack(all_preds, dim=0)  # (E, B, 102, H, W)
            return {
                "mean": members.mean(dim=0),
                "std": members.std(dim=0),
                "members": members,
            }


# ============================================================
# 评估指标
# ============================================================

def compute_rmse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """逐通道 RMSE: (C,)"""
    mse = ((pred - target) ** 2).mean(dim=(0, 2, 3))  # (C,)
    return torch.sqrt(mse)


def compute_mae(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """逐通道 MAE: (C,)"""
    return (pred - target).abs().mean(dim=(0, 2, 3))  # (C,)


def compute_acc(pred: torch.Tensor, target: torch.Tensor, clim: torch.Tensor) -> torch.Tensor:
    """
    异常相关系数 ACC（逐通道）

    ACC = sum((pred-clim)*(target-clim)) / sqrt(sum((pred-clim)^2) * sum((target-clim)^2))
    """
    pred_anom = pred - clim
    target_anom = target - clim

    numerator = (pred_anom * target_anom).sum(dim=(0, 2, 3))
    denominator = torch.sqrt(
        (pred_anom ** 2).sum(dim=(0, 2, 3)) * (target_anom ** 2).sum(dim=(0, 2, 3))
    )
    return numerator / (denominator + 1e-8)


def compute_ssim_channel(pred: torch.Tensor, target: torch.Tensor) -> float:
    """
    简化版 SSIM（全图计算），适用于气象场

    pred, target: (H, W) 单通道
    """
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    mu_p = pred.mean()
    mu_t = target.mean()
    sigma_p = pred.var()
    sigma_t = target.var()
    sigma_pt = ((pred - mu_p) * (target - mu_t)).mean()

    ssim = ((2 * mu_p * mu_t + C1) * (2 * sigma_pt + C2)) / (
        (mu_p ** 2 + mu_t ** 2 + C1) * (sigma_p + sigma_t + C2)
    )
    return ssim.item()


def evaluate_predictions(
    predictions: List[torch.Tensor],
    ground_truth: List[torch.Tensor],
    norm_mean: np.ndarray,
    norm_std: np.ndarray,
    var_names: List[str],
) -> Dict[str, np.ndarray]:
    """
    全面评估预测结果

    predictions: list of (B, 34, H, W) 归一化空间的预测
    ground_truth: list of (B, 34, H, W) 归一化空间的真值
    var_names: 变量名列表 (34个)

    返回: {
        'rmse': (T, C) 逐时效逐变量 RMSE,
        'mae': (T, C) 逐时效逐变量 MAE,
    }
    """
    mean_t = torch.from_numpy(norm_mean).float()
    std_t = torch.from_numpy(norm_std).float()
    std_t = torch.where(std_t < 1e-8, torch.ones_like(std_t), std_t)

    T = len(predictions)
    C = predictions[0].shape[1]
    rmse_all = np.zeros((T, C))
    mae_all = np.zeros((T, C))

    for t_idx in range(T):
        pred = predictions[t_idx]
        gt = ground_truth[t_idx]

        # 反归一化
        pred_phys = pred * std_t.reshape(1, -1, 1, 1).to(pred.device) + mean_t.reshape(1, -1, 1, 1).to(pred.device)
        gt_phys = gt * std_t.reshape(1, -1, 1, 1).to(gt.device) + mean_t.reshape(1, -1, 1, 1).to(gt.device)

        rmse = compute_rmse(pred_phys, gt_phys).cpu().numpy()
        mae = compute_mae(pred_phys, gt_phys).cpu().numpy()
        rmse_all[t_idx] = rmse
        mae_all[t_idx] = mae

    return {"rmse": rmse_all, "mae": mae_all}


# ============================================================
# 入口
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="ERA5-Diffusion Inference")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data_root", type=str, default="/Volumes/T7 Shield/Typhoon_data_final")
    parser.add_argument("--work_dir", type=str, default=".")
    parser.add_argument("--mode", choices=["single", "autoregressive", "ensemble"], default="autoregressive")
    parser.add_argument("--num_ar_steps", type=int, default=24)
    parser.add_argument("--ensemble_size", type=int, default=10)
    parser.add_argument("--num_samples", type=int, default=5, help="测试样本数量")
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--preprocess_dir", type=str, default=None,
                        help="预处理NPY目录, 如 /root/autodl-tmp/preprocessed_npy")
    parser.add_argument("--ar_ensemble", type=int, default=None,
                        help="自回归逐步集合数 (覆盖 config 中的 ar_ensemble_per_step，默认=5)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 配置
    data_cfg, model_cfg, _, infer_cfg = get_config(data_root=args.data_root)
    infer_cfg.checkpoint_path = args.checkpoint
    infer_cfg.output_dir = args.output_dir
    if args.ar_ensemble is not None:
        infer_cfg.ar_ensemble_per_step = args.ar_ensemble
    os.makedirs(args.output_dir, exist_ok=True)

    # 加载归一化统计
    norm_stats_path = os.path.join(args.work_dir, "norm_stats.pt")
    assert os.path.exists(norm_stats_path), f"找不到归一化统计文件: {norm_stats_path}"
    stats = torch.load(norm_stats_path, weights_only=True)
    norm_mean = stats["mean"].numpy()
    norm_std = stats["std"].numpy()

    # 加载模型
    logger.info(f"加载模型: {args.checkpoint}")
    model = ERA5DiffusionModel(model_cfg, data_cfg).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)

    # 使用 EMA 参数
    if "ema_state_dict" in ckpt:
        ema = EMA(model, decay=0.9999)
        ema.load_state_dict(ckpt["ema_state_dict"])
        ema.apply_shadow(model)
        logger.info("已加载 EMA 参数")
    else:
        model.load_state_dict(ckpt["model_state_dict"])

    # 构建测试数据
    _, _, test_ids = split_typhoon_ids(data_cfg.data_root, seed=42)
    test_dataset = ERA5TyphoonDataset(
        typhoon_ids=test_ids[:10],  # 限制台风数量
        data_root=data_cfg.data_root,
        pl_vars=data_cfg.pressure_level_vars,
        sfc_vars=data_cfg.surface_vars,
        pressure_levels=data_cfg.pressure_levels,
        history_steps=data_cfg.history_steps,
        forecast_steps=data_cfg.forecast_steps,
        norm_mean=norm_mean,
        norm_std=norm_std,
        preprocessed_dir=args.preprocess_dir,
    )

    predictor = ERA5Predictor(model, data_cfg, infer_cfg, norm_mean, norm_std, device)

    if args.mode == "single":
        logger.info("单步预测模式")
        n_samples = min(args.num_samples, len(test_dataset))
        for i in tqdm(range(n_samples), desc="单步预测", unit="样本"):
            sample = test_dataset[i]
            cond = sample["condition"].unsqueeze(0).to(device)
            target = sample["target"].unsqueeze(0).to(device)

            pred = predictor.predict_single(cond)
            pred_phys = predictor.denormalize(pred)
            target_phys = predictor.denormalize(target)

            rmse = compute_rmse(pred_phys, target_phys)
            logger.info(f"样本 {i}: RMSE 均值 = {rmse.mean().item():.4f}")

            # 保存
            torch.save({
                "prediction": pred_phys.cpu(),
                "ground_truth": target_phys.cpu(),
                "typhoon_id": sample["typhoon_id"],
            }, os.path.join(args.output_dir, f"single_pred_{i}.pt"))

    elif args.mode == "autoregressive":
        logger.info(f"自回归预测模式: {args.num_ar_steps} 步 -> {args.num_ar_steps * 3}h, 逐步集合={infer_cfg.ar_ensemble_per_step}")
        n_samples = min(args.num_samples, len(test_dataset))
        ar_steps = args.num_ar_steps
        C = data_cfg.num_channels

        # 收集所有样本的预测与真值用于汇总 RMSE
        all_preds = [[] for _ in range(ar_steps)]  # [step][sample] = (1, C, H, W)
        all_gts = [[] for _ in range(ar_steps)]
        valid_counts = [0] * ar_steps  # 每个时效有效真值数量

        for i in tqdm(range(n_samples), desc="自回归推理", unit="样本"):
            sample = test_dataset[i]
            cond = sample["condition"].unsqueeze(0).to(device)
            tid = sample["typhoon_id"]

            preds = predictor.predict_autoregressive(
                cond, num_steps=ar_steps,
                noise_sigma=infer_cfg.autoregressive_noise_sigma,
                ensemble_per_step=infer_cfg.ar_ensemble_per_step,
            )

            logger.info(f"样本 {i} ({tid}): 完成 {len(preds)} 步预测")

            # 保存
            preds_stacked = torch.stack(preds, dim=1).cpu()  # (1, T, C, H, W)
            torch.save({
                "predictions": preds_stacked,
                "typhoon_id": tid,
            }, os.path.join(args.output_dir, f"ar_pred_{i}.pt"))

            # 收集预测
            for t in range(ar_steps):
                all_preds[t].append(preds[t].cpu())

            # 收集真值: 利用滑动窗口连续性
            for t in range(ar_steps):
                gt_idx = i + t
                if gt_idx < len(test_dataset):
                    gt_sample = test_dataset[gt_idx]
                    if gt_sample["typhoon_id"] == tid:
                        gt_step = gt_sample["target"][:C]  # (C, H, W)
                        all_gts[t].append(gt_step.unsqueeze(0))
                        valid_counts[t] += 1
                    else:
                        all_gts[t].append(None)
                else:
                    all_gts[t].append(None)

        # ---- 汇总 RMSE 表格 (鲁棒统计) ----
        var_names = []
        for var in data_cfg.pressure_level_vars:
            for lev in data_cfg.pressure_levels:
                var_names.append(f"{var}_{lev}")
        for var in data_cfg.surface_vars:
            var_names.append(var)

        mean_t = torch.from_numpy(norm_mean).float()
        std_t = torch.from_numpy(norm_std).float()
        std_t = torch.where(std_t < 1e-8, torch.ones_like(std_t), std_t)

        # 逐样本逐时效计算 RMSE: per_sample_rmse[t][s] = (C,) or None
        n_vars = len(var_names)
        per_sample_rmse = [[None] * n_samples for _ in range(ar_steps)]

        for t in range(ar_steps):
            for s in range(n_samples):
                if all_gts[t][s] is not None:
                    p = all_preds[t][s]  # (1, C, H, W)
                    g = all_gts[t][s]    # (1, C, H, W)
                    # 反归一化
                    p_phys = p * std_t.reshape(1, -1, 1, 1) + mean_t.reshape(1, -1, 1, 1)
                    g_phys = g * std_t.reshape(1, -1, 1, 1) + mean_t.reshape(1, -1, 1, 1)
                    # 逐通道 RMSE: (C,)
                    rmse_per_ch = torch.sqrt(((p_phys - g_phys) ** 2).mean(dim=(2, 3))).squeeze(0).numpy()
                    per_sample_rmse[t][s] = rmse_per_ch

        # 汇总统计: 均值 / 中位数 / P90
        rmse_mean = np.zeros((ar_steps, n_vars))
        rmse_median = np.zeros((ar_steps, n_vars))
        rmse_p90 = np.zeros((ar_steps, n_vars))

        for t in range(ar_steps):
            valid_rmses = [r for r in per_sample_rmse[t] if r is not None]
            if valid_rmses:
                stacked = np.stack(valid_rmses, axis=0)  # (n_valid, C)
                rmse_mean[t] = stacked.mean(axis=0)
                rmse_median[t] = np.median(stacked, axis=0)
                rmse_p90[t] = np.percentile(stacked, 90, axis=0)
                valid_counts[t] = len(valid_rmses)

        # ---- 打印中位数 RMSE 表格 (主表, 抗异常值) ----
        logger.info("")
        logger.info("中位数 RMSE (抗异常值, 推荐参考):")
        header = f"{'时效':>10}"
        for v in var_names:
            header += f"  {v:>8}"
        logger.info(header)
        logger.info("=" * (10 + 10 * n_vars))

        for t in range(ar_steps):
            lead_h = (t + 1) * data_cfg.time_interval_hours
            row = f"+ {lead_h:>3}h    "
            for c in range(n_vars):
                row += f"  {rmse_median[t, c]:>8.2f}"
            row += f"  (n={valid_counts[t]})"
            logger.info(row)

        med_avg = f"{'平均':>10}"
        for c in range(n_vars):
            med_avg += f"  {rmse_median[:, c].mean():>8.2f}"
        logger.info("")
        logger.info(med_avg)

        # ---- 打印均值 RMSE 表格 ----
        logger.info("")
        logger.info("均值 RMSE (含异常值):")
        header2 = f"{'时效':>10}"
        for v in var_names:
            header2 += f"  {v:>8}"
        logger.info(header2)
        logger.info("-" * (10 + 10 * n_vars))

        for t in range(ar_steps):
            lead_h = (t + 1) * data_cfg.time_interval_hours
            row = f"+ {lead_h:>3}h    "
            for c in range(n_vars):
                row += f"  {rmse_mean[t, c]:>8.2f}"
            row += f"  (n={valid_counts[t]})"
            logger.info(row)

        mean_avg = f"{'平均':>10}"
        for c in range(n_vars):
            mean_avg += f"  {rmse_mean[:, c].mean():>8.2f}"
        logger.info("")
        logger.info(mean_avg)

        # ---- 异常样本报告 ----
        logger.info("")
        logger.info("异常样本检测 (z_850 RMSE > 中位数 × 3):")
        outlier_count = 0
        for t in range(ar_steps):
            lead_h = (t + 1) * data_cfg.time_interval_hours
            z850_median = rmse_median[t, 6] if n_vars > 6 else 0
            threshold = max(z850_median * 3, 500)  # 至少 500
            for s in range(n_samples):
                if per_sample_rmse[t][s] is not None and n_vars > 6:
                    z850_rmse = per_sample_rmse[t][s][6]
                    if z850_rmse > threshold:
                        logger.info(f"  +{lead_h:3d}h 样本{s}: z_850 RMSE={z850_rmse:.0f} (中位数={z850_median:.0f}, 阈值={threshold:.0f})")
                        outlier_count += 1
        if outlier_count == 0:
            logger.info("  无异常样本")
        else:
            logger.info(f"  共 {outlier_count} 个异常 (时效, 样本) 对")

        logger.info(f"\n共 {n_samples} 个样本, 每个时效有效真值数量见 n= 列")

    elif args.mode == "ensemble":
        logger.info(f"集合预报模式: {args.ensemble_size} 成员")
        n_samples = min(args.num_samples, len(test_dataset))
        for i in tqdm(range(n_samples), desc="集合预报推理", unit="样本"):
            sample = test_dataset[i]
            cond = sample["condition"].unsqueeze(0).to(device)

            result = predictor.predict_ensemble(
                cond,
                ensemble_size=args.ensemble_size,
                autoregressive=True,
                num_ar_steps=args.num_ar_steps,
            )

            logger.info(
                f"样本 {i}: 集合均值 shape={result['mean'].shape}, "
                f"spread 均值={result['std'].mean().item():.4f}"
            )

            torch.save({
                "ensemble_mean": result["mean"].cpu(),
                "ensemble_std": result["std"].cpu(),
                "typhoon_id": sample["typhoon_id"],
            }, os.path.join(args.output_dir, f"ensemble_pred_{i}.pt"))

    logger.info("推理完成!")


if __name__ == "__main__":
    main()
