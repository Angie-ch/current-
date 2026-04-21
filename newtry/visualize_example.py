"""
ERA5-Diffusion 可视化脚本

功能:
  1. 预测对比图: 真值 vs 预测 vs 差值（逐变量、逐时效）
  2. RMSE 增长曲线: RMSE 随预测时效的变化趋势
  3. 空间误差分布: 预测误差的空间热图
  4. 集合预报可视化: 集合均值 + spread 分布

使用方式:
  python visualize_example.py --pred_dir outputs --output_dir figures
"""
import os
import argparse
import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")  # 非交互后端
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import TwoSlopeNorm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# 字体设置: 使用默认英文字体，避免中文字体缺失警告
plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["axes.unicode_minus"] = False


# ============================================================
# 变量信息
# ============================================================

# 12 个通道的名称（与 DataConfig 中定义的通道顺序一致, 无 vo）
CHANNEL_NAMES = []
PL_VARS = ["u", "v", "z"]
SFC_VARS = ["u10m", "v10m", "msl"]
PRESSURE_LEVELS = [850, 500, 250]

for var in PL_VARS:
    for lev in PRESSURE_LEVELS:
        CHANNEL_NAMES.append(f"{var}_{lev}")
for var in SFC_VARS:
    CHANNEL_NAMES.append(var)

# 展示用的标签和物理单位
DISPLAY_NAMES = {
    "u": "U Wind", "v": "V Wind", "z": "Geopotential Height",
    "u10m": "10m U Wind", "v10m": "10m V Wind", "msl": "Mean Sea Level Pressure",
}

UNITS = {
    "u": "m/s", "v": "m/s", "z": "m²/s²",
    "u10m": "m/s", "v10m": "m/s", "msl": "Pa",
}

# 重点展示的变量索引（用于默认可视化）
KEY_VARS = {
    "u_850": CHANNEL_NAMES.index("u_850"),
    "v_850": CHANNEL_NAMES.index("v_850"),
    "z_500": CHANNEL_NAMES.index("z_500"),
    "u10m": CHANNEL_NAMES.index("u10m"),
    "v10m": CHANNEL_NAMES.index("v10m"),
    "msl": CHANNEL_NAMES.index("msl"),
}


def get_var_base_name(channel_name: str) -> str:
    """从通道名提取变量基础名 (如 'z_1000' -> 'z')"""
    parts = channel_name.split("_")
    if len(parts) == 2 and parts[1].isdigit():
        return parts[0]
    return channel_name


# ============================================================
# 图 1: 单步预测对比图
# ============================================================

def plot_prediction_comparison(
    pred: np.ndarray,
    truth: np.ndarray,
    var_indices: Optional[Dict[str, int]] = None,
    title_prefix: str = "",
    save_path: Optional[str] = None,
):
    """
    预测 vs 真值 vs 差值 对比图

    pred:  (C, H, W) 物理量空间的预测
    truth: (C, H, W) 物理量空间的真值
    var_indices: {变量名: 通道索引} 字典，默认使用 KEY_VARS
    """
    if var_indices is None:
        var_indices = KEY_VARS

    n_vars = len(var_indices)
    fig, axes = plt.subplots(n_vars, 3, figsize=(15, 4 * n_vars))

    if n_vars == 1:
        axes = axes.reshape(1, -1)

    for row, (var_name, ch_idx) in enumerate(var_indices.items()):
        p = pred[ch_idx]
        t = truth[ch_idx]
        diff = p - t

        base_name = get_var_base_name(var_name)
        display = DISPLAY_NAMES.get(base_name, var_name)
        unit = UNITS.get(base_name, "")

        vmin = min(p.min(), t.min())
        vmax = max(p.max(), t.max())

        # Truth
        im0 = axes[row, 0].imshow(t, origin="lower", cmap="RdYlBu_r", vmin=vmin, vmax=vmax)
        axes[row, 0].set_title(f"Truth: {display} ({var_name})")
        plt.colorbar(im0, ax=axes[row, 0], shrink=0.8, label=unit)

        # Prediction
        im1 = axes[row, 1].imshow(p, origin="lower", cmap="RdYlBu_r", vmin=vmin, vmax=vmax)
        axes[row, 1].set_title(f"Prediction: {display} ({var_name})")
        plt.colorbar(im1, ax=axes[row, 1], shrink=0.8, label=unit)

        # Difference
        abs_max = max(abs(diff.min()), abs(diff.max()))
        if abs_max < 1e-12:
            abs_max = 1.0
        norm = TwoSlopeNorm(vmin=-abs_max, vcenter=0, vmax=abs_max)
        im2 = axes[row, 2].imshow(diff, origin="lower", cmap="RdBu_r", norm=norm)
        rmse_val = np.sqrt(np.mean(diff ** 2))
        axes[row, 2].set_title(f"Difference (RMSE={rmse_val:.4f})")
        plt.colorbar(im2, ax=axes[row, 2], shrink=0.8, label=unit)

    fig.suptitle(f"{title_prefix}Prediction Comparison", fontsize=16, fontweight="bold", y=1.01)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"预测对比图已保存: {save_path}")
    plt.close(fig)


# ============================================================
# 图 2: RMSE 增长曲线（逐时效）
# ============================================================

def plot_rmse_curves(
    rmse_data: np.ndarray,
    var_names: Optional[List[str]] = None,
    time_interval_hours: int = 3,
    title: str = "RMSE vs Lead Time",
    save_path: Optional[str] = None,
):
    """
    绘制 RMSE 随预测时效的增长曲线

    rmse_data: (T, C) 逐时效逐变量的 RMSE
    var_names: 要绘制的变量名列表，默认使用重点变量
    """
    T, C = rmse_data.shape
    lead_times = np.arange(1, T + 1) * time_interval_hours  # 小时

    if var_names is None:
        var_names = list(KEY_VARS.keys())

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    # 按类型分组绘制
    # 图1: 地面变量
    ax = axes[0]
    sfc_vars = ["msl", "t2m", "sp"]
    for var in sfc_vars:
        if var in KEY_VARS and KEY_VARS[var] < C:
            base = get_var_base_name(var)
            ax.plot(lead_times, rmse_data[:, KEY_VARS[var]],
                    marker="o", markersize=4, label=f"{DISPLAY_NAMES.get(base, var)} ({var})")
    ax.set_xlabel("Lead Time (h)")
    ax.set_ylabel("RMSE")
    ax.set_title("Surface Scalar Variables")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # 图2: 风场变量
    ax = axes[1]
    wind_vars = ["u10m", "v10m", "u_250"]
    for var in wind_vars:
        if var in KEY_VARS and KEY_VARS[var] < C:
            base = get_var_base_name(var)
            ax.plot(lead_times, rmse_data[:, KEY_VARS[var]],
                    marker="s", markersize=4, label=f"{DISPLAY_NAMES.get(base, var)} ({var})")
    ax.set_xlabel("Lead Time (h)")
    ax.set_ylabel("RMSE (m/s)")
    ax.set_title("Wind Variables")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # 图3: 高空变量
    ax = axes[2]
    upper_vars = ["z_500", "t_850"]
    for var in upper_vars:
        if var in KEY_VARS and KEY_VARS[var] < C:
            base = get_var_base_name(var)
            ax.plot(lead_times, rmse_data[:, KEY_VARS[var]],
                    marker="^", markersize=4, label=f"{DISPLAY_NAMES.get(base, var)} ({var})")
    ax.set_xlabel("Lead Time (h)")
    ax.set_ylabel("RMSE")
    ax.set_title("Upper-air Variables")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # 图4: 所有变量 RMSE 均值
    ax = axes[3]
    mean_rmse = rmse_data.mean(axis=1)
    ax.plot(lead_times, mean_rmse, "k-", marker="D", markersize=5, linewidth=2, label="All-variable Mean")
    ax.fill_between(
        lead_times,
        rmse_data.min(axis=1),
        rmse_data.max(axis=1),
        alpha=0.2, color="gray", label="Min-Max Range",
    )
    ax.set_xlabel("Lead Time (h)")
    ax.set_ylabel("RMSE (normalized)")
    ax.set_title("All-variable Mean RMSE")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=16, fontweight="bold")
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"RMSE 曲线已保存: {save_path}")
    plt.close(fig)


# ============================================================
# 图 3: 空间误差分布
# ============================================================

def plot_spatial_error(
    pred: np.ndarray,
    truth: np.ndarray,
    var_indices: Optional[Dict[str, int]] = None,
    title_prefix: str = "",
    save_path: Optional[str] = None,
):
    """
    可视化预测误差的空间分布（RMSE 热图 + 绝对误差热图）

    pred:  (C, H, W) 或 (T, C, H, W) 物理量空间
    truth: (C, H, W) 或 (T, C, H, W) 物理量空间
    """
    if var_indices is None:
        var_indices = KEY_VARS

    # 如果有多个时间步，计算时间平均
    if pred.ndim == 4:
        # (T, C, H, W) -> 对 T 计算 RMSE
        spatial_rmse = np.sqrt(np.mean((pred - truth) ** 2, axis=0))  # (C, H, W)
    else:
        spatial_rmse = np.abs(pred - truth)  # (C, H, W)

    n_vars = len(var_indices)
    n_cols = min(4, n_vars)
    n_rows = (n_vars + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)

    for i, (var_name, ch_idx) in enumerate(var_indices.items()):
        row = i // n_cols
        col = i % n_cols
        ax = axes[row, col]

        err = spatial_rmse[ch_idx]
        base_name = get_var_base_name(var_name)
        display = DISPLAY_NAMES.get(base_name, var_name)
        unit = UNITS.get(base_name, "")

        im = ax.imshow(err, origin="lower", cmap="hot_r")
        ax.set_title(f"{display}\n({var_name}, {unit})")
        plt.colorbar(im, ax=ax, shrink=0.8)

    # 隐藏多余子图
    for i in range(n_vars, n_rows * n_cols):
        row = i // n_cols
        col = i % n_cols
        axes[row, col].set_visible(False)

    fig.suptitle(f"{title_prefix}Spatial Error Distribution", fontsize=16, fontweight="bold")
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"空间误差图已保存: {save_path}")
    plt.close(fig)


# ============================================================
# 图 4: 集合预报可视化
# ============================================================

def plot_ensemble_spread(
    ensemble_mean: np.ndarray,
    ensemble_std: np.ndarray,
    truth: Optional[np.ndarray] = None,
    var_indices: Optional[Dict[str, int]] = None,
    title_prefix: str = "",
    save_path: Optional[str] = None,
):
    """
    集合预报: 均值场 + 离散度(spread)

    ensemble_mean: (C, H, W)
    ensemble_std:  (C, H, W)
    truth: (C, H, W) 可选
    """
    if var_indices is None:
        var_indices = {k: v for k, v in list(KEY_VARS.items())[:4]}

    n_vars = len(var_indices)
    n_cols = 3 if truth is not None else 2
    fig, axes = plt.subplots(n_vars, n_cols, figsize=(5 * n_cols, 4 * n_vars))

    if n_vars == 1:
        axes = axes.reshape(1, -1)

    for row, (var_name, ch_idx) in enumerate(var_indices.items()):
        base_name = get_var_base_name(var_name)
        display = DISPLAY_NAMES.get(base_name, var_name)
        unit = UNITS.get(base_name, "")

        mean_field = ensemble_mean[ch_idx]
        std_field = ensemble_std[ch_idx]

        col = 0
        # Ensemble mean
        im0 = axes[row, col].imshow(mean_field, origin="lower", cmap="RdYlBu_r")
        axes[row, col].set_title(f"Ensemble Mean: {display}")
        plt.colorbar(im0, ax=axes[row, col], shrink=0.8, label=unit)

        col = 1
        # Ensemble spread
        im1 = axes[row, col].imshow(std_field, origin="lower", cmap="YlOrRd")
        axes[row, col].set_title("Ensemble Spread")
        plt.colorbar(im1, ax=axes[row, col], shrink=0.8, label=unit)

        if truth is not None:
            col = 2
            t = truth[ch_idx]
            diff = mean_field - t
            abs_max = max(abs(diff.min()), abs(diff.max()))
            if abs_max < 1e-12:
                abs_max = 1.0
            norm = TwoSlopeNorm(vmin=-abs_max, vcenter=0, vmax=abs_max)
            im2 = axes[row, col].imshow(diff, origin="lower", cmap="RdBu_r", norm=norm)
            rmse_val = np.sqrt(np.mean(diff ** 2))
            axes[row, col].set_title(f"Mean - Truth (RMSE={rmse_val:.4f})")
            plt.colorbar(im2, ax=axes[row, col], shrink=0.8, label=unit)

    fig.suptitle(f"{title_prefix}Ensemble Visualization", fontsize=16, fontweight="bold", y=1.01)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"集合预报图已保存: {save_path}")
    plt.close(fig)


# ============================================================
# 图 5: 自回归多步预测序列
# ============================================================

def plot_autoregressive_sequence(
    predictions: np.ndarray,
    var_idx: int,
    var_name: str = "",
    time_interval_hours: int = 3,
    max_show_steps: int = 8,
    save_path: Optional[str] = None,
):
    """
    绘制自回归预测的时间演化序列

    predictions: (T, C, H, W) 多步预测结果
    var_idx: 要展示的变量通道索引
    max_show_steps: 最多展示多少个时间步
    """
    T = predictions.shape[0]
    # 均匀选择展示步
    if T > max_show_steps:
        indices = np.linspace(0, T - 1, max_show_steps, dtype=int)
    else:
        indices = np.arange(T)

    n = len(indices)
    n_cols = min(4, n)
    n_rows = (n + n_cols - 1) // n_cols

    base_name = get_var_base_name(var_name) if var_name else ""
    display = DISPLAY_NAMES.get(base_name, var_name) if var_name else f"Channel {var_idx}"
    unit = UNITS.get(base_name, "") if var_name else ""

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)

    # 统一色标
    all_vals = predictions[indices, var_idx]
    vmin, vmax = all_vals.min(), all_vals.max()

    for i, t_idx in enumerate(indices):
        row = i // n_cols
        col = i % n_cols
        ax = axes[row, col]

        field = predictions[t_idx, var_idx]
        lead_h = (t_idx + 1) * time_interval_hours

        im = ax.imshow(field, origin="lower", cmap="RdYlBu_r", vmin=vmin, vmax=vmax)
        ax.set_title(f"+{lead_h}h (Step {t_idx+1})")
        plt.colorbar(im, ax=ax, shrink=0.8)

    # 隐藏多余子图
    for i in range(n, n_rows * n_cols):
        row = i // n_cols
        col = i % n_cols
        axes[row, col].set_visible(False)

    fig.suptitle(f"{display} Autoregressive Forecast Sequence", fontsize=16, fontweight="bold")
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"自回归序列图已保存: {save_path}")
    plt.close(fig)


# ============================================================
# 图 6: 逐变量 RMSE 柱状图
# ============================================================

def plot_variable_rmse_bar(
    rmse_per_var: np.ndarray,
    channel_names: Optional[List[str]] = None,
    title: str = "RMSE per Variable",
    save_path: Optional[str] = None,
):
    """
    逐变量 RMSE 柱状图

    rmse_per_var: (C,) 每个变量的 RMSE
    """
    C = len(rmse_per_var)
    if channel_names is None:
        channel_names = CHANNEL_NAMES[:C]

    fig, ax = plt.subplots(figsize=(max(14, C * 0.5), 6))

    colors = plt.cm.tab20(np.linspace(0, 1, C))
    bars = ax.bar(range(C), rmse_per_var, color=colors, edgecolor="gray", linewidth=0.5)

    ax.set_xticks(range(C))
    ax.set_xticklabels(channel_names, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("RMSE")
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.grid(True, axis="y", alpha=0.3)

    # 在柱顶标注数值
    for bar, val in zip(bars, rmse_per_var):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height(),
            f"{val:.3f}", ha="center", va="bottom", fontsize=6, rotation=45,
        )

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"变量RMSE柱状图已保存: {save_path}")
    plt.close(fig)


# ============================================================
# 辅助函数: 反归一化
# ============================================================

def denormalize(data: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """
    反归一化

    data: (..., C, H, W)
    mean: (C,)
    std: (C,)
    """
    std = np.where(std < 1e-8, 1.0, std)
    # 广播: mean/std -> (..., C, 1, 1)
    ndim = data.ndim
    shape = [1] * (ndim - 3) + [len(mean), 1, 1]
    mean = mean.reshape(shape)
    std = std.reshape(shape)
    return data * std + mean


# ============================================================
# 主入口
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="ERA5-Diffusion 可视化")
    parser.add_argument("--pred_dir", type=str, default="outputs",
                        help="预测结果目录（包含 .pt 文件）")
    parser.add_argument("--output_dir", type=str, default="figures",
                        help="图片输出目录")
    parser.add_argument("--norm_stats", type=str, default="norm_stats.pt",
                        help="归一化统计文件路径")
    parser.add_argument("--mode", type=str, default="all",
                        choices=["single", "autoregressive", "ensemble", "all"],
                        help="可视化模式")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # 加载归一化统计
    norm_mean, norm_std = None, None
    if os.path.exists(args.norm_stats):
        stats = torch.load(args.norm_stats, weights_only=True)
        norm_mean = stats["mean"].numpy()
        norm_std = stats["std"].numpy()
        logger.info(f"已加载归一化统计: {args.norm_stats}")
    else:
        logger.warning(f"找不到归一化统计文件 {args.norm_stats}，将使用归一化空间数据")

    # ========== 单步预测可视化 ==========
    if args.mode in ("single", "all"):
        import glob
        single_files = sorted(glob.glob(os.path.join(args.pred_dir, "single_pred_*.pt")))
        for f in single_files:
            logger.info(f"处理单步预测: {f}")
            data = torch.load(f, weights_only=False)

            pred = data["prediction"].numpy()  # (1, C_target, 40, 40)
            gt = data["ground_truth"].numpy()    # (1, C_target, 40, 40)
            tid = data.get("typhoon_id", "unknown")

            # 取第一个 batch 和第一个预测时间步
            n_ch = len(CHANNEL_NAMES)  # 15
            pred_step1 = pred[0, :n_ch]  # (15, 40, 40)
            gt_step1 = gt[0, :n_ch]

            basename = os.path.splitext(os.path.basename(f))[0]

            # 预测对比图
            plot_prediction_comparison(
                pred_step1, gt_step1,
                title_prefix=f"Typhoon {tid} | ",
                save_path=os.path.join(args.output_dir, f"{basename}_comparison.png"),
            )

            # 空间误差分布
            plot_spatial_error(
                pred_step1, gt_step1,
                title_prefix=f"Typhoon {tid} | ",
                save_path=os.path.join(args.output_dir, f"{basename}_spatial_error.png"),
            )

            # 逐变量 RMSE 柱状图
            rmse_per_var = np.sqrt(np.mean((pred_step1 - gt_step1) ** 2, axis=(1, 2)))
            plot_variable_rmse_bar(
                rmse_per_var,
                title=f"Typhoon {tid} - RMSE per Variable (+3h)",
                save_path=os.path.join(args.output_dir, f"{basename}_var_rmse.png"),
            )

    # ========== 自回归预测可视化 ==========
    if args.mode in ("autoregressive", "all"):
        import glob
        ar_files = sorted(glob.glob(os.path.join(args.pred_dir, "ar_pred_*.pt")))

        # 收集所有样本的 RMSE 用于绘制增长曲线
        all_rmse = []

        for f in ar_files:
            logger.info(f"处理自回归预测: {f}")
            data = torch.load(f, weights_only=False)

            preds = data["predictions"].numpy()  # (1, T, C, H, W)
            tid = data.get("typhoon_id", "unknown")

            preds_b0 = preds[0]  # (T, C, H, W)
            T = preds_b0.shape[0]
            basename = os.path.splitext(os.path.basename(f))[0]

            # 自回归序列（展示 msl 海平面气压）
            n_ch = len(CHANNEL_NAMES)
            msl_idx = KEY_VARS.get("msl", CHANNEL_NAMES.index("msl"))
            plot_autoregressive_sequence(
                preds_b0, msl_idx, var_name="msl",
                save_path=os.path.join(args.output_dir, f"{basename}_msl_sequence.png"),
            )

            # 展示 z_500 位势高度
            z500_idx = KEY_VARS.get("z_500", CHANNEL_NAMES.index("z_500"))
            plot_autoregressive_sequence(
                preds_b0, z500_idx, var_name="z_500",
                save_path=os.path.join(args.output_dir, f"{basename}_z500_sequence.png"),
            )

        # 如果有多个样本的 RMSE 数据，绘制 RMSE 增长曲线
        # 注意：自回归模式保存的结果中没有真值，这里只展示预测序列
        # 若需要 RMSE 曲线，需在推理时同时保存真值
        logger.info("注意: 自回归 RMSE 增长曲线需要真值数据，请在推理脚本中同时保存真值")

    # ========== 集合预报可视化 ==========
    if args.mode in ("ensemble", "all"):
        import glob
        ens_files = sorted(glob.glob(os.path.join(args.pred_dir, "ensemble_pred_*.pt")))

        for f in ens_files:
            logger.info(f"处理集合预报: {f}")
            data = torch.load(f, weights_only=False)

            ens_mean = data["ensemble_mean"].numpy()  # (B, T, 34, H, W) 或 (B, 102, H, W)
            ens_std = data["ensemble_std"].numpy()
            tid = data.get("typhoon_id", "unknown")
            basename = os.path.splitext(os.path.basename(f))[0]

            # 取第一个 batch
            mean_b0 = ens_mean[0]
            std_b0 = ens_std[0]

            # 如果是自回归集合 (T, 34, H, W)，取第一个时间步
            if mean_b0.ndim == 3:
                # (T*34, H, W) -> 取前 34 通道
                mean_show = mean_b0[:34]
                std_show = std_b0[:34]
            elif mean_b0.ndim == 4:
                # (T, 34, H, W) -> 取第一个时间步
                mean_show = mean_b0[0]
                std_show = std_b0[0]
            else:
                logger.warning(f"集合数据维度异常: {mean_b0.shape}")
                continue

            plot_ensemble_spread(
                mean_show, std_show,
                title_prefix=f"Typhoon {tid} | ",
                save_path=os.path.join(args.output_dir, f"{basename}_ensemble.png"),
            )

    logger.info(f"所有可视化完成! 图片保存在: {args.output_dir}")


if __name__ == "__main__":
    main()
