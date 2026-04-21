"""
可视化模块 — 生成论文所需图表

图表清单:
1. PSD动能谱对比图 — 证明FM细节保留更好
2. NFE效率曲线 — 证明FM推理速度优势
3. 逐时效RMSE对比 — 各通道分时效对比
4. 散度/地转平衡对比 — 物理一致性
5. 台风个例空间分布 — 台风眼壁结构
6. 路径直线性分析 — FM的ODE路径优势
"""
import os
import json
import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap

logger = logging.getLogger(__name__)

# 设置中文字体
plt.rcParams['font.family'] = ['DejaVu Sans', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False


def plot_psd_comparison(
    results: Dict,
    save_path: str,
    title: str = "Kinetic Energy Spectrum Comparison",
):
    """
    绘制PSD动能谱对比图 (论文核心Figure)

    预期发现:
    - 真值谱斜率接近 -5/3 (Kolmogorov)
    - Diffusion 在高频部分能量衰减 (过平滑)
    - Flow Matching 更好地保持谱斜率
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 左图: 动能谱
    ax = axes[0]

    if "FM_spectral" in results:
        k = np.array(results["FM_spectral"]["k"])
        psd_fm = np.array(results["FM_spectral"]["psd_pred"])
        ax.plot(k, psd_fm, 'b-', linewidth=2, label='FM (Flow Matching)', marker='o', markersize=3)

    if "DM_spectral" in results:
        k = np.array(results["DM_spectral"]["k"])
        psd_dm = np.array(results["DM_spectral"]["psd_pred"])
        ax.plot(k, psd_dm, 'r--', linewidth=2, label='DM (Diffusion)', marker='s', markersize=3)

    # 真值
    if "FM_spectral" in results:
        psd_gt = np.array(results["FM_spectral"]["psd_gt"])
        ax.plot(k, psd_gt, 'k-', linewidth=2.5, label='ERA5 (Ground Truth)')

    # 理论斜率参考线
    k_ref = np.linspace(k.min(), k.max(), 50)
    slope_53 = -5/3
    slope_3 = -3
    ref_psd = psd_gt.max() * (k_ref / k_ref.mean()) ** slope_53
    ax.plot(k_ref, ref_psd, 'g:', linewidth=1.5, label=f'k^{slope_53:.2f} (Kolmogorov)')

    ax.set_xlabel('Wavenumber k (cycles/degree)', fontsize=12)
    ax.set_ylabel('Normalized PSD', fontsize=12)
    ax.set_title('Kinetic Energy Spectrum', fontsize=14)
    ax.legend(fontsize=10)
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(k.min() * 0.8, k.max() * 1.2)

    # 右图: 谱斜率对比
    ax2 = axes[1]
    methods = []
    slopes = []
    colors = []

    if "FM_spectral" in results:
        methods.append('FM')
        slopes.append(results["FM_spectral"]["spectral_slope_pred"])
        colors.append('blue')
    if "DM_spectral" in results:
        methods.append('DM')
        slopes.append(results["DM_spectral"]["spectral_slope_pred"])
        colors.append('red')

    if "FM_spectral" in results:
        methods.append('ERA5')
        slopes.append(results["FM_spectral"]["spectral_slope_gt"])
        colors.append('gray')

    bars = ax2.bar(methods, slopes, color=colors, alpha=0.7, edgecolor='black')
    ax2.axhline(y=-5/3, color='green', linestyle=':', linewidth=2, label=f'-5/3 (Kolmogorov)')
    ax2.axhline(y=-3, color='orange', linestyle=':', linewidth=2, label=f'-3 (Charney)')

    ax2.set_ylabel('Spectral Slope', fontsize=12)
    ax2.set_title('Spectral Slope Comparison', fontsize=14)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3, axis='y')

    plt.suptitle(title, fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"PSD对比图已保存: {save_path}")


def plot_nfe_efficiency_curve(
    nfe_results: Dict,
    save_path: str,
):
    """
    绘制NFE效率曲线 (论文核心Figure)

    X轴: NFE (Number of Function Evaluations)
    Y轴: RMSE

    核心卖点: FM仅需4步即可达到DM 50步的精度
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # 左图: NFE vs RMSE
    ax = axes[0]

    if "FM" in nfe_results:
        fm_data = nfe_results["FM"]
        fm_steps = sorted(fm_data.keys())
        fm_rmses = [fm_data[s]["rmse_mean"] for s in fm_steps]
        fm_stds = [fm_data[s]["rmse_std"] for s in fm_steps]
        ax.errorbar(fm_steps, fm_rmses, yerr=fm_stds, fmt='bo-', linewidth=2,
                    markersize=8, capsize=5, label='FM (Flow Matching)')

    if "DM" in nfe_results:
        dm_data = nfe_results["DM"]
        dm_steps = sorted(dm_data.keys())
        dm_rmses = [dm_data[s]["rmse_mean"] for s in dm_steps]
        dm_stds = [dm_data[s]["rmse_std"] for s in dm_steps]
        ax.errorbar(dm_steps, dm_rmses, yerr=dm_stds, fmt='rs--', linewidth=2,
                    markersize=8, capsize=5, label='DM (Diffusion)')

    ax.set_xlabel('NFE (Number of Function Evaluations)', fontsize=12)
    ax.set_ylabel('RMSE', fontsize=12)
    ax.set_title('Inference Efficiency vs Accuracy', fontsize=14)
    ax.legend(fontsize=11)
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3)

    # 右图: 加速比
    ax2 = axes[1]

    if "FM" in nfe_results and "DM" in nfe_results:
        fm_data = nfe_results["FM"]
        dm_data = nfe_results["DM"]

        # 找DM的RMSE at 50步作为基准
        dm_50_rmse = dm_data.get(50, {}).get("rmse_mean", None)
        if dm_50_rmse is None:
            dm_50_rmse = dm_data.get(max(dm_data.keys()), {}).get("rmse_mean", 1.0)

        # 计算FM各步对应的"等效DM步数"
        speedups = {}
        for step in sorted(fm_data.keys()):
            fm_rmse = fm_data[step]["rmse_mean"]
            # 找最接近FM_4step精度的DM步数
            closest_dm_step = min(dm_data.keys(), key=lambda x: abs(dm_data[x]["rmse_mean"] - fm_rmse))
            speedup = closest_dm_step / step if step > 0 else 0
            speedups[step] = speedup

        steps = sorted(speedups.keys())
        speedup_vals = [speedups[s] for s in steps]

        bars = ax2.bar([f"FM {s}步" for s in steps], speedup_vals,
                       color=['blue' if s >= 4 else 'lightblue' for s in steps],
                       alpha=0.7, edgecolor='black')
        ax2.axhline(y=1, color='gray', linestyle='--', linewidth=1, label='No speedup')
        ax2.axhline(y=50/4, color='red', linestyle=':', linewidth=2, label='FM 4-step vs DM 50-step')

        for bar, val in zip(bars, speedup_vals):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{val:.1f}x', ha='center', va='bottom', fontsize=10)

        ax2.set_ylabel('Speedup vs DM (NFE ratio)', fontsize=12)
        ax2.set_title('Inference Speedup', fontsize=14)
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3, axis='y')

    plt.suptitle('Sampling Efficiency: FM vs DM', fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"NFE效率曲线已保存: {save_path}")


def plot_channel_rmse_comparison(
    results: Dict,
    save_path: str,
    var_names: List[str] = None,
):
    """
    绘制分通道RMSE对比柱状图
    """
    if var_names is None:
        var_names = ['u_850', 'u_500', 'u_250', 'v_850', 'v_500', 'v_250', 'z_850', 'z_500', 'z_250']

    fig, ax = plt.subplots(figsize=(14, 6))

    x = np.arange(len(var_names))
    width = 0.35

    fm_rmse = [results.get("FM", {}).get("rmse_per_channel", {}).get(v, 0) for v in var_names]
    dm_rmse = [results.get("DM", {}).get("rmse_per_channel", {}).get(v, 0) for v in var_names]

    if any(fm_rmse):
        bars1 = ax.bar(x - width/2, fm_rmse, width, label='FM', color='steelblue', alpha=0.8)
    if any(dm_rmse):
        bars2 = ax.bar(x + width/2, dm_rmse, width, label='DM', color='indianred', alpha=0.8)

    ax.set_xlabel('Variable', fontsize=12)
    ax.set_ylabel('RMSE', fontsize=12)
    ax.set_title('Per-Channel RMSE Comparison', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(var_names, rotation=45, ha='right')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')

    # 标注优劣
    if any(fm_rmse) and any(dm_rmse):
        for i, (f, d) in enumerate(zip(fm_rmse, dm_rmse)):
            if f < d:
                ax.annotate('FM', xy=(i - width/2, f), xytext=(i - width/2, f * 0.95),
                           ha='center', fontsize=8, color='blue')
            else:
                ax.annotate('DM', xy=(i + width/2, d), xytext=(i + width/2, d * 0.95),
                           ha='center', fontsize=8, color='red')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"分通道RMSE对比图已保存: {save_path}")


def plot_lat_weighted_rmse_heatmap(
    results: Dict,
    save_path: str,
    var_names: List[str] = None,
):
    """
    绘制纬度加权RMSE热力图
    """
    if var_names is None:
        var_names = ['u_850', 'u_500', 'u_250', 'v_850', 'v_500', 'v_250', 'z_850', 'z_500', 'z_250']

    methods = []
    if "FM" in results:
        methods.append("FM")
    if "DM" in results:
        methods.append("DM")

    if len(methods) < 2:
        logger.warning("需要FM和DM结果才能绘制对比热力图")
        return

    fig, axes = plt.subplots(1, len(methods), figsize=(5 * len(methods), 6), squeeze=False)

    data_matrix = []
    for m in methods:
        row = [results[m].get("lat_weighted_rmse_per_channel", {}).get(v, 0) for v in var_names]
        data_matrix.append(row)

    data_matrix = np.array(data_matrix)

    for idx, (m, data) in enumerate(zip(methods, data_matrix)):
        ax = axes[0, idx]
        im = ax.imshow(data.reshape(1, -1), cmap='YlOrRd', aspect='auto', vmin=0)
        ax.set_yticks([0])
        ax.set_yticklabels([m])
        ax.set_xticks(range(len(var_names)))
        ax.set_xticklabels(var_names, rotation=45, ha='right')
        ax.set_title(f'{m} Lat-Weighted RMSE')
        plt.colorbar(im, ax=ax, shrink=0.8)

    plt.suptitle('Latitude-Weighted RMSE Heatmap', fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"纬度加权RMSE热力图已保存: {save_path}")


def plot_temporal_evolution(
    fm_results: Dict,
    dm_results: Optional[Dict],
    save_path: str,
    lead_times: List[int] = None,
):
    """
    绘制时效演化曲线 — 各时效的RMSE变化
    """
    if lead_times is None:
        lead_times = list(range(24, 169, 24))

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    var_groups = [
        ('u_850', 'u_500', 'u_250'),
        ('v_850', 'v_500', 'v_250'),
        ('z_850', 'z_500', 'z_250'),
    ]

    for row_idx, (u_vars, v_vars, z_vars) in enumerate(var_groups):
        ax = axes[row_idx, 0]
        for var in u_vars:
            if fm_results and "FM" in fm_results:
                f_vals = [fm_results["FM"].get(f"rmse_{var}_t{t}", 0) for t in lead_times]
                ax.plot(lead_times, f_vals, 'b-o', label=f'FM {var}', markersize=4)
            if dm_results and "DM" in dm_results:
                d_vals = [dm_results["DM"].get(f"rmse_{var}_t{t}", 0) for t in lead_times]
                ax.plot(lead_times, d_vals, 'r--s', label=f'DM {var}', markersize=4)

        ax.set_title(f'{u_vars[0].split("_")[0]} Wind RMSE Evolution')
        ax.set_xlabel('Lead Time (h)')
        ax.set_ylabel('RMSE')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.suptitle('RMSE Temporal Evolution by Variable Group', fontsize=16)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"时效演化曲线已保存: {save_path}")


def plot_physics_consistency(
    results: Dict,
    save_path: str,
):
    """
    绘制物理一致性指标对比
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    metrics = []

    # Divergence RMSE
    ax = axes[0]
    methods = []
    div_values = []

    if "FM" in results:
        methods.append("FM")
        div_values.append(results["FM"].get("divergence_rmse_pred", 0))
    if "DM" in results:
        methods.append("DM")
        div_values.append(results["DM"].get("divergence_rmse_pred", 0))

    if results.get("FM", {}).get("divergence_rmse_gt"):
        gt_div = results["FM"]["divergence_rmse_gt"]
        methods.append("ERA5")
        div_values.append(gt_div)

    if methods:
        colors = ['blue' if m == 'FM' else 'red' if m == 'DM' else 'gray' for m in methods]
        ax.bar(methods, div_values, color=colors, alpha=0.7, edgecolor='black')
        ax.set_ylabel('Divergence RMSE')
        ax.set_title('Physical Consistency: Divergence')
        ax.grid(True, alpha=0.3, axis='y')

    # Bias
    ax = axes[1]
    if "FM" in results and "DM" in results:
        z_biases = ["z_850", "z_500", "z_250"]
        x = np.arange(len(z_biases))
        width = 0.35

        fm_bias = [results["FM"].get("bias_per_channel", {}).get(z, 0) for z in z_biases]
        dm_bias = [results["DM"].get("bias_per_channel", {}).get(z, 0) for z in z_biases]

        ax.bar(x - width/2, fm_bias, width, label='FM', color='steelblue', alpha=0.8)
        ax.bar(x + width/2, dm_bias, width, label='DM', color='indianred', alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(z_biases)
        ax.set_ylabel('Bias')
        ax.set_title('Z Channel Systematic Bias')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

    # High frequency energy ratio
    ax = axes[2]
    if "FM_spectral" in results and "DM_spectral" in results:
        hf_fm = results["FM_spectral"].get("high_freq_energy_ratio_pred", 0)
        hf_dm = results["DM_spectral"].get("high_freq_energy_ratio_pred", 0)
        hf_gt = results["FM_spectral"].get("high_freq_energy_ratio_gt", 0)

        methods = ["FM", "DM", "ERA5"]
        values = [hf_fm, hf_dm, hf_gt]
        colors = ['steelblue', 'indianred', 'gray']

        ax.bar(methods, values, color=colors, alpha=0.7, edgecolor='black')
        ax.set_ylabel('High-Freq Energy Ratio')
        ax.set_title('Spectral Fidelity: High-Frequency Energy')
        ax.grid(True, alpha=0.3, axis='y')

        for i, (m, v) in enumerate(zip(methods, values)):
            ax.text(i, v + 0.01, f'{v:.3f}', ha='center', fontsize=10)

    plt.suptitle('Physical Consistency Metrics', fontsize=16)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"物理一致性对比图已保存: {save_path}")


def plot_summary_table(
    results: Dict,
    save_path: str,
):
    """
    生成论文所需的汇总表格 (LaTeX格式)
    """
    lines = []
    lines.append("% ===========================================")
    lines.append("% FM vs DM Comparison Summary Table")
    lines.append("% ===========================================")
    lines.append("")
    lines.append("\\begin{table}[htbp]")
    lines.append("\\centering")
    lines.append("\\caption{Comparison of Flow Matching and Diffusion Models}")
    lines.append("\\begin{tabular}{lccc}")
    lines.append("\\hline")
    lines.append("\\textbf{Metric} & \\textbf{FM} & \\textbf{DM} & \\textbf{Difference} \\\\")
    lines.append("\\hline")

    # RMSE
    fm_rmse = results.get("FM", {}).get("rmse_mean", 0)
    dm_rmse = results.get("DM", {}).get("rmse_mean", 0)
    diff_rmse = fm_rmse - dm_rmse
    lines.append(f"RMSE (mean) & {fm_rmse:.4f} & {dm_rmse:.4f} & {diff_rmse:+.4f} \\\\")

    # Lat-Weighted RMSE
    fm_lat = results.get("FM", {}).get("lat_weighted_rmse_mean", 0)
    dm_lat = results.get("DM", {}).get("lat_weighted_rmse_mean", 0)
    diff_lat = fm_lat - dm_lat
    lines.append(f"Lat-Weighted RMSE & {fm_lat:.4f} & {dm_lat:.4f} & {diff_lat:+.4f} \\\\")

    # Spectral Slope
    if "FM_spectral" in results:
        fm_slope = results["FM_spectral"].get("spectral_slope_pred", 0)
        dm_slope = results.get("DM_spectral", {}).get("spectral_slope_pred", 0)
        gt_slope = results["FM_spectral"].get("spectral_slope_gt", 0)
        lines.append(f"Spectral Slope & {fm_slope:.3f} & {dm_slope:.3f} & -- \\\\")
        lines.append(f"GT Slope & \\multicolumn{{3}}{{c}}{{{gt_slope:.3f}}} \\\\")

    # Divergence
    if "FM" in results and "DM" in results:
        fm_div = results["FM"].get("divergence_rmse_pred", 0)
        dm_div = results["DM"].get("divergence_rmse_pred", 0)
        lines.append(f"Divergence RMSE & {fm_div:.4f} & {dm_div:.4f} & {fm_div-dm_div:+.4f} \\\\")

    lines.append("\\hline")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")

    with open(save_path.replace('.png', '.tex'), 'w') as f:
        f.write('\n'.join(lines))

    logger.info(f"汇总表格(LaTeX)已保存: {save_path.replace('.png', '.tex')}")


def visualize_comparison_results(
    results: Dict,
    output_dir: str,
):
    """生成所有可视化图表"""
    os.makedirs(output_dir, exist_ok=True)

    # 1. PSD对比图
    if "FM_spectral" in results or "DM_spectral" in results:
        plot_psd_comparison(
            results,
            os.path.join(output_dir, "fig1_psd_comparison.png"),
            title="Kinetic Energy Spectrum: FM vs DM"
        )

    # 2. NFE效率曲线
    if "NFE_efficiency" in results:
        plot_nfe_efficiency_curve(
            results["NFE_efficiency"],
            os.path.join(output_dir, "fig2_nfe_efficiency.png")
        )

    # 3. 分通道RMSE
    var_names = ['u_850', 'u_500', 'u_250', 'v_850', 'v_500', 'v_250', 'z_850', 'z_500', 'z_250']
    plot_channel_rmse_comparison(
        results,
        os.path.join(output_dir, "fig3_channel_rmse.png"),
        var_names=var_names
    )

    # 4. 物理一致性
    plot_physics_consistency(
        results,
        os.path.join(output_dir, "fig4_physics_consistency.png")
    )

    # 5. 汇总表格
    plot_summary_table(
        results,
        os.path.join(output_dir, "summary_table.png")
    )

    logger.info(f"所有可视化图表已生成在: {output_dir}")
