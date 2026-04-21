"""
NFE Sweep Evaluation — Publication-Ready FM vs DM Efficiency Comparison

Evaluates both FM and DM across varying NFE (Number of Function Evaluations)
to generate the efficiency-accuracy trade-off curve for the paper.

Generates:
1. Table 3: NFE vs. ADE vs. Inference Time
2. Figure 1: NFE vs. ADE & Time trade-off (Pareto frontier)
3. Speedup factor table

This is an EVALUATION-ONLY script — requires pre-trained checkpoints.
For training + evaluation, use run_multi_seed.py.

Usage:
    # Evaluate FM with different Euler steps
    python run_nfe_sweep.py --method fm --euler_steps 1 4 8 16 --skip_train

    # Evaluate DM with different DDIM steps
    python run_nfe_sweep.py --method dm --ddim_steps 10 25 50 100 --skip_train

    # Full comparison (both methods, all NFE values)
    python run_nfe_sweep.py --both --skip_train --output_dir nfe_sweep_results/
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

script_dir = os.path.dirname(os.path.abspath(__file__))
pkg_root = os.path.dirname(script_dir)
if pkg_root not in sys.path:
    sys.path.insert(0, pkg_root)

from compare_fm_dm.configs import get_config
from compare_fm_dm.data.dataset import build_dataloaders
from compare_fm_dm.models.unified_model import create_model
from compare_fm_dm.evaluation.metrics import ComparisonEvaluator, compute_rmse, compute_mae


def set_seed(seed: int):
    """Set all random seeds for reproducibility."""
    import random
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def load_model(
    checkpoint_path: str,
    method: str,
    data_cfg,
    model_cfg,
    train_cfg,
    device: str = "cuda",
) -> Tuple:
    """
    Load a pre-trained UnifiedModel (FM or DM) from checkpoint.

    Returns:
        (model, norm_mean, norm_std)
    """
    device = torch.device(device if torch.cuda.is_available() else "cpu")

    model = create_model(model_cfg, data_cfg, train_cfg, method=method)
    model = model.to(device)

    if checkpoint_path and os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        if "ema_state_dict" in ckpt:
            model.load_state_dict(ckpt["ema_state_dict"], strict=False)
            print(f"[模型] 已加载 EMA 参数: {checkpoint_path}")
        elif "model_state_dict" in ckpt:
            model.load_state_dict(ckpt["model_state_dict"], strict=False)
            print(f"[模型] 已加载参数: {checkpoint_path}")
        else:
            model.load_state_dict(ckpt, strict=False)
            print(f"[模型] 已加载参数 (直接): {checkpoint_path}")
    else:
        print(f"[警告] Checkpoint 不存在: {checkpoint_path}，使用随机初始化")

    model.eval()
    return model, device


def evaluate_nfe_single(
    model,
    test_loader,
    method: str,
    nfe_steps: int,
    device: torch.device,
    num_eval_samples: int = 100,
    clamp_range: Optional[Tuple[float, float]] = None,  # was (-5.0, 5.0)
    z_clamp_range: Optional[Tuple[float, float]] = None,  # was (-1.0, 1.0)
) -> Dict:
    """
    Evaluate model at a specific NFE setting.

    Returns:
        Dict with ADE (km), inference time (ms), and NFE
    """
    model.eval()

    n_vars = len(test_loader.dataset.data_cfg.var_names)
    all_rmse = []
    all_mae = []
    inference_times = []
    n_processed = 0

    print(f"  NFE={nfe_steps}: 开始评估 (上限 {num_eval_samples} 样本)...")

    for batch_idx, batch in enumerate(test_loader):
        if n_processed >= num_eval_samples:
            break

        condition = batch["condition"].to(device)
        target = batch["target"].to(device)
        B = condition.shape[0]

        start_time = time.perf_counter()

        with torch.no_grad():
            if method == "fm":
                pred = model.sample_fm(
                    condition,
                    device,
                    euler_steps=nfe_steps,
                    euler_mode="midpoint",
                    clamp_range=clamp_range,
                    z_clamp_range=z_clamp_range,
                )
            else:
                pred = model.sample_dm(
                    condition,
                    device,
                    ddim_steps=nfe_steps,
                    clamp_range=clamp_range,
                    z_clamp_range=z_clamp_range,
                )

        torch.cuda.synchronize() if device.type == "cuda" else None
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0 / B

        pred_phys = denormalize(pred, test_loader.dataset.norm_mean, test_loader.dataset.norm_std, device)
        target_phys = denormalize(target, test_loader.dataset.norm_mean, test_loader.dataset.norm_std, device)

        rmse_vals = compute_rmse(pred_phys, target_phys).cpu().numpy()
        mae_vals = compute_mae(pred_phys, target_phys).cpu().numpy()

        all_rmse.extend(rmse_vals.tolist())
        all_mae.extend(mae_vals.tolist())
        inference_times.append(elapsed_ms)
        n_processed += B

        if n_processed % 200 == 0:
            print(f"    已处理 {n_processed} 样本...")

    result = {
        "method": method,
        "nfe": nfe_steps,
        "rmse_per_channel": all_rmse,
        "mae_per_channel": all_mae,
        "rmse_mean": float(np.mean(all_rmse)),
        "rmse_std": float(np.std(all_rmse)),
        "mae_mean": float(np.mean(all_mae)),
        "mae_std": float(np.std(all_mae)),
        "inference_time_ms_mean": float(np.mean(inference_times)),
        "inference_time_ms_std": float(np.std(inference_times)),
        "n_samples": n_processed,
    }

    print(f"  NFE={nfe_steps}: RMSE={result['rmse_mean']:.4f} ± {result['rmse_std']:.4f}, "
          f"Time={result['inference_time_ms_mean']:.1f} ± {result['inference_time_ms_std']:.1f} ms")

    return result


def denormalize(
    data: torch.Tensor,
    norm_mean,
    norm_std,
    device: torch.device,
) -> torch.Tensor:
    """Denormalize using dataset statistics."""
    if hasattr(norm_mean, 'numpy'):
        mean_t = torch.from_numpy(norm_mean.numpy()).float().to(device)
        std_t = torch.from_numpy(norm_std.numpy()).float().to(device)
    else:
        mean_t = torch.from_numpy(norm_mean).float().to(device)
        std_t = torch.from_numpy(norm_std).float().to(device)

    std_t = torch.where(std_t < 1e-8, torch.ones_like(std_t), std_t)

    if data.ndim == 4:
        mean_t = mean_t.reshape(1, -1, 1, 1)
        std_t = std_t.reshape(1, -1, 1, 1)

    return data * std_t + mean_t


def run_nfe_sweep(
    method: str,
    nfe_list: List[int],
    data_cfg,
    model_cfg,
    train_cfg,
    device: str = "cuda",
    checkpoint_path: str = "",
    num_eval_samples: int = 100,
    output_dir: str = "nfe_sweep_results",
) -> List[Dict]:
    """
    Run NFE sweep for a single method (FM or DM).

    Returns:
        List of result dicts, one per NFE value
    """
    os.makedirs(output_dir, exist_ok=True)

    _, _, test_loader, norm_mean, norm_std = build_dataloaders(data_cfg, train_cfg)
    test_loader.dataset.norm_mean = norm_mean
    test_loader.dataset.norm_std = norm_std
    test_loader.dataset.data_cfg = data_cfg

    model, device = load_model(checkpoint_path, method, data_cfg, model_cfg, train_cfg, device)

    results = []
    for nfe in nfe_list:
        result = evaluate_nfe_single(
            model=model,
            test_loader=test_loader,
            method=method,
            nfe_steps=nfe,
            device=device,
            num_eval_samples=num_eval_samples,
        )
        results.append(result)

    method_name = "fm" if method == "fm" else "dm"
    results_path = os.path.join(output_dir, f"nfe_sweep_{method_name}.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"[NFE Sweep] {method.upper()} 结果已保存: {results_path}")

    return results


def compute_speedup_and_pareto(
    fm_results: List[Dict],
    dm_results: List[Dict],
    baseline_dm_nfe: int = 50,
) -> Dict:
    """
    Compute speedup factors and identify Pareto frontier.

    Returns:
        Dict with speedup table and Pareto analysis
    """
    dm_baseline = next((r for r in dm_results if r["nfe"] == baseline_dm_nfe), dm_results[0])
    baseline_time = dm_baseline["inference_time_ms_mean"]
    baseline_rmse = dm_baseline["rmse_mean"]

    table = []
    for r in sorted(fm_results, key=lambda x: x["nfe"]):
        speedup = baseline_time / max(r["inference_time_ms_mean"], 0.1)
        delta_rmse = r["rmse_mean"] - baseline_rmse
        table.append({
            "method": "FM",
            "nfe": r["nfe"],
            "rmse_mean": r["rmse_mean"],
            "rmse_std": r["rmse_std"],
            "time_ms": r["inference_time_ms_mean"],
            "time_std": r["inference_time_ms_std"],
            "speedup_vs_dm50": speedup,
            "delta_rmse_vs_dm50": delta_rmse,
        })

    for r in sorted(dm_results, key=lambda x: x["nfe"]):
        speedup = baseline_time / max(r["inference_time_ms_mean"], 0.1)
        delta_rmse = r["rmse_mean"] - baseline_rmse
        table.append({
            "method": "DM",
            "nfe": r["nfe"],
            "rmse_mean": r["rmse_mean"],
            "rmse_std": r["rmse_std"],
            "time_ms": r["inference_time_ms_mean"],
            "time_std": r["inference_time_ms_std"],
            "speedup_vs_dm50": speedup,
            "delta_rmse_vs_dm50": delta_rmse,
        })

    return {
        "baseline_dm_nfe": baseline_dm_nfe,
        "baseline_time_ms": baseline_time,
        "baseline_rmse": baseline_rmse,
        "table": table,
    }


def print_nfe_table(results: List[Dict], method: str):
    """Print a formatted NFE sweep table."""
    print(f"\n{'=' * 80}")
    print(f"NFE Sweep Results — {method.upper()}")
    print(f"{'=' * 80}")
    print(f"{'NFE':>6} {'RMSE':>12} {'MAE':>12} {'Time (ms)':>12} {'Speedup':>10}")
    print("-" * 56)
    for r in results:
        print(
            f"{r['nfe']:>6} "
            f"{r['rmse_mean']:>10.4f} ± {r['rmse_std']:.4f} "
            f"{r['mae_mean']:>10.4f} ± {r['mae_std']:.4f} "
            f"{r['inference_time_ms_mean']:>10.1f} ± {r['inference_time_ms_std']:.1f} "
            f"{(1.0 / r['inference_time_ms_mean'] * 100 if r['inference_time_ms_mean'] > 0 else 0):>8.1f}x"
        )
    print(f"{'=' * 80}")


def print_comparison_table(fm_results: List[Dict], dm_results: List[Dict]):
    """Print the paper-style comparison table."""
    print("\n" + "=" * 100)
    print("Table 3: NFE vs. ADE vs. Inference Time (Publication Format)")
    print("=" * 100)
    print(
        f"{'Method':>6} {'NFE':>6} {'ADE (km)':>12} {'Time (ms)':>12} {'Speedup vs DM-50':>20} {'Δ ADE vs DM-50':>18}"
    )
    print("-" * 100)

    dm_baseline = next((r for r in dm_results if r["nfe"] == 50), dm_results[-1])
    baseline_time = dm_baseline["inference_time_ms_mean"]
    baseline_rmse = dm_baseline["rmse_mean"]

    all_results = sorted(
        [(r, "FM") for r in fm_results] + [(r, "DM") for r in dm_results],
        key=lambda x: x[0]["nfe"],
    )

    for r, method in all_results:
        speedup = baseline_time / max(r["inference_time_ms_mean"], 0.1)
        delta = r["rmse_mean"] - baseline_rmse
        sign = "+" if delta > 0 else ""
        print(
            f"{method:>6} "
            f"{r['nfe']:>6} "
            f"{r['rmse_mean']:>10.4f} ± {r['rmse_std']:.4f} "
            f"{r['inference_time_ms_mean']:>10.1f} ± {r['inference_time_ms_std']:.1f} "
            f"{speedup:>18.1f}× "
            f"{sign}{delta:>15.4f}"
        )

    print("-" * 100)
    print("Note: Speedup = Time(DM-50) / Time(current). Δ = ADE(current) - ADE(DM-50)")
    print("=" * 100)


def parse_args():
    parser = argparse.ArgumentParser(
        description="NFE Sweep Evaluation: FM vs DM efficiency comparison"
    )
    parser.add_argument("--method", type=str, choices=["fm", "dm", "both"], default="both",
                        help="Which method to evaluate")
    parser.add_argument("--fm_ckpt", type=str, default="",
                        help="FM checkpoint path")
    parser.add_argument("--dm_ckpt", type=str, default="",
                        help="DM checkpoint path")
    parser.add_argument("--fm_euler_steps", type=int, nargs="+",
                        default=[1, 4, 8, 16],
                        help="Euler steps for FM")
    parser.add_argument("--dm_ddim_steps", type=int, nargs="+",
                        default=[10, 25, 50, 100],
                        help="DDIM steps for DM")
    parser.add_argument("--num_eval_samples", type=int, default=100,
                        help="Number of evaluation samples")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output_dir", type=str, default="nfe_sweep_results",
                        help="Output directory")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42],
                        help="Random seeds for evaluation")
    parser.add_argument("--epochs", type=int, default=100,
                        help="Max epochs (for config)")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Batch size (for config)")
    parser.add_argument("--era5_dir", type=str, default="",
                        help="ERA5 data directory")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"[NFE Sweep] Output directory: {args.output_dir}")
    print(f"[NFE Sweep] Device: {args.device}")
    print(f"[NFE Sweep] Num eval samples: {args.num_eval_samples}")

    data_cfg, model_cfg, train_cfg, _ = get_config(
        data_root="/root/autodl-tmp/fyp_final/Ver4/Trajectory/preprocessed_era5"
    )
    data_cfg.preprocessed_dir = "/root/autodl-tmp/fyp_final/Ver4/Trajectory/preprocessed_era5"
    data_cfg.norm_stats_path = "/root/autodl-tmp/fyp_final/Ver4/Trajectory/preprocessed_era5/norm_stats.pt"
    train_cfg.max_epochs = args.epochs
    train_cfg.batch_size = args.batch_size

    set_seed(args.seeds[0])

    fm_results_all = []
    dm_results_all = []

    for seed in args.seeds:
        set_seed(seed)
        seed_dir = os.path.join(args.output_dir, f"seed_{seed}")
        os.makedirs(seed_dir, exist_ok=True)

        if args.method in ("fm", "both"):
            fm_steps = args.fm_euler_steps
            fm_ckpt = args.fm_ckpt or os.path.join(seed_dir, "checkpoints_fm", "best.pt")
            fm_results = run_nfe_sweep(
                method="fm",
                nfe_list=fm_steps,
                data_cfg=data_cfg,
                model_cfg=model_cfg,
                train_cfg=train_cfg,
                device=args.device,
                checkpoint_path=fm_ckpt,
                num_eval_samples=args.num_eval_samples,
                output_dir=seed_dir,
            )
            fm_results_all.append(fm_results)
            print_nfe_table(fm_results, "FM")

        if args.method in ("dm", "both"):
            dm_steps = args.dm_ddim_steps
            dm_ckpt = args.dm_ckpt or os.path.join(seed_dir, "checkpoints_dm", "best.pt")
            dm_results = run_nfe_sweep(
                method="dm",
                nfe_list=dm_steps,
                data_cfg=data_cfg,
                model_cfg=model_cfg,
                train_cfg=train_cfg,
                device=args.device,
                checkpoint_path=dm_ckpt,
                num_eval_samples=args.num_eval_samples,
                output_dir=seed_dir,
            )
            dm_results_all.append(dm_results)
            print_nfe_table(dm_results, "DM")

    if len(args.seeds) > 1 and (fm_results_all or dm_results_all):
        print("\n" + "=" * 80)
        print("MULTI-SEED AGGREGATION")
        print("=" * 80)

        if fm_results_all:
            agg_fm = aggregate_multi_seed(fm_results_all, "FM")
            print_nfe_table(agg_fm, "FM (aggregated)")

        if dm_results_all:
            agg_dm = aggregate_multi_seed(dm_results_all, "DM")
            print_nfe_table(agg_dm, "DM (aggregated)")

    if fm_results_all and dm_results_all:
        agg_fm = aggregate_multi_seed(fm_results_all, "FM")
        agg_dm = aggregate_multi_seed(dm_results_all, "DM")
        print_comparison_table(agg_fm, agg_dm)

        pareto = compute_speedup_and_pareto(agg_fm, agg_dm)
        pareto_path = os.path.join(args.output_dir, "pareto_analysis.json")
        with open(pareto_path, "w", encoding="utf-8") as f:
            json.dump(pareto, f, indent=2, default=str)
        print(f"\n[Pareto] 分析结果已保存: {pareto_path}")

    print("\n[NFE Sweep] 完成!")


def aggregate_multi_seed(results_per_seed: List[List[Dict]], method: str) -> List[Dict]:
    """Aggregate results across multiple seeds."""
    if not results_per_seed:
        return []

    nfe_to_runs = {}
    for seed_results in results_per_seed:
        for r in seed_results:
            nfe = r["nfe"]
            if nfe not in nfe_to_runs:
                nfe_to_runs[nfe] = []
            nfe_to_runs[nfe].append(r)

    aggregated = []
    for nfe, runs in sorted(nfe_to_runs.items()):
        agg = {
            "method": method,
            "nfe": nfe,
            "rmse_mean": np.mean([r["rmse_mean"] for r in runs]),
            "rmse_std": np.sqrt(np.sum([r["rmse_std"] ** 2 for r in runs])) / len(runs),
            "mae_mean": np.mean([r["mae_mean"] for r in runs]),
            "mae_std": np.sqrt(np.sum([r["mae_std"] ** 2 for r in runs])) / len(runs),
            "inference_time_ms_mean": np.mean([r["inference_time_ms_mean"] for r in runs]),
            "inference_time_ms_std": np.sqrt(np.sum([r["inference_time_ms_std"] ** 2 for r in runs])) / len(runs),
            "n_seeds": len(runs),
        }
        aggregated.append(agg)

    return aggregated


if __name__ == "__main__":
    main()
