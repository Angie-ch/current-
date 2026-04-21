"""
Multi-Seed Evaluation Runner — Publication-Ready FM vs DM Comparison

Trains FM and DM with multiple random seeds and reports mean ± std
across seeds for all metrics. This is essential for publication because
it quantifies the variance due to random initialization.

Usage:
    python run_multi_seed.py --seeds 42 43 44 --epochs 100
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
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import DataLoader

script_dir = os.path.dirname(os.path.abspath(__file__))
pkg_root = os.path.dirname(script_dir)
if pkg_root not in sys.path:
    sys.path.insert(0, pkg_root)

from compare_fm_dm.configs import get_config, get_comparison_config
from compare_fm_dm.data.dataset import build_dataloaders
from compare_fm_dm.models.unified_model import create_model
from compare_fm_dm.models.trainer import UnifiedTrainer, EMA
from compare_fm_dm.evaluation.publication_pipeline import PublicationPipeline
from compare_fm_dm.evaluation import (
    quick_run_full_pipeline,
    compute_climatology,
    BaselineForecaster,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def set_seed(seed: int):
    """Set all random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    import random
    random.seed(seed)


def train_and_evaluate_single_seed(
    seed: int,
    data_cfg,
    model_cfg,
    train_cfg,
    infer_cfg,
    work_dir: str,
    device: str = "cuda",
    skip_train: bool = False,
) -> Dict:
    """
    Train and evaluate FM + DM for a single seed.

    Returns:
        Dict with all metrics for this seed
    """
    set_seed(seed)
    logger.info(f"{'='*70}")
    logger.info(f"SEED {seed}")
    logger.info(f"{'='*70}")

    seed_dir = os.path.join(work_dir, f"seed_{seed}")
    os.makedirs(seed_dir, exist_ok=True)

    device = torch.device(device if torch.cuda.is_available() else "cpu")

    # Build dataloaders
    train_loader, val_loader, test_loader, norm_mean, norm_std = build_dataloaders(
        data_cfg, train_cfg
    )

    # FM model
    fm_ckpt = os.path.join(seed_dir, "checkpoints_fm", "best.pt")
    if skip_train and os.path.exists(fm_ckpt):
        fm_model = create_model(model_cfg, data_cfg, train_cfg, method="fm")
        fm_model = fm_model.to(device)
        ckpt = torch.load(fm_ckpt, map_location=device, weights_only=False)
        fm_model.load_state_dict(ckpt["model_state_dict"], strict=False)
        logger.info(f"Loaded FM checkpoint: {fm_ckpt}")
    else:
        fm_model = create_model(model_cfg, data_cfg, train_cfg, method="fm")
        fm_model = fm_model.to(device)

        trainer_fm = UnifiedTrainer(
            fm_model, train_loader, val_loader, train_cfg, data_cfg,
            work_dir=seed_dir, method="fm",
        )
        trainer_fm.train()

        # Load best
        if os.path.exists(fm_ckpt):
            ckpt = torch.load(fm_ckpt, map_location=device, weights_only=False)
            fm_model.load_state_dict(ckpt["model_state_dict"], strict=False)
            if "ema_state_dict" in ckpt:
                ema = EMA(fm_model, decay=0.999)
                ema.load_state_dict(ckpt["ema_state_dict"])
                ema.apply_shadow(fm_model)
                logger.info(f"Loaded EMA FM for seed {seed}")

    fm_model.eval()

    # DM model
    dm_ckpt = os.path.join(seed_dir, "checkpoints_dm", "best.pt")
    if skip_train and os.path.exists(dm_ckpt):
        dm_model = create_model(model_cfg, data_cfg, train_cfg, method="dm")
        dm_model = dm_model.to(device)
        ckpt = torch.load(dm_ckpt, map_location=device, weights_only=False)
        dm_model.load_state_dict(ckpt["model_state_dict"], strict=False)
        logger.info(f"Loaded DM checkpoint: {dm_ckpt}")
    else:
        dm_model = create_model(model_cfg, data_cfg, train_cfg, method="dm")
        dm_model = dm_model.to(device)

        trainer_dm = UnifiedTrainer(
            dm_model, train_loader, val_loader, train_cfg, data_cfg,
            work_dir=seed_dir, method="dm",
        )
        trainer_dm.train()

        if os.path.exists(dm_ckpt):
            ckpt = torch.load(dm_ckpt, map_location=device, weights_only=False)
            dm_model.load_state_dict(ckpt["model_state_dict"], strict=False)
            if "ema_state_dict" in ckpt:
                ema = EMA(dm_model, decay=0.999)
                ema.load_state_dict(ckpt["ema_state_dict"])
                ema.apply_shadow(dm_model)
                logger.info(f"Loaded EMA DM for seed {seed}")

    dm_model.eval()

    # Compute climatology
    climatology_path = os.path.join(seed_dir, "climatology.npy")
    if os.path.exists(climatology_path):
        climatology = np.load(climatology_path)
    else:
        logger.info("Computing climatology...")
        from evaluation.compute_climatology import quick_compute_climatology
        typhoon_ids = list(set([
            s[0] for s in test_loader.dataset.samples
        ]))[:500]
        climatology = quick_compute_climatology(
            data_cfg.era5_dir,
            climatology_path,
            n_typhoons=500,
            n_samples_per_typhoon=30,
            grid_size=data_cfg.grid_size,
            num_channels=data_cfg.num_channels,
            var_names=data_cfg.var_names,
        )

    # Collect predictions for evaluation
    logger.info("Collecting predictions...")
    fm_preds, dm_preds, gts, conditions = collect_predictions(
        fm_model, dm_model, test_loader.dataset,
        num_samples=min(50, len(test_loader.dataset)),
        seed=seed,
        infer_cfg=infer_cfg,
        device=device,
    )

    # Run evaluation pipeline
    logger.info("Running evaluation pipeline...")
    results = quick_run_full_pipeline(
        fm_predictions=fm_preds,
        dm_predictions=dm_preds,
        ground_truth=gts,
        data_cfg=data_cfg,
        output_dir=os.path.join(seed_dir, "evaluation"),
        climatology=climatology,
    )

    # Save seed results
    import json
    def make_serializable(obj):
        if isinstance(obj, (np.floating, np.integer)):
            return float(obj) if isinstance(obj, np.floating) else int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [make_serializable(v) for v in obj]
        return obj

    results_path = os.path.join(seed_dir, "results_seed.json")
    with open(results_path, 'w') as f:
        json.dump(make_serializable(results), f, indent=2)

    logger.info(f"Seed {seed} complete. Results: {results_path}")

    return results


def collect_predictions(
    fm_model,
    dm_model,
    dataset,
    num_samples: int = 50,
    seed: int = 42,
    infer_cfg=None,
    device=None,
):
    """Collect predictions from both models on the test set."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    fm_preds = []
    dm_preds = []
    gts = []
    conditions = []

    rng = np.random.RandomState(seed)
    indices = rng.choice(len(dataset), min(num_samples, len(dataset)), replace=False)

    ddim_steps = infer_cfg.ddim_steps if infer_cfg is not None else 50

    for idx in indices:
        sample = dataset[idx]
        cond = sample["condition"].unsqueeze(0).to(device)
        target = sample["target"].unsqueeze(0).to(device)

        with torch.no_grad():
            fm_pred = fm_model.sample_fm(cond, device, euler_steps=4, euler_mode="midpoint", z_clamp_range=None)
            dm_pred = dm_model.sample_dm(cond, device, ddim_steps=ddim_steps)

        fm_preds.append(fm_pred.cpu())
        dm_preds.append(dm_pred.cpu())
        gts.append(target.cpu())
        conditions.append(cond.cpu())

    return fm_preds, dm_preds, gts, conditions


def aggregate_results(all_seed_results: List[Dict]) -> Dict:
    """
    Aggregate results across seeds: compute mean ± std.

    Args:
        all_seed_results: List of result dicts, one per seed

    Returns:
        Dict with mean ± std for each metric
    """
    aggregated = {
        "seeds": [r.get("seed", i) for i, r in enumerate(all_seed_results)],
        "n_seeds": len(all_seed_results),
    }

    def extract_nested(data, path):
        """Extract a nested value from dict using path like 'FM/rmse_mean'."""
        parts = path.split("/")
        obj = data
        for p in parts:
            if isinstance(obj, dict) and p in obj:
                obj = obj[p]
            else:
                return None
        return obj

    # Keys to aggregate
    metrics_to_aggregate = [
        ("FM/rmse_mean", "fm_rmse_mean"),
        ("FM/rmse_mean_std", "fm_rmse_std"),
        ("DM/rmse_mean", "dm_rmse_mean"),
        ("DM/rmse_mean_std", "dm_rmse_std"),
        ("FM/lat_weighted_rmse_mean", "fm_lat_rmse_mean"),
        ("DM/lat_weighted_rmse_mean", "dm_lat_rmse_mean"),
        ("FM/acc_mean", "fm_acc_mean"),
        ("DM/acc_mean", "dm_acc_mean"),
        ("FM_spectral/spectral_slope_pred", "fm_spectral_slope"),
        ("DM_spectral/spectral_slope_pred", "dm_spectral_slope"),
        ("FM_spectral/spectral_slope_gt", "gt_spectral_slope"),
        ("FM_physics/divergence_rmse_pred", "fm_div_rmse"),
        ("DM_physics/divergence_rmse_pred", "dm_div_rmse"),
        ("FM_path/straightness_mean", "fm_straightness"),
        ("DM_path/straightness_mean", "dm_straightness"),
    ]

    for result in all_seed_results:
        result["seed"] = result.get("seed", all_seed_results.index(result))

    # Build per-metric arrays
    metric_arrays = defaultdict(list)
    for result in all_seed_results:
        for path, key in metrics_to_aggregate:
            val = extract_nested(result, path)
            if val is not None:
                metric_arrays[key].append(val)

    # Compute mean and std
    for key, values in metric_arrays.items():
        arr = np.array(values)
        aggregated[f"{key}_mean"] = float(arr.mean())
        aggregated[f"{key}_std"] = float(arr.std())
        aggregated[f"{key}_values"] = values

    return aggregated


def print_aggregated_summary(agg: Dict):
    """Print a publication-quality summary table."""
    print("\n" + "=" * 80)
    print("AGGREGATED RESULTS ACROSS SEEDS (mean ± std)")
    print("=" * 80)

    print(f"\n{'Metric':<35} {'FM':<20} {'DM':<20}")
    print("-" * 75)

    def fmt(mean, std):
        return f"{mean:.4f} ± {std:.4f}" if std > 0 else f"{mean:.4f}"

    # RMSE
    fm_rmse = fmt(agg.get("fm_rmse_mean_mean", 0), agg.get("fm_rmse_mean_std", 0))
    dm_rmse = fmt(agg.get("dm_rmse_mean_mean", 0), agg.get("dm_rmse_mean_std", 0))
    print(f"{'RMSE (mean)':<35} {fm_rmse:<20} {dm_rmse:<20}")

    # Lat-weighted RMSE
    fm_lat = fmt(agg.get("fm_lat_rmse_mean_mean", 0), agg.get("fm_lat_rmse_mean_std", 0))
    dm_lat = fmt(agg.get("dm_lat_rmse_mean_mean", 0), agg.get("dm_lat_rmse_mean_std", 0))
    print(f"{'Latitude-Weighted RMSE':<35} {fm_lat:<20} {dm_lat:<20}")

    # ACC
    fm_acc = fmt(agg.get("fm_acc_mean_mean", 0), agg.get("fm_acc_mean_std", 0))
    dm_acc = fmt(agg.get("dm_acc_mean_mean", 0), agg.get("dm_acc_mean_std", 0))
    print(f"{'ACC (mean)':<35} {fm_acc:<20} {dm_acc:<20}")

    # Spectral slope
    fm_slope = fmt(agg.get("fm_spectral_slope_mean", 0), agg.get("fm_spectral_slope_std", 0))
    dm_slope = fmt(agg.get("dm_spectral_slope_mean", 0), agg.get("dm_spectral_slope_std", 0))
    gt_slope = agg.get("gt_spectral_slope_mean", 0)
    print(f"{'Spectral Slope':<35} {fm_slope:<20} {dm_slope:<20}")
    print(f"{'  (GT reference: {:.3f})':<35} {'':20} {'':20}".format(gt_slope))

    # Physics
    fm_div = fmt(agg.get("fm_div_rmse_mean", 0), agg.get("fm_div_rmse_std", 0))
    dm_div = fmt(agg.get("dm_div_rmse_mean", 0), agg.get("dm_div_rmse_std", 0))
    print(f"{'Divergence RMSE':<35} {fm_div:<20} {dm_div:<20}")

    # Path straightness
    fm_path = fmt(agg.get("fm_straightness_mean", 0), agg.get("fm_straightness_std", 0))
    dm_path = fmt(agg.get("dm_straightness_mean", 0), agg.get("dm_straightness_std", 0))
    print(f"{'Path Straightness':<35} {fm_path:<20} {dm_path:<20}")

    print(f"\nSeeds evaluated: {agg.get('n_seeds', 0)}")
    print(f"Seed values: {agg.get('seeds', [])}")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Multi-seed FM vs DM comparison")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=4e-4)
    parser.add_argument("--work_dir", type=str, default="./multi_seed_results_v2")
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num_eval_samples", type=int, default=50)

    args = parser.parse_args()

    torch.set_float32_matmul_precision("high")

    # Get configs with year-based preprocessed_dir
    PREPROCESSED_DIR = "/root/autodl-tmp/fyp_final/Ver4/Trajectory/preprocessed_era5"
    data_cfg, model_cfg, train_cfg, infer_cfg = get_config(
        preprocess_dir=PREPROCESSED_DIR,
    )
    train_cfg.max_epochs = args.epochs
    train_cfg.batch_size = args.batch_size
    train_cfg.learning_rate = args.lr

    os.makedirs(args.work_dir, exist_ok=True)

    logger.info("=" * 70)
    logger.info("Year-based data split (no data leakage):")
    logger.info(f"  Train:  {data_cfg.train_years[0]}-{data_cfg.train_years[1]} (years)")
    logger.info(f"  Val:    {data_cfg.val_years[0]}-{data_cfg.val_years[1]} (years)")
    logger.info(f"  Test:   {data_cfg.test_years[0]}-{data_cfg.test_years[1]} (years)")
    logger.info(f"  Norm:   (var - mean) / std, computed over TRAINING SET ONLY")
    logger.info(f"  Preprocessed dir: {PREPROCESSED_DIR}")
    logger.info("=" * 70)
    logger.info(f"Multi-seed evaluation with seeds: {args.seeds}")
    logger.info(f"Output directory: {args.work_dir}")
    logger.info(f"Epochs: {args.epochs}, Batch size: {args.batch_size}, LR: {args.lr}")
    logger.info(f"Model architecture: use_grouped_conv={model_cfg.use_grouped_conv}")

    all_results = []

    for seed in args.seeds:
        try:
            result = train_and_evaluate_single_seed(
                seed=seed,
                data_cfg=data_cfg,
                model_cfg=model_cfg,
                train_cfg=train_cfg,
                infer_cfg=infer_cfg,
                work_dir=args.work_dir,
                device=args.device,
                skip_train=args.skip_train,
            )
            result["seed"] = seed
            all_results.append(result)
        except Exception as e:
            logger.error(f"Seed {seed} failed: {e}")
            import traceback
            traceback.print_exc()
            continue

    if not all_results:
        logger.error("No seeds completed successfully!")
        return

    # Aggregate
    logger.info("Aggregating results across seeds...")
    aggregated = aggregate_results(all_results)
    print_aggregated_summary(aggregated)

    # Save aggregated results
    agg_path = os.path.join(args.work_dir, "aggregated_results.json")
    import json

    def make_serializable(obj):
        if isinstance(obj, (np.floating, np.integer)):
            return float(obj) if isinstance(obj, np.floating) else int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [make_serializable(v) for v in obj]
        return obj

    with open(agg_path, 'w') as f:
        json.dump(make_serializable(aggregated), f, indent=2)

    logger.info(f"Aggregated results saved: {agg_path}")
    logger.info("Multi-seed evaluation complete!")


if __name__ == "__main__":
    main()
