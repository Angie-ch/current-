"""
Z-Channel Fix Sweep — compares 4 approaches for fixing z_500/z_250 variance collapse.

Variants:
  A: baseline          — use_z_predictor=False, z_channel_weight_override=1.0
  B: z-weight          — use_z_predictor=False, z_channel_weight_override=5.0
  C: z-predictor       — use_z_predictor=True,  z_predictor_weight=1.0
  D: z-predictor+weight — use_z_predictor=True,  z_predictor_weight=2.0, z_channel_weight_override=3.0

Usage:
  # Quick test (20 epochs per variant, 1 seed)
  python run_z_sweep.py --epochs 20 --seeds 42 --quick

  # Full run (100 epochs per variant, 1 seed)
  python run_z_sweep.py --epochs 100 --seeds 42 --output_dir ./z_sweep_results

  # Full run with 3 seeds
  python run_z_sweep.py --epochs 100 --seeds 42 43 44 --output_dir ./z_sweep_results
"""
import os
import sys
import json
import copy
import time
import logging
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from collections import defaultdict

script_dir = os.path.dirname(os.path.abspath(__file__))
pkg_root = os.path.dirname(script_dir)
if pkg_root not in sys.path:
    sys.path.insert(0, pkg_root)

from compare_fm_dm.configs.config import (
    DataConfig, ModelConfig, TrainConfig, InferenceConfig, get_config
)
from compare_fm_dm.data.dataset import build_dataloaders
from compare_fm_dm.models.unified_model import create_model
from compare_fm_dm.models.trainer import UnifiedTrainer, EMA

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


# ── Variant definitions ────────────────────────────────────────────────────────

VARIANTS = {
    "A_baseline": dict(
        use_z_predictor=False,
        z_predictor_weight=0.0,
        z_channel_weight_override=1.0,
    ),
    "B_z_weight": dict(
        use_z_predictor=False,
        z_predictor_weight=0.0,
        z_channel_weight_override=5.0,
    ),
    "C_z_predictor": dict(
        use_z_predictor=True,
        z_predictor_weight=1.0,
        z_channel_weight_override=1.0,
    ),
    "D_z_pred+weight": dict(
        use_z_predictor=True,
        z_predictor_weight=2.0,
        z_channel_weight_override=3.0,
    ),
}


# ── Per-channel prediction verification ───────────────────────────────────────

def verify_predictions(model, test_loader, device, n_batches=5):
    """Compute per-channel RMSE, StdR, and correlation on test set."""
    model.eval()
    channel_names = [f"{v}_{l}" for v in ["u", "v", "z"] for l in [850, 500, 250]]

    preds_batched = [[] for _ in range(9)]
    gts_batched   = [[] for _ in range(9)]

    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            if batch_idx >= n_batches:
                break
            cond   = batch["condition"].to(device)
            target = batch["target"]
            method = getattr(model, 'method', None) or ('fm' if hasattr(model, 'sample_fm') else 'dm')

            if method == 'fm' or not hasattr(model, 'sample_dm'):
                pred = model.sample_fm(cond, device,
                    euler_steps=4, euler_mode="midpoint",
                    clamp_range=None, z_clamp_range=None)
            else:
                pred = model.sample_dm(cond, device,
                    ddim_steps=50, clamp_range=None, z_clamp_range=None)

            for ch in range(9):
                preds_batched[ch].append(pred[:, ch].cpu().numpy())
                gts_batched[ch].append(target[:, ch].numpy())

    results = {}
    for ch, name in enumerate(channel_names):
        p_all = np.concatenate(preds_batched[ch], axis=0)
        t_all = np.concatenate(gts_batched[ch], axis=0)

        rmse  = float(np.sqrt(((p_all - t_all) ** 2).mean()))
        std_r = float(p_all.std() / t_all.std()) if t_all.std() > 1e-6 else 0.0
        corr  = float(np.corrcoef(p_all.flatten(), t_all.flatten())[0, 1])
        bias  = float(p_all.mean() - t_all.mean())
        results[name] = {"rmse": rmse, "std_r": std_r, "corr": corr, "bias": bias}

    return results


# ── Train single variant, single seed ────────────────────────────────────────

def train_variant(
    variant_name: str,
    variant_cfg: dict,
    seed: int,
    base_data_cfg,
    base_model_cfg,
    base_train_cfg,
    base_infer_cfg,
    work_dir: str,
    epochs: int,
    device,
    skip_existing: bool = True,
):
    """Train one FM variant for one seed. Returns eval metrics."""
    set_seed(seed)

    variant_dir = os.path.join(work_dir, f"{variant_name}_seed{seed}")
    os.makedirs(variant_dir, exist_ok=True)

    ckpt_path = os.path.join(variant_dir, "checkpoints_fm", "best.pt")
    if skip_existing and os.path.exists(ckpt_path):
        logger.info(f"[{variant_name}] Checkpoint exists, skipping training (seed={seed})")
        # Still need to evaluate
    else:
        # Build modified train config for this variant
        train_cfg = copy.deepcopy(base_train_cfg)
        train_cfg.max_epochs = epochs
        for k, v in variant_cfg.items():
            setattr(train_cfg, k, v)

        # Z-weight override: modify channel_weights on z channels only
        if variant_cfg.get("z_channel_weight_override", 1.0) != 1.0:
            override = variant_cfg["z_channel_weight_override"]
            cw = list(train_cfg.channel_weights)  # (9,)
            for i in range(6, 9):                 # z channels
                cw[i] = override
            train_cfg.channel_weights = tuple(cw)
            logger.info(f"[{variant_name}] channel_weights after override: {train_cfg.channel_weights}")

        # Build dataloaders
        train_loader, val_loader, test_loader, norm_mean, norm_std = build_dataloaders(
            base_data_cfg, train_cfg
        )

        # Create FM model (ZPredictor is instantiated inside create_model via UnifiedModel.__init__)
        model = create_model(base_model_cfg, base_data_cfg, train_cfg, method="fm")
        model = model.to(device)

        trainer = UnifiedTrainer(
            model, train_loader, val_loader, train_cfg, base_data_cfg,
            work_dir=variant_dir, method="fm",
        )
        trainer.train()

        # Load best EMA
        if os.path.exists(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            model.load_state_dict(ckpt["model_state_dict"], strict=False)
            if "ema_state_dict" in ckpt:
                ema = EMA(model, decay=0.999)
                ema.load_state_dict(ckpt["ema_state_dict"])
                ema.apply_shadow(model)
            logger.info(f"[{variant_name}] Loaded best checkpoint for seed {seed}")

    # Evaluate
    logger.info(f"[{variant_name}] Verifying predictions (seed={seed})...")
    model.eval()

    train_cfg = copy.deepcopy(base_train_cfg)
    for k, v in variant_cfg.items():
        setattr(train_cfg, k, v)
    if variant_cfg.get("z_channel_weight_override", 1.0) != 1.0:
        override = variant_cfg["z_channel_weight_override"]
        cw = list(train_cfg.channel_weights)
        for i in range(6, 9):
            cw[i] = override
        train_cfg.channel_weights = tuple(cw)

    _, _, test_loader, _, _ = build_dataloaders(base_data_cfg, train_cfg)

    metrics = verify_predictions(model, test_loader, device, n_batches=10)

    # Save
    metrics_path = os.path.join(variant_dir, f"metrics_seed{seed}.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"[{variant_name}] Metrics saved to {metrics_path}")

    return metrics


# ── Main sweep ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Z-channel fix sweep")
    parser.add_argument("--epochs", type=int, default=100, help="Epochs per variant")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    parser.add_argument("--output_dir", type=str, default="./z_sweep_results")
    parser.add_argument("--quick", action="store_true", help="20 epochs, 5 eval batches")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--skip_existing", action="store_true", default=True)
    args = parser.parse_args()

    if args.quick:
        args.epochs = 20
        eval_batches = 5
        logger.info("Quick mode: 20 epochs, 5 eval batches")
    else:
        eval_batches = 10

    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Base configs
    data_cfg, model_cfg, train_cfg, infer_cfg = get_config()
    train_cfg.batch_size = args.batch_size

    # Global results
    all_results = {}  # variant_name -> {seed -> metrics}

    start_time = time.time()

    for variant_name, variant_cfg in VARIANTS.items():
        logger.info(f"\n{'='*70}")
        logger.info(f"VARIANT: {variant_name}")
        logger.info(f"Config: {variant_cfg}")
        logger.info(f"{'='*70}")

        all_results[variant_name] = {}

        for seed in args.seeds:
            t0 = time.time()
            metrics = train_variant(
                variant_name=variant_name,
                variant_cfg=variant_cfg,
                seed=seed,
                base_data_cfg=data_cfg,
                base_model_cfg=model_cfg,
                base_train_cfg=train_cfg,
                base_infer_cfg=infer_cfg,
                work_dir=os.path.join(args.output_dir, variant_name),
                epochs=args.epochs,
                device=device,
                skip_existing=args.skip_existing,
            )
            elapsed = time.time() - t0
            all_results[variant_name][seed] = metrics

            # Print per-channel summary
            logger.info(f"[{variant_name}] Seed {seed} — per-channel results:")
            header = f"{'Channel':>8}  {'RMSE':>8}  {'StdR':>8}  {'Corr':>8}  {'Bias':>8}"
            logger.info(header)
            logger.info("-" * len(header))
            for ch_name, m in metrics.items():
                flag = "✅" if m["corr"] > 0.1 else "⚠️"
                logger.info(f"{flag} {ch_name:>7}  {m['rmse']:>8.4f}  {m['std_r']:>8.3f}  {m['corr']:>8.4f}  {m['bias']:>8.4f}  ({elapsed:.0f}s)")

    # ── Aggregate & print summary ───────────────────────────────────────────────
    elapsed_total = time.time() - start_time

    logger.info(f"\n{'='*80}")
    logger.info("SWEEP SUMMARY")
    logger.info(f"Variants: {list(VARIANTS.keys())}")
    logger.info(f"Epochs: {args.epochs}, Seeds: {args.seeds}")
    logger.info(f"Total time: {elapsed_total/60:.1f} min")
    logger.info(f"{'='*80}")

    channel_names = [f"{v}_{l}" for v in ["u", "v", "z"] for l in [850, 500, 250]]

    # Per-channel summary table
    logger.info(f"\n{'Channel':>8}", end="")
    for vn in VARIANTS:
        logger.info(f"  {vn:>22}", end="")
    logger.info("")
    logger.info(f"{'Metric':>8}", end="")
    for _ in VARIANTS:
        logger.info(f"  {'RMSE':>10}  {'StdR':>8}  {'Corr':>8}", end="")
    logger.info("")

    for ch_name in channel_names:
        logger.info(f"{ch_name:>8}", end="")
        for variant_name in VARIANTS:
            seeds = all_results[variant_name]
            if not seeds:
                logger.info(f"  {'N/A':>22}", end="")
                continue
            rmses = [seeds[s][ch_name]["rmse"] for s in seeds]
            stdrs = [seeds[s][ch_name]["std_r"] for s in seeds]
            corrs = [seeds[s][ch_name]["corr"] for s in seeds]
            rmse_mean = np.mean(rmses)
            stdr_mean = np.mean(stdrs)
            corr_mean = np.mean(corrs)
            logger.info(f"  {rmse_mean:>10.4f}  {stdr_mean:>8.3f}  {corr_mean:>8.4f}", end="")
        logger.info("")

    # Best variant per channel
    logger.info(f"\n{'Best variant per channel (by Corr):'}")
    for ch_name in channel_names:
        best_variant = None
        best_corr = -999
        for variant_name in VARIANTS:
            seeds = all_results[variant_name]
            if seeds:
                mean_corr = np.mean([seeds[s][ch_name]["corr"] for s in seeds])
                if mean_corr > best_corr:
                    best_corr = mean_corr
                    best_variant = variant_name
        logger.info(f"  {ch_name}: {best_variant} (corr={best_corr:.4f})")

    # Save results
    summary_path = os.path.join(args.output_dir, "sweep_summary.json")
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"\nFull results saved to {summary_path}")


if __name__ == "__main__":
    main()
