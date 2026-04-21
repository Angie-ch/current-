# Using compare_fm_dm FM Model for Trajectory Finetuning

## Overview

The original `flow_matching/finetune_train.py` uses the standalone `flow_matching` module's `ERA5FlowMatchingModel`. If you want to use the **compare_fm_dm** FM model (UnifiedModel architecture) instead, use the new files created below.

## New Files

### 1. `unified_model_adapter.py`
Adapter utilities to bridge compare_fm_dm's `UnifiedModel` to the trajectory pipeline.

**Key classes:**
- `UnifiedModelAutoregressiveWrapper` — wraps a trained FM model for autoregressive inference
- `coords_to_spatial_field()` — converts trajectory history to spatial condition fields

### 2. `finetune_train_compare.py`
Complete finetuning script modeled after `Trajectory/finetune_train.py`, but:
- Uses `compare_fm_dm`'s FM model (`UnifiedModel` in FM mode)
- Generates cache via `generate_comparefm_era5_cache()` (Euler ODE, ~50× faster than DDIM)
- Same training logic, freeze strategies, evaluation, bias correction

### 3. `run_comparefm_finetune.sh`
Convenience bash script with all paths pre-configured.

## Quick Start

### Option A: Use the helper script (recommended)

```bash
cd /root/autodl-tmp/fyp_final/Ver4/flow_matching
chmod +x run_comparefm_finetune.sh
./run_comparefm_finetune.sh
```

### Option B: Manual command

```bash
cd /root/autodl-tmp/fyp_final/Ver4/flow_matching

python finetune_train_compare.py \
    --pretrained_ckpt /root/autodl-tmp/fyp_final/Ver4/Trajectory/checkpoints/best.pt \
    --comparefm_code /root/autodl-tmp/fyp_final/Ver4/compare_fm_dm \
    --comparefm_ckpt /root/autodl-tmp/fyp_final/Ver4/compare_fm_dm/multi_seed_results/seed_42/checkpoints_fm/best.pt \
    --norm_stats /root/autodl-tmp/fyp_final/Ver4/Trajectory/preprocessed_era5/norm_stats.pt \
    --data_root /root/autodl-tmp/fyp_final/Typhoon_data_final \
    --track_csv /root/autodl-tmp/fyp_final/Ver4/Trajectory/processed_typhoon_tracks.csv \
    --checkpoint_dir checkpoints_finetune_comparefm \
    --finetune_epochs 80 \
    --finetune_lr 2e-5 \
    --freeze_strategy bridge \
    --euler_steps 4
```

## Key Arguments

| Argument | Description |
|----------|-------------|
| `--pretrained_ckpt` | Trajectory model checkpoint from stage 1 (real ERA5 pre-training) |
| `--comparefm_code` | Path to `compare_fm_dm/` code directory |
| `--comparefm_ckpt` | FM model checkpoint (e.g., `multi_seed_results/seed_42/checkpoints_fm/best.pt`) |
| `--norm_stats` | Normalization stats (`.pt` file with mean/std) |
| `--data_root` | Path to raw ERA5 data (`Typhoon_data_final`) |
| `--track_csv` | Processed typhoon tracks CSV |
| `--checkpoint_dir` | Output directory for finetuned model |
| `--euler_steps` | FM Euler integration steps (default: 4, 1–10 typical) |
| `--freeze_strategy` | Same options as original: `physics_only`, `discriminative`, `bridge`, `all` |

## What It Does

1. **Loads stage 1 pretrained trajectory model** (`LT3PModel` from `Trajectory/checkpoints/best.pt`)
2. **Generates FM-ERA5 cache** using compare_fm_dm's FM model:
   - Autoregressive rollout (24 steps = 72h)
   - Euler ODE solver (4 steps typically, much faster than DDIM)
   - Saves to `{checkpoint_dir}/../comparefm_era5_cache/era5_cache.npz` (reused on subsequent runs)
3. **Finetunes** the trajectory model on FM-ERA5 using the same two-stage logic as `Trajectory/finetune_train.py`:
   - `DiffERA5Dataset` loads real trajectories + synthetic FM-ERA5
   - Wraps model with `ERA5ConvAdaptedModel` adapter (1×1 conv bottleneck)
   - Selective freezing (physics encoder + adapter trainable, rest frozen)
   - EMA, cosine LR, early stopping
4. **Evaluates** on test set:
   - Real ERA5 input
   - FM-ERA5 input
   - Optional bias correction (MOS)

## Output

Finetuned model saved to:
```
{checkpoint_dir}/best_finetune_comparefm.pt
```

Configuration saved to:
```
{checkpoint_dir}/finetune_config.json
```

## Notes

- **Compatibility:** The compare_fm_dm FM model uses a different architecture (`UnifiedModel`) than the standalone flow_matching module. This script bridges that gap by directly loading the compare_fm_dm checkpoint and running its `sample_fm()` method.
- **Cache reuse:** Once generated, the FM-ERA5 cache is saved and reused across runs (delete `comparefm_era5_cache/era5_cache.npz` to regenerate).
- **Speed:** Euler 4-step FM sampling is ~50× faster than DDIM 50-step diffusion. Expect ~15–30 minutes for cache generation (depending on dataset size) vs ~10 hours for diffusion.
- **Memory:** Batch size for cache generation is conservative (`BATCH_SIZE=4`) to avoid OOM with larger compare_fm_dm models. Adjust based on your GPU.

## Troubleshooting

### Import errors
Make sure `--comparefm_code` points to the **root of compare_fm_dm** (the directory containing `configs/`, `models/`, `data/`).

### Cache generation fails (OOM)
Reduce `BATCH_SIZE` in `finetune_train_compare.py` (line ~366) to 2 or 1.

### Missing `condition` key in dataset
The `compare_fm_dm` dataset returns `condition` and `target`. This script assumes that interface. If your version differs, adjust `full_dataset[ds_idx]['condition']`.

### No test storms in cache
The script automatically supplements missing test storms. If a storm has no valid forecast windows, it's skipped (check logs).

## Next Steps

After finetuning, you can:
1. Run evaluation on 2019–2021 as in `Trajectory/eval_2019_2021.py` (modify to point to the new checkpoint)
2. Use the finetuned model in your end-to-end pipeline
3. Compare performance between:
   - Diffusion-ERA5 finetuned (`Trajectory/checkpoints_finetune/best_finetune.pt`)
   - FM-ERA5 finetuned (`flow_matching/checkpoints_finetune_comparefm/best_finetune_comparefm.pt`)
   - No finetuning (stage 1 only)
