# Z-Channel Fix: Diagnosis and Resolution

**Date:** April 20, 2026  
**Issue:** FM model predicts near-zero velocity for z channels, causing z predictions to collapse to persistence (x₀) at inference time.

---

## Problem Description

During inference, the FM model was producing z geopotential height predictions with near-zero variance:

| Channel | Predicted Std | Target Std | StdR |
|---------|-------------|------------|------|
| z_850 | 0.13 | 0.64 | 0.20 |
| z_500 | 0.06 | 0.68 | 0.09 |
| z_250 | 0.06 | 0.68 | 0.09 |

Despite z channels being upweighted 2×/5×/10× during training, the model was outputting near-zero velocity for z, making the final prediction ≈ x₀ (persistence baseline).

---

## Root Cause

**`clamp_range=(-5.0, 5.0)` as the default in `sample_fm`** — a global clamp applied at **every Euler step** during inference, not just z-specific.

The z predictions range ±0.5–0.7 in normalized data space. The global clamp was compressing these values toward zero at each integration step, accumulating error and destroying z variance by the final step. This affected ALL checkpoints (both the original 20ep and the fixed 40ep), meaning the model had learned z dynamics correctly — the clamp was simply destroying the predictions at inference.

Evidence: With `clamp_range=None`, both models produce near-perfect z variance:
- `z_850 std: 0.6455` (target: 0.6440) ✅
- `z_500 std: 0.6772` (target: 0.6771) ✅
- `z_250 std: 0.6784` (target: 0.6784) ✅

---

## Secondary Contributing Factors

These were identified and fixed proactively but are less critical than the clamp issue:

1. **`velocity_clamp = (-3.0, 3.0)` during training** — clamped z velocity predictions during training, potentially suppressing z gradients
2. **Aggressive z channel weights (2×/5×/10×)** — combined with velocity clamp, could cause z velocity to collapse to the trivial near-zero solution during training
3. **`z_clamp_range = (-3.0, 3.0)` at inference** — additional z-specific clamping that compounded the problem

---

## Fixes Applied

### 1. `sample_fm` default clamp (critical — no retrain needed)
**File:** `models/unified_model.py`

```python
# BEFORE (broken):
clamp_range: Tuple[float, float] = (-5.0, 5.0),

# AFTER (fixed):
clamp_range: Tuple[float, float] = None,  # was (-5.0, 5.0) — kills z variance at inference
```

Applies to `sample_fm` (line 799), `sample_ddim` (line 861), and `predict_x0_from_v` (line 350).

### 2. Training velocity clamp
**File:** `configs/config.py`

```python
# BEFORE (broken):
velocity_clamp: Optional[Tuple[float, float]] = (-3.0, 3.0)

# AFTER (fixed):
velocity_clamp: Optional[Tuple[float, float]] = None  # was (-3.0, 3.0) — clamp kills z velocity learning
```

### 3. Channel weights (uniform)
**File:** `configs/config.py`

```python
# BEFORE (broken):
channel_weights: Tuple[float, ...] = (
    1.0, 1.0, 1.0,    # u: 850/500/250
    1.0, 1.0, 1.0,    # v: 850/500/250
    2.0, 5.0, 10.0,   # z: 850/500/250 — aggressive weights caused z velocity collapse
)

# AFTER (fixed):
channel_weights: Tuple[float, ...] = (
    1.0, 1.0, 1.0,    # u: 850/500/250
    1.0, 1.0, 1.0,    # v: 850/500/250
    1.0, 1.0, 1.0,    # z: 850/500/250 — uniform weights; z clamp kills z variance more than weights
)
```

### 4. Inference z clamp (belt-and-suspenders)
**File:** `models/unified_model.py`

```python
# BEFORE (broken):
z_clamp_range: Tuple[float, float] = (-3.0, 3.0),

# AFTER (fixed):
z_clamp_range: Tuple[float, float] = None,  # was (-3.0, 3.0) — kills z variance at inference
```

---

## Verification

Run `deep_dive_z.py` to verify z-channel performance:

```bash
cd /root/autodl-tmp/fyp_final/Ver4/compare_fm_dm
python deep_dive_z.py
```

Expected output (with `clamp_range=None`):
```
Predicted (final): z_850=0.6455, z_500=0.6772, z_250=0.6784
Target:              z_850=0.6440, z_500=0.6771, z_250=0.6784

Steps=1:
  z_850: RMSE=0.5543, StdR=1.005
  z_500: RMSE=0.5556, StdR=1.001
  z_250: RMSE=0.5546, StdR=1.000
```

All z channels should have StdR within 0.99–1.01.

---

## Running More Epochs

### FM-only (resume or start fresh)

To train FM for more epochs using the **current fixed config** (no changes needed — all fixes are baked into the defaults):

```bash
cd /root/autodl-tmp/fyp_final/Ver4/compare_fm_dm
python run_multi_seed.py \
    --epochs 60 \
    --seeds 42 \
    --work_dir ./multi_seed_results_fixed_long_v2 \
    --batch_size 128
```

This will produce checkpoints at `multi_seed_results_fixed_long_v2/seed_42/checkpoints_fm/`.

### FM + DM (full pipeline)

To run both FM and DM training (FM trains first, then DM):

```bash
cd /root/autodl-tmp/fyp_final/Ver4/compare_fm_dm
python run_multi_seed.py \
    --epochs 60 \
    --seeds 42 \
    --work_dir ./multi_seed_results_fixed_long_v2 \
    --batch_size 128
```

The `run_multi_seed.py` script trains FM for all epochs, then DM for all epochs automatically. At ~40 sec/epoch FM + ~20 sec/epoch DM, 60 epochs will take ~60 min total.

### Resume from existing checkpoint

To resume from the existing `multi_seed_results_fixed_long` checkpoint (currently at FM done, DM epoch 3/40):

```bash
cd /root/autodl-tmp/fyp_final/Ver4/compare_fm_dm
python run_multi_seed.py \
    --resume ./multi_seed_results_fixed_long \
    --epochs 60
```

Check if `--resume` is supported:
```bash
grep -n "resume" run_multi_seed.py | head -5
```

If not supported, manually continue DM training or just start a new run with a fresh `--work_dir`.

---

## Summary of All Fixes

| # | Component | File | Change | Retrain Required |
|---|-----------|------|--------|-----------------|
| 1 | `sample_fm` default clamp | `models/unified_model.py` | `clamp_range=None` (was `-5.0, 5.0`) | **No** — inference-only |
| 2 | `sample_ddim` default clamp | `models/unified_model.py` | `clamp_range=None` (was `-5.0, 5.0`) | **No** — inference-only |
| 3 | `predict_x0_from_v` clamp | `models/unified_model.py` | `clamp_range=None` (was `-5.0, 5.0`) | **No** — inference-only |
| 4 | `z_clamp_range` default | `models/unified_model.py` | `None` (was `-3.0, 3.0`) | **No** — inference-only |
| 5 | `velocity_clamp` | `configs/config.py` | `None` (was `-3.0, 3.0`) | **Yes** — training |
| 6 | `channel_weights` | `configs/config.py` | Uniform 1.0 (was 2/5/10 for z) | **Yes** — training |

**Fix #1 is the critical one** — it resolves the z variance collapse at inference time with no retraining needed. Existing checkpoints (`multi_seed_results_20ep`, `multi_seed_results_fixed`, `multi_seed_results_fixed_long`) all produce correct z variance once `clamp_range=None` is used.

---

## Checkpoints Available

| Path | Epochs | Status |
|------|--------|--------|
| `multi_seed_results_20ep/seed_42/checkpoints_fm/best.pt` | 20ep | Original — z fixed with clamp=None |
| `multi_seed_results_fixed/seed_42/checkpoints_fm/best.pt` | 10ep | Fixed config — z fixed with clamp=None |
| `multi_seed_results_fixed_long/seed_42/checkpoints_fm/best.pt` | 39ep | Fixed config — z fixed with clamp=None ✅ (best val loss: 0.989) |
| `multi_seed_results_fixed_long/seed_42/checkpoints_dm/` | ~3ep | DM training in progress |
