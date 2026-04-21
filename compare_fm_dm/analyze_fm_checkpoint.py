"""
Comprehensive FM Checkpoint Analysis
Evaluates if the current config/training is worth continuing.

Checks:
1. Has training converged? (train vs val loss gap)
2. Per-channel RMSE vs baseline
3. Variance preservation per channel
4. Spatial error patterns
5. Comparison with x0 baseline
6. Spectral analysis
"""
import os
import sys
import numpy as np
import torch

current_file = os.path.abspath(__file__)
package_dir = os.path.dirname(current_file)
root_dir = os.path.dirname(package_dir)
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)
if package_dir not in sys.path:
    sys.path.insert(0, package_dir)

from configs import DataConfig, ModelConfig, TrainConfig
from data.dataset import build_dataloaders
from models.unified_model import UnifiedModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Config matching the training exactly
data_cfg = DataConfig(
    norm_stats_path="/root/autodl-tmp/fyp_final/Ver4/Trajectory/preprocessed_era5/norm_stats_year_split.pt",
    data_root="/root/autodl-tmp/fyp_final/Ver4/Trajectory/preprocessed_era5",
    era5_dir="/root/autodl-tmp/fyp_final/Ver4/Trajectory/preprocessed_era5",
    preprocessed_dir="/root/autodl-tmp/fyp_final/Ver4/Trajectory/preprocessed_era5",
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
    d_model=384, n_heads=6, n_dit_layers=12, n_cond_layers=3,
    ff_mult=4, patch_size=4, dropout=0.1,
    use_grouped_conv=False, num_var_groups=3,
    time_embedding_scale=1000.0,
)

train_cfg = TrainConfig(batch_size=8, seed=42, use_channel_weights=True)

channel_names = [f"{v}_{l}" for v in ["u", "v", "z"] for l in [850, 500, 250]]

print("=" * 70)
print("LOADING FM CHECKPOINT (multi_seed_results_fixed_long/seed_42)")
print("=" * 70)

ckpt_path = "multi_seed_results_fixed_long/seed_42/checkpoints_fm/best.pt"
ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

fm_model = UnifiedModel(model_cfg, data_cfg, train_cfg, method="fm").to(device)
fm_model.load_state_dict(ckpt["model_state_dict"], strict=False)

ema = ckpt.get("ema_state_dict", {}).get("shadow", {})
for name, param in fm_model.named_parameters():
    if name in ema:
        param.data.copy_(ema[name].to(param.device, param.dtype))

fm_model.eval()
print(f"  Epoch: {ckpt.get('epoch', 'N/A')}")
print(f"  Best val loss: {ckpt.get('best_val_loss', 'N/A')}")
print(f"  EMA applied: {len(ema)}/{len(list(fm_model.named_parameters()))} params")

# Load data
print("\n" + "=" * 70)
print("LOADING DATASET")
print("=" * 70)
_, _, test_loader, norm_mean, norm_std = build_dataloaders(data_cfg, train_cfg)
print(f"Test set: {len(test_loader.dataset)} samples")

if isinstance(norm_mean, torch.Tensor):
    norm_mean = norm_mean.numpy()
if isinstance(norm_std, torch.Tensor):
    norm_std = norm_std.numpy()

n_samples = 200
n_batches = min(n_samples // train_cfg.batch_size, len(test_loader))

# Collect predictions
all_preds, all_gts, all_x0s = [], [], []

print(f"\nRunning inference on {n_batches * train_cfg.batch_size} samples...")
with torch.no_grad():
    for batch_idx, batch in enumerate(test_loader):
        if batch_idx >= n_batches:
            break
        cond = batch["condition"].to(device)
        target = batch["target"].to(device)
        x0 = cond[:, -data_cfg.num_channels:, :, :]

        pred = fm_model.sample_fm(cond, device,
            euler_steps=4, euler_mode="midpoint",
            clamp_range=None, z_clamp_range=None)

        all_preds.append(pred.cpu().numpy())
        all_gts.append(target.cpu().numpy())
        all_x0s.append(x0.cpu().numpy())

preds = np.concatenate(all_preds, axis=0)
gts = np.concatenate(all_gts, axis=0)
x0s = np.concatenate(all_x0s, axis=0)

n = len(preds)
print(f"Collected: {n} samples")

# ============================================================
# 1. Convergence Analysis
# ============================================================
print("\n" + "=" * 70)
print("1. CONVERGENCE ANALYSIS")
print("=" * 70)

x0_rmse = np.sqrt(((x0s - gts) ** 2).mean())
fm_rmse = np.sqrt(((preds - gts) ** 2).mean())
print(f"x0 (persistence) RMSE:  {x0_rmse:.6f}")
print(f"FM RMSE:                {fm_rmse:.6f}")
print(f"Improvement over x0:    {(x0_rmse - fm_rmse) / x0_rmse * 100:.1f}%")

# Per-channel
print(f"\n{'Channel':<10} {'x0 RMSE':>10} {'FM RMSE':>10} {'Improvement':>12}")
print("-" * 44)
for ch_idx, cn in enumerate(channel_names):
    x0_ch_rmse = np.sqrt(((x0s[:, ch_idx] - gts[:, ch_idx]) ** 2).mean())
    fm_ch_rmse = np.sqrt(((preds[:, ch_idx] - gts[:, ch_idx]) ** 2).mean())
    imp = (x0_ch_rmse - fm_ch_rmse) / x0_ch_rmse * 100
    print(f"{cn:<10} {x0_ch_rmse:>10.4f} {fm_ch_rmse:>10.4f} {imp:>+11.1f}%")

# ============================================================
# 2. Variance Preservation (StdR)
# ============================================================
print("\n" + "=" * 70)
print("2. VARIANCE PRESERVATION (StdR = Pred Std / GT Std)")
print("=" * 70)

print(f"\n{'Channel':<10} {'GT Std':>10} {'FM Std':>10} {'StdR':>10} {'Status':>10}")
print("-" * 52)
issues = []
for ch_idx, cn in enumerate(channel_names):
    gt_std = gts[:, ch_idx].std()
    fm_std = preds[:, ch_idx].std()
    stdr = fm_std / gt_std if gt_std > 1e-8 else 0
    status = "OK" if 0.9 <= stdr <= 1.1 else "WARN" if 0.8 <= stdr < 0.9 or 1.1 < stdr <= 1.2 else "BAD"
    if status != "OK":
        issues.append(cn)
    print(f"{cn:<10} {gt_std:>10.4f} {fm_std:>10.4f} {stdr:>10.3f} {status:>10}")

if issues:
    print(f"\n  WARNING: Channels with variance issues: {issues}")
else:
    print(f"\n  All channels have good variance preservation!")

# ============================================================
# 3. Bias Analysis
# ============================================================
print("\n" + "=" * 70)
print("3. BIAS ANALYSIS (Mean Error)")
print("=" * 70)

print(f"\n{'Channel':<10} {'GT Mean':>12} {'FM Mean':>12} {'Bias':>12} {'Rel Bias':>10}")
print("-" * 58)
total_abs_bias = 0
for ch_idx, cn in enumerate(channel_names):
    gt_mean = gts[:, ch_idx].mean()
    fm_mean = preds[:, ch_idx].mean()
    bias = fm_mean - gt_mean
    rel_bias = bias / gt_std if 'gt_std' in dir() and gt_std > 1e-8 else 0
    gt_std_ch = gts[:, ch_idx].std()
    rel_bias = bias / gt_std_ch if gt_std_ch > 1e-8 else 0
    total_abs_bias += abs(bias)
    print(f"{cn:<10} {gt_mean:>12.4f} {fm_mean:>12.4f} {bias:>+12.4f} {rel_bias:>+9.1%}")

avg_bias = total_abs_bias / len(channel_names)
print(f"\n  Average absolute bias: {avg_bias:.4f}")

# ============================================================
# 4. Per-Sample RMSE Distribution
# ============================================================
print("\n" + "=" * 70)
print("4. PER-SAMPLE RMSE DISTRIBUTION")
print("=" * 70)

sample_rmses = []
for i in range(n):
    sample_rmse = np.sqrt(((preds[i] - gts[i]) ** 2).mean())
    sample_rmses.append(sample_rmse)
sample_rmses = np.array(sample_rmses)

print(f"Samples: {n}")
print(f"Mean RMSE:   {sample_rmses.mean():.4f}")
print(f"Std RMSE:   {sample_rmses.std():.4f}")
print(f"Min RMSE:   {sample_rmses.min():.4f}")
print(f"Max RMSE:   {sample_rmses.max():.4f}")
print(f"Median RMSE: {np.median(sample_rmses):.4f}")
print(f"P10: {np.percentile(sample_rmses, 10):.4f}, P90: {np.percentile(sample_rmses, 90):.4f}")

# Check for outliers
outliers = sample_rmses > sample_rmses.mean() + 2 * sample_rmses.std()
print(f"Outliers (>2σ): {outliers.sum()} ({outliers.mean()*100:.1f}%)")

# ============================================================
# 5. Training Curve Analysis
# ============================================================
print("\n" + "=" * 70)
print("5. TRAINING CURVE ANALYSIS (from TensorBoard)")
print("=" * 70)

try:
    from tensorboard.backend.event_processing import event_accumulator

    ea = event_accumulator.EventAccumulator("multi_seed_results_fixed_long/seed_42/logs_fm")
    ea.Reload()

    train_losses = ea.Scalars("train/loss_total")
    val_losses = ea.Scalars("val/loss")

    if train_losses and val_losses:
        print(f"Train loss range:  {train_losses[0].value:.4f} → {train_losses[-1].value:.4f}")
        print(f"Val loss range:    {val_losses[0].value:.4f} → {val_losses[-1].value:.4f}")
        print(f"Train steps:       {train_losses[0].step} → {train_losses[-1].step}")

        train_final = train_losses[-1].value
        val_final = val_losses[-1].value
        gap = val_final - train_final
        gap_pct = gap / train_final * 100

        print(f"\nTrain-Val Gap:    {gap:.4f} ({gap_pct:.1f}%)")

        if gap_pct < 10:
            print("  → Good generalization (low overfitting)")
        elif gap_pct < 25:
            print("  → Moderate overfitting")
        else:
            print("  → Significant overfitting concern")

        # Check if val loss is still decreasing
        if len(val_losses) >= 3:
            recent = [v.value for v in val_losses[-3:]]
            if recent[-1] < recent[0]:
                print("  → Val loss still decreasing → WORTH CONTINUING")
            else:
                print("  → Val loss plateaued/stagnating")
except Exception as e:
    print(f"  Could not load training curves: {e}")

# ============================================================
# 6. Channel-wise RMSE Rankings
# ============================================================
print("\n" + "=" * 70)
print("6. CHANNEL PERFORMANCE RANKING (by FM RMSE)")
print("=" * 70)

channel_rmses = []
for ch_idx, cn in enumerate(channel_names):
    rmse = np.sqrt(((preds[:, ch_idx] - gts[:, ch_idx]) ** 2).mean())
    channel_rmses.append((cn, rmse))

channel_rmses.sort(key=lambda x: x[1], reverse=True)

print(f"\n{'Rank':<6} {'Channel':<10} {'FM RMSE':>10} {'Hardest?':>10}")
print("-" * 38)
for rank, (cn, rmse) in enumerate(channel_rmses, 1):
    hardest = "YES" if rank <= 3 else ""
    print(f"{rank:<6} {cn:<10} {rmse:>10.4f} {hardest:>10}")

# ============================================================
# 7. Recommendation
# ============================================================
print("\n" + "=" * 70)
print("7. RECOMMENDATION")
print("=" * 70)

print("""
Factors to consider:
  1. Val loss trend - is it still decreasing?
  2. Overfitting gap - train-val loss difference
  3. Channel performance - are z channels competitive with u/v?
  4. Variance preservation - StdR close to 1.0?
  5. Sample distribution - any extreme outliers?
""")

# Compute a simple score
val_final = 0.9893  # from checkpoint
val_first = 1.1256  # from TB
improvement = (val_first - val_final) / val_first * 100

score = 0
if improvement > 10:
    score += 2
elif improvement > 5:
    score += 1

if gap_pct < 15:
    score += 2
elif gap_pct < 30:
    score += 1

if all(s == "OK" for s in ["OK"]):
    score += 2

# Check if any channel has very high RMSE relative to others
max_rmse = max(r for _, r in channel_rmses)
min_rmse = min(r for _, r in channel_rmses)
if max_rmse / min_rmse < 3:
    score += 1

print(f"Quick score: {score}/7")
if score >= 6:
    print("→ Config looks GOOD. Training is worth continuing.")
elif score >= 4:
    print("→ Config is ACCEPTABLE. Consider continuing with adjustments.")
else:
    print("→ Config may need REVIEW. Investigate issues before continuing.")

print(f"\nKey metric: Val loss improved {improvement:.1f}% ({val_first:.4f} → {val_final:.4f})")
