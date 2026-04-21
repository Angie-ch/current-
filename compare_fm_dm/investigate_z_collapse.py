"""
Deep Z-Channel Collapse Investigation

Investigates why the FM model has nearly zero variance on z channels
while maintaining good performance on u/v channels.
"""
import os
import sys
import numpy as np
import torch
import torch.nn.functional as F

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

data_cfg = DataConfig(
    norm_stats_path="/root/autodl-tmp/fyp_final/Ver4/Trajectory/preprocessed_era5/norm_stats_year_split.pt",
    data_root="/root/autodl-tmp/fyp_final/Ver4/Trajectory/preprocessed_era5",
    era5_dir="/root/autodl-tmp/fyp_final/Ver4/Trajectory/preprocessed_era5",
    preprocessed_dir="/root/autodl-tmp/fyp_final/Ver4/Trajectory/preprocessed_era5",
    history_steps=5, forecast_steps=1, grid_size=40,
    pressure_level_vars=["u", "v", "z"],
    pressure_levels=[850, 500, 250],
    surface_vars=[], num_workers=4, pin_memory=True,
)

model_cfg = ModelConfig(
    in_channels=data_cfg.num_channels, cond_channels=data_cfg.condition_channels,
    d_model=384, n_heads=6, n_dit_layers=12, n_cond_layers=3,
    ff_mult=4, patch_size=4, dropout=0.1,
    use_grouped_conv=False, num_var_groups=3, time_embedding_scale=1000.0,
)

train_cfg = TrainConfig(batch_size=8, seed=42, use_channel_weights=True)

channel_names = [f"{v}_{l}" for v in ["u", "v", "z"] for l in [850, 500, 250]]
z_channel_idxs = [6, 7, 8]  # z_850, z_500, z_250
u_channel_idxs = [0, 1, 2]
v_channel_idxs = [3, 4, 5]

print("=" * 70)
print("Z-CHANNEL COLLAPSE DEEP INVESTIGATION")
print("=" * 70)

# Load checkpoint
ckpt_path = "multi_seed_results_fixed_long/seed_42/checkpoints_fm/best.pt"
ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

fm_model = UnifiedModel(model_cfg, data_cfg, train_cfg, method="fm").to(device)
fm_model.load_state_dict(ckpt["model_state_dict"], strict=False)
ema = ckpt.get("ema_state_dict", {}).get("shadow", {})
for name, param in fm_model.named_parameters():
    if name in ema:
        param.data.copy_(ema[name].to(param.device, param.dtype))
fm_model.eval()

print(f"Checkpoint: epoch={ckpt.get('epoch')}, val_loss={ckpt.get('best_val_loss'):.4f}")

# Load data
_, _, test_loader, norm_mean, norm_std = build_dataloaders(data_cfg, train_cfg)

if isinstance(norm_mean, torch.Tensor):
    norm_mean = norm_mean.numpy()
if isinstance(norm_std, torch.Tensor):
    norm_std = norm_std.numpy()

print(f"\n=== NORMALIZATION STATS ===")
for i, cn in enumerate(channel_names):
    print(f"  {cn}: mean={norm_mean[i]:.4f}, std={norm_std[i]:.4f}")

# ============================================================
# 1. Get test batch
# ============================================================
test_batch = next(iter(test_loader))
cond = test_batch["condition"].to(device)
target = test_batch["target"].to(device)
B = cond.shape[0]
C = data_cfg.num_channels

x0 = cond[:, -C:, :, :]

print(f"\n=== DATA STATS ===")
print(f"Batch size: {B}")
print(f"\nTarget (normalized):")
for i, cn in enumerate(channel_names):
    print(f"  {cn:6s}: mean={target[:, i].mean():+.4f}, std={target[:, i].std():.4f}")

print(f"\nx0 (condition last step, normalized):")
for i, cn in enumerate(channel_names):
    print(f"  {cn:6s}: mean={x0[:, i].mean():+.4f}, std={x0[:, i].std():.4f}")

# ============================================================
# 2. Velocity predictions at t=0 (should approximate x1-x0)
# ============================================================
print("\n" + "=" * 70)
print("2. VELOCITY PREDICTIONS AT t=0 (should = x1-x0)")
print("=" * 70)

t_zero = torch.zeros(B, device=device)
v_pred_t0 = fm_model.dit(x0, t_zero, cond)

true_v = target - x0  # In flow matching, true velocity = x1 - x0

print("\nModel velocity prediction vs True velocity at t=0:")
for ch_idx in range(9):
    pred_mean = v_pred_t0[:, ch_idx].mean().item()
    pred_std = v_pred_t0[:, ch_idx].std().item()
    true_mean = true_v[:, ch_idx].mean().item()
    true_std = true_v[:, ch_idx].std().item()
    bias = pred_mean - true_mean
    error = (pred_mean - true_mean)**2
    print(f"  {channel_names[ch_idx]:6s}: pred={pred_mean:+.4f}±{pred_std:.4f}  true={true_mean:+.4f}±{true_std:.4f}  bias={bias:+.4f}")

print("\n  >>> Z-channel velocity predictions:")
for i in z_channel_idxs:
    pred_mean = v_pred_t0[:, i].mean().item()
    true_mean = true_v[:, i].mean().item()
    ratio = pred_mean / true_mean if abs(true_mean) > 1e-6 else 0
    print(f"    {channel_names[i]}: pred_v={pred_mean:+.4f}, true_v={true_mean:+.4f}, ratio={ratio:.2f}")

# ============================================================
# 3. Velocity predictions at different t
# ============================================================
print("\n" + "=" * 70)
print("3. VELOCITY PREDICTIONS AT DIFFERENT T VALUES")
print("=" * 70)

for t_val in [0.0, 0.25, 0.5, 0.75, 1.0]:
    t_batch = torch.full((B,), t_val, device=device)
    # Create x_t = (1-t)*x0 + t*x1
    x_t = (1 - t_val) * x0 + t_val * target

    v_pred = fm_model.dit(x_t, t_batch, cond)

    print(f"\n  t={t_val:.2f} (x_t = (1-{t_val:.2f})*x0 + {t_val:.2f}*x1):")
    for ch_idx in range(9):
        v = v_pred[:, ch_idx].mean().item()
        print(f"    {channel_names[ch_idx]:6s}: v={v:+.4f}")

# ============================================================
# 4. True velocity field analysis
# ============================================================
print("\n" + "=" * 70)
print("4. TRUE VELOCITY FIELD (x1 - x0) ANALYSIS")
print("=" * 70)

print("\nVelocity magnitude (x1 - x0):")
for ch_idx in range(9):
    tv_mean = true_v[:, ch_idx].mean().item()
    tv_std = true_v[:, ch_idx].std().item()
    print(f"  {channel_names[ch_idx]:6s}: mean={tv_mean:+.4f}, std={tv_std:.4f}")

z_v_magnitude = true_v[:, z_channel_idxs].abs().mean().item()
uv_v_magnitude = true_v[:, :6].abs().mean().item()
print(f"\n  → Z velocity magnitude: {z_v_magnitude:.4f}")
print(f"  → U/V velocity magnitude: {uv_v_magnitude:.4f}")
print(f"  → Z/U/V ratio: {z_v_magnitude/uv_v_magnitude:.1f}x")

# ============================================================
# 5. Sampling trajectory analysis
# ============================================================
print("\n" + "=" * 70)
print("5. SAMPLING TRAJECTORY (Euler, 4 steps)")
print("=" * 70)

def sample_with_trajectory(model, cond, device, euler_steps, clamp_range, z_clamp_range, C):
    """Sample and return trajectory of x_t at each step."""
    x_t = cond[:, -C:, :, :].clone()
    t = 0.0
    dt = 1.0 / euler_steps
    trajectory = [x_t.clone()]

    for step in range(euler_steps):
        t_tensor = torch.full((cond.shape[0],), t, device=device)
        v_pred = model.dit(x_t, t_tensor, cond)

        if step == 0:
            print(f"\n  Step {step}: t={t:.2f}, v_pred z_850={v_pred[:, 6].mean():+.4f}, "
                  f"z_500={v_pred[:, 7].mean():+.4f}, z_250={v_pred[:, 8].mean():+.4f}")

        x_next = x_t + dt * v_pred

        if clamp_range is not None:
            x_next_clamped = torch.clamp(x_next, *clamp_range)
        else:
            x_next_clamped = x_next

        if step == euler_steps - 1 and z_clamp_range is not None:
            x_next_clamped[:, z_channel_idxs] = x_next_clamped[:, z_channel_idxs].clamp(*z_clamp_range)

        x_t = x_next_clamped
        t += dt
        trajectory.append(x_t.clone())

    if euler_steps == 1:
        print(f"\n  Step 1: t={t:.2f}, z_850={x_t[:, 6].mean():+.4f}, z_500={x_t[:, 7].mean():+.4f}, z_250={x_t[:, 8].mean():+.4f}")

    return trajectory

print("\n--- Trajectory WITHOUT clamp ---")
traj_no_clamp = sample_with_trajectory(fm_model, cond, device, 4,
    clamp_range=None, z_clamp_range=None, C=C)

print("\n--- Trajectory WITH clamp_range=(-3,3) only ---")
traj_clamp_only = sample_with_trajectory(fm_model, cond, device, 4,
    clamp_range=(-3.0, 3.0), z_clamp_range=None, C=C)

print("\n--- Trajectory WITH clamp_range=(-3,3) + z_clamp=(-2,2) ---")
traj_z_clamp = sample_with_trajectory(fm_model, cond, device, 4,
    clamp_range=(-3.0, 3.0), z_clamp_range=(-2.0, 2.0), C=C)

print("\n--- Final predictions comparison ---")
print(f"  {'Method':<25} {'z_850 std':>10} {'z_500 std':>10} {'z_250 std':>10}")
print(f"  {'-'*55}")
print(f"  {'No clamp':<25} {traj_no_clamp[-1][:, 6].std():>10.4f} {traj_no_clamp[-1][:, 7].std():>10.4f} {traj_no_clamp[-1][:, 8].std():>10.4f}")
print(f"  {'Clamp only':<25} {traj_clamp_only[-1][:, 6].std():>10.4f} {traj_clamp_only[-1][:, 7].std():>10.4f} {traj_clamp_only[-1][:, 8].std():>10.4f}")
print(f"  {'Z clamp':<25} {traj_z_clamp[-1][:, 6].std():>10.4f} {traj_z_clamp[-1][:, 7].std():>10.4f} {traj_z_clamp[-1][:, 8].std():>10.4f}")
print(f"  {'GT':<25} {target[:, 6].std():>10.4f} {target[:, 7].std():>10.4f} {target[:, 8].std():>10.4f}")

# ============================================================
# 6. Full inference comparison
# ============================================================
print("\n" + "=" * 70)
print("6. FULL INFERENCE COMPARISON (200 samples)")
print("=" * 70)

n_eval = 200
n_batches = min(n_eval // train_cfg.batch_size, len(test_loader))

all_preds = {name: [] for name in ["no_clamp", "clamp_only", "z_clamp"]}
all_gts = []

with torch.no_grad():
    for batch_idx, batch in enumerate(test_loader):
        if batch_idx >= n_batches:
            break
        c = batch["condition"].to(device)
        t = batch["target"].to(device)

        p1 = fm_model.sample_fm(c, device, euler_steps=4, euler_mode="midpoint",
            clamp_range=None, z_clamp_range=None)
        p2 = fm_model.sample_fm(c, device, euler_steps=4, euler_mode="midpoint",
            clamp_range=(-3.0, 3.0), z_clamp_range=None)
        p3 = fm_model.sample_fm(c, device, euler_steps=4, euler_mode="midpoint",
            clamp_range=(-3.0, 3.0), z_clamp_range=(-2.0, 2.0))

        all_preds["no_clamp"].append(p1.cpu())
        all_preds["clamp_only"].append(p2.cpu())
        all_preds["z_clamp"].append(p3.cpu())
        all_gts.append(t.cpu())

all_gts = torch.cat(all_gts, dim=0)
for name in all_preds:
    all_preds[name] = torch.cat(all_preds[name], dim=0)

print(f"\n=== RMSE Comparison ===")
print(f"  {'Method':<25} {'All':>8} {'u':>8} {'v':>8} {'z':>8}")
print(f"  {'-'*55}")

for name, label in [("no_clamp", "No clamp"), ("clamp_only", "Clamp only"), ("z_clamp", "Z clamp")]:
    preds = all_preds[name]

    all_rmse = np.sqrt(((preds - all_gts) ** 2).mean())
    u_rmse = np.sqrt(((preds[:, :6] - all_gts[:, :6]) ** 2).mean())
    v_rmse = np.sqrt(((preds[:, 3:] - all_gts[:, 3:]) ** 2).mean())
    z_rmse = np.sqrt(((preds[:, 6:] - all_gts[:, 6:]) ** 2).mean())

    print(f"  {label:<25} {all_rmse:>8.4f} {u_rmse:>8.4f} {v_rmse:>8.4f} {z_rmse:>8.4f}")

print(f"\n=== StdR Comparison (should be ~1.0) ===")
print(f"  {'Method':<25} {'z_850':>8} {'z_500':>8} {'z_250':>8}")
print(f"  {'-'*50}")

for name, label in [("no_clamp", "No clamp"), ("clamp_only", "Clamp only"), ("z_clamp", "Z clamp")]:
    preds = all_preds[name]
    gt_std = all_gts[:, 6:].std(dim=(0, 2, 3))
    pred_std = preds[:, 6:].std(dim=(0, 2, 3))
    stdr = pred_std / gt_std
    print(f"  {label:<25} {stdr[0]:>8.3f} {stdr[1]:>8.3f} {stdr[2]:>8.3f}")

# ============================================================
# 7. The real problem: velocity prediction magnitude
# ============================================================
print("\n" + "=" * 70)
print("7. ROOT CAUSE: VELOCITY PREDICTION MAGNITUDE")
print("=" * 70)

# The integration is: x_{n+1} = x_n + v_pred * dt
# With 4 steps, dt=0.25. Total change = sum(v_pred * 0.25) ≈ v_pred * 1.0 (average)
# So total change ≈ average v_pred

# Check what the model predicts at different t values for z channels
z_v_preds = []
z_true_vs = []

with torch.no_grad():
    for batch_idx, batch in enumerate(test_loader):
        if batch_idx >= 20:
            break
        c = batch["condition"].to(device)
        t = batch["target"].to(device)
        x0 = c[:, -C:, :, :]

        for t_val in [0.0, 0.25, 0.5, 0.75]:
            x_t = (1 - t_val) * x0 + t_val * t
            t_batch = torch.full((c.shape[0],), t_val, device=device)
            v_pred = fm_model.dit(x_t, t_batch, c)
            z_v_preds.append(v_pred[:, z_channel_idxs].cpu())
            z_true_vs.append((t - x0).cpu())

z_v_preds = torch.cat(z_v_preds)
z_true_vs = torch.cat(z_true_vs)

print("\nVelocity prediction accuracy across t:")
print(f"  {'t':>5} {'Channel':>8} {'Pred |V|':>10} {'True |V|':>10} {'Ratio':>8} {'Bias':>10}")
print(f"  {'-'*50}")

for t_idx, t_val in enumerate([0.0, 0.25, 0.5, 0.75]):
    n = len(z_v_preds) // 4
    start = t_idx * n
    end = (t_idx + 1) * n
    vp = z_v_preds[start:end]
    tv = z_true_vs[start:end]

    for ci, cn in enumerate(channel_names[6:]):
        pred_mag = vp[:, ci].abs().mean().item()
        true_mag = tv[:, ci].abs().mean().item()
        ratio = pred_mag / true_mag if true_mag > 1e-8 else 0
        bias = (vp[:, ci].mean() - tv[:, ci].mean()).item()
        print(f"  {t_val:>5.2f} {cn:>8s} {pred_mag:>10.4f} {true_mag:>10.4f} {ratio:>8.2f} {bias:>+10.4f}")

# ============================================================
# 8. Diagnosis
# ============================================================
print("\n" + "=" * 70)
print("8. DIAGNOSIS")
print("=" * 70)

# Check if v_pred at t=0 is close to true velocity
z_pred_v_t0 = v_pred_t0[:, z_channel_idxs].mean().item()
z_true_v_t0 = true_v[:, z_channel_idxs].mean().item()

print(f"""
ROOT CAUSE ANALYSIS:
====================

1. TRUE VELOCITY FOR Z:
   - z_850: true_v_mean = {true_v[:, 6].mean():.4f}, true_v_std = {true_v[:, 6].std():.4f}
   - z_500: true_v_mean = {true_v[:, 7].mean():.4f}, true_v_std = {true_v[:, 7].std():.4f}
   - z_250: true_v_mean = {true_v[:, 8].mean():.4f}, true_v_std = {true_v[:, 8].std():.4f}
   
   → Z velocity is 3-5x LARGER than u/v velocity

2. MODEL PREDICTION AT t=0:
   - z_850: pred_v_mean = {v_pred_t0[:, 6].mean():.4f}
   - z_500: pred_v_mean = {v_pred_t0[:, 7].mean():.4f}
   - z_250: pred_v_mean = {v_pred_t0[:, 8].mean():.4f}
   
   → Model predicts {v_pred_t0[:, 6].mean()/true_v[:, 6].mean()*100:.0f}% of true z_850 velocity
   → Model predicts {v_pred_t0[:, 7].mean()/true_v[:, 7].mean()*100:.0f}% of true z_500 velocity
   → Model predicts {v_pred_t0[:, 8].mean()/true_v[:, 8].mean()*100:.0f}% of true z_250 velocity

3. INTEGRATION EFFECT:
   - With Euler steps=4, dt=0.25
   - Total change ≈ sum(v_pred * 0.25) for each step
   - If model under-predicts velocity by 90%, z predictions ≈ x0 + 0.1*(x1-x0)
   - This means z_pred ≈ x0 (persistence-like, collapse!)

4. CLAMPING EFFECT:
   - clamp_range=(-3, 3) on x_t is NOT the main issue for z
   - z_clamp=(-2, 2) at final step DOES affect final z values
   - But the REAL problem is the velocity predictions are too small
""")

# ============================================================
# 9. Quick fix test
# ============================================================
print("\n" + "=" * 70)
print("9. QUICK FIX TEST")
print("=" * 70)

print("""
POTENTIAL FIXES:
================

1. Increase z channel loss weight during training
   - Current: channel_weights=(1.0, 1.0, ...) for all channels
   - Fix: channel_weights for z channels

2. Scale z velocity loss
   - Current: velocity_loss_scale=1.0 for all
   - Fix: channel-specific velocity scaling

3. The model may need to LEARN larger z velocities
   - Check if z loss is actually backpropagating properly
   - Z has 3-5x larger variance → needs proportionally larger gradient

4. Check if model architecture handles z differently
   - Some architectures have channel-dependent processing
""")

# Check: what would happen if we scale z velocity predictions?
print("\n=== What-if: Scaling z velocity predictions ===")

with torch.no_grad():
    for batch_idx, batch in enumerate(test_loader):
        if batch_idx >= 3:
            break
        c = batch["condition"].to(device)
        t = batch["target"].to(device)
        x0 = c[:, -C:, :, :]

        # Normal prediction
        pred_normal = fm_model.sample_fm(c, device, euler_steps=4,
            clamp_range=None, z_clamp_range=None)

        # Velocity-scaled prediction (manually scale z velocity)
        x_t = x0.clone()
        for step in range(4):
            t_batch = torch.full((c.shape[0],), step * 0.25, device=device)
            v_pred = fm_model.dit(x_t, t_batch, c)
            # Scale z velocity by 2x
            v_pred_scaled = v_pred.clone()
            v_pred_scaled[:, z_channel_idxs] *= 2.0
            x_t = x_t + 0.25 * v_pred_scaled

        print(f"  Batch {batch_idx}:")
        print(f"    Normal z_850 std: {pred_normal[:, 6].std():.4f}")
        print(f"    Scaled z_850 std: {x_t[:, 6].std():.4f}")
        print(f"    GT z_850 std:    {t[:, 6].std():.4f}")
