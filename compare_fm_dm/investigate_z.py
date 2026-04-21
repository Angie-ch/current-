"""
Deep investigation: why does the FM model predict near-zero z variation?
1. Check raw model predictions vs GT in normalized space
2. Check if z has less gradient signal than u/v
3. Check per-channel loss contribution during training
"""
import os, sys, numpy as np, torch

sys.path.insert(0, '.')
from configs.config import DataConfig, ModelConfig, TrainConfig
from data.dataset import build_dataloaders
from models.unified_model import UnifiedModel

device = torch.device('cuda')

data_cfg = DataConfig(
    norm_stats_path="/root/autodl-tmp/fyp_final/Ver4/Trajectory/preprocessed_era5/norm_stats_year_split.pt",
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

_, _, test_loader, norm_mean, norm_std = build_dataloaders(data_cfg, train_cfg)
if isinstance(norm_mean, torch.Tensor): norm_mean = norm_mean.numpy()
if isinstance(norm_std, torch.Tensor): norm_std = norm_std.numpy()

fm_model = UnifiedModel(model_cfg, data_cfg, train_cfg, method="fm").to(device)

# Load checkpoint
ckpt = torch.load("multi_seed_results_20ep/seed_42/checkpoints_fm/best.pt", map_location=device, weights_only=False)
fm_model.load_state_dict(ckpt["model_state_dict"], strict=False)
ema_shadow = ckpt.get("ema_state_dict", {}).get("shadow", {})
for name, param in fm_model.named_parameters():
    if name in ema_shadow:
        param.data.copy_(ema_shadow[name].to(param.device, param.dtype))
fm_model.eval()

# ============================================================
# 1. Raw GT statistics in normalized space
# ============================================================
print("=== 1. GT statistics (normalized space) ===")
all_gts = []
with torch.no_grad():
    for batch_idx, batch in enumerate(test_loader):
        if batch_idx >= 3:
            break
        target = batch["target"].numpy()
        all_gts.append(target)
all_gts = np.concatenate(all_gts, axis=0)

channel_names = [f"{v}_{l}" for v in ["u", "v", "z"] for l in [850, 500, 250]]
for i, cn in enumerate(channel_names):
    ch_data = all_gts[:, i]
    print(f"  {cn:<8}: mean={ch_data.mean():+.4f}, std={ch_data.std():.4f}, "
          f"min={ch_data.min():.4f}, max={ch_data.max():.4f}, "
          f"std/mean_ratio={abs(ch_data.std()/ch_data.mean()):.4f}")

# ============================================================
# 2. Model velocity predictions (FM predicts x1 - x0)
# ============================================================
print("\n=== 2. Model velocity predictions (normalized space) ===")
all_vel = []
all_x0 = []
all_target = []
with torch.no_grad():
    for batch_idx, batch in enumerate(test_loader):
        if batch_idx >= 3:
            break
        cond = batch["condition"].to(device)
        target = batch["target"].to(device)
        x0 = cond[:, -data_cfg.num_channels:, :, :]
        pred = fm_model.sample_fm(cond, device,
            euler_steps=4, euler_mode="midpoint",
            clamp_range=None, z_clamp_range=None)

        all_vel.append((pred - x0).cpu().numpy())
        all_x0.append(x0.cpu().numpy())
        all_target.append(target.cpu().numpy())

vel = np.concatenate(all_vel, axis=0)
x0 = np.concatenate(all_x0, axis=0)
target = np.concatenate(all_target, axis=0)

for i, cn in enumerate(channel_names):
    v = vel[:, i]
    t = target[:, i]
    x = x0[:, i]
    print(f"  {cn:<8}: vel_mean={v.mean():+.4f}, vel_std={v.std():.4f}, "
          f"target_std={t.std():.4f}, x0_std={x.std():.4f}, "
          f"|vel|/target_std={np.abs(v).mean()/(t.std()+1e-8):.4f}")

# ============================================================
# 3. Per-step velocity analysis
# ============================================================
print("\n=== 3. Per-step velocity (step-by-step Euler) ===")
all_steps_vel = []
with torch.no_grad():
    for batch_idx, batch in enumerate(test_loader):
        if batch_idx >= 1:
            break
        cond = batch["condition"].to(device)
        B, C, H, W = cond.shape
        x_t = cond[:, -C:, :, :].clone()
        t = 0.0
        dt = 1.0 / 4

        cond_proc = fm_model._prepare_condition(cond)
        if cond_proc.ndim == 5:
            B5, T_c, _, H5, W5 = cond_proc.shape
            cond_proc = cond_proc.view(B5, T_c * data_cfg.num_channels, H5, W5)

        step_vels = []
        for step in range(4):
            t_next = t + dt
            t_tensor = torch.full((B,), t, device=device, dtype=torch.float32)
            v_pred = fm_model.dit(x_t, t_tensor, cond_proc)

            if step == 0:  # just record first step v_pred
                step_vels.append(v_pred.cpu().numpy())

            x_mid = x_t + dt / 2 * v_pred
            t_mid = t + dt / 2
            t_tensor2 = torch.full((B,), t_mid, device=device, dtype=torch.float32)
            v_pred2 = fm_model.dit(x_mid, t_tensor2, cond_proc)

            x_t = x_t + dt * v_pred2
            t = t_next

        all_steps_vel.append(step_vels[0])

vel_first = np.concatenate(all_steps_vel, axis=0)
for i, cn in enumerate(channel_names):
    v = vel_first[:, i]
    t = target[:, i]
    print(f"  {cn:<8}: first_step_vel_mean={v.mean():+.6f}, first_step_vel_std={v.std():.6f}, "
          f"first_step_vel/max_abs={np.abs(v).mean()/(np.abs(v).max()+1e-8):.4f}, "
          f"target_std={t.std():.4f}")

# ============================================================
# 4. Relative scale of z channels in model output
# ============================================================
print("\n=== 4. Model output magnitude per channel ===")
for i, cn in enumerate(channel_names):
    v = vel_first[:, i]
    all_ch_vel = np.concatenate([vel_first[:, j].reshape(-1) for j in range(9)])
    print(f"  {cn:<8}: vel_abs_mean={np.abs(v).mean():.6f}, "
          f"vel_frac_of_total={np.abs(v).mean()/(np.abs(all_ch_vel).mean()+1e-8):.4f}, "
          f"vel_std/other_ch_avg_std={v.std()/(vel_first[:, :6].std()+1e-8):.4f}")

# ============================================================
# 5. How much does the model deviate from x0?
# ============================================================
print("\n=== 5. Prediction vs x0 (starting point) ===")
for i, cn in enumerate(channel_names):
    p = pred.cpu().numpy()[:, i]
    t = target[:, i]
    x = x0[:, i]
    improvement = np.sqrt(((x - t)**2).mean()) - np.sqrt(((p - t)**2).mean())
    print(f"  {cn:<8}: x0_rmse={np.sqrt(((x-t)**2).mean()):.4f}, "
          f"pred_rmse={np.sqrt(((p-t)**2).mean()):.4f}, "
          f"improvement={improvement:+.4f}")
