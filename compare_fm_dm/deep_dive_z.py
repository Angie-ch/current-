"""
Deep dive: why does FM underpredict z channel variance?
1. Check if loss weights down-weight z channels
2. Check per-channel training loss from logs
3. Check channel_weights config
4. Check if z prediction is stuck or diverges
"""
import os, sys, numpy as np, torch

sys.path.insert(0, '.')
from configs.config import DataConfig, ModelConfig, TrainConfig
from data.dataset import build_dataloaders
from models.unified_model import UnifiedModel
import torch.nn.functional as F

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
print(f"use_channel_weights: {train_cfg.use_channel_weights}")
print(f"channel_weights: {getattr(train_cfg, 'channel_weights', None)}")
print(f"loss_mode: {getattr(train_cfg, 'loss_mode', None)}")

# Load model
fm_model = UnifiedModel(model_cfg, data_cfg, train_cfg, method="fm").to(device)
ckpt = torch.load("multi_seed_results_fixed_long/seed_42/checkpoints_fm/best.pt", map_location=device, weights_only=False)
fm_model.load_state_dict(ckpt["model_state_dict"], strict=False)
ema = ckpt.get("ema_state_dict", {}).get("shadow", {})
for name, param in fm_model.named_parameters():
    if name in ema:
        param.data.copy_(ema[name].to(param.device, param.dtype))
fm_model.eval()

C = data_cfg.num_channels  # 9
channel_names = [f"{v}_{l}" for v in ["u", "v", "z"] for l in [850, 500, 250]]
print(f"C={C}, condition_channels={data_cfg.condition_channels}")

# ============================================================
# 1. Check channel weights used in loss
# ============================================================
print("\n=== 1. Channel weight analysis ===")
if hasattr(fm_model, 'channel_weights') and fm_model.channel_weights is not None:
    cw = fm_model.channel_weights
    print(f"channel_weights tensor: {cw}")
else:
    print("No channel_weights found on model")

if hasattr(train_cfg, 'channel_weights') and train_cfg.channel_weights:
    print(f"TrainConfig channel_weights: {train_cfg.channel_weights}")
else:
    print("TrainConfig has no channel_weights")

# ============================================================
# 2. Compute per-channel loss for a batch (with and without weights)
# ============================================================
print("\n=== 2. Per-channel MSE loss (unweighted) ===")
_, _, test_loader, norm_mean, norm_std = build_dataloaders(data_cfg, train_cfg)
if isinstance(norm_mean, torch.Tensor): norm_mean = norm_mean.numpy()
if isinstance(norm_std, torch.Tensor): norm_std = norm_std.numpy()

n_batches = 5
all_channel_mse = {cn: [] for cn in channel_names}

with torch.no_grad():
    for batch_idx, batch in enumerate(test_loader):
        if batch_idx >= n_batches:
            break
        cond = batch["condition"].to(device)
        target = batch["target"].to(device)
        x0 = cond[:, -C:, :, :]

        x0_mse = ((x0 - target) ** 2)
        for ch_idx, cn in enumerate(channel_names):
            all_channel_mse[cn].append(x0_mse[:, ch_idx].mean().item())

print(f"{'Channel':<10} {'x0 MSE':>12} {'x0 RMSE':>12} {'x0 Std':>10}")
for cn in channel_names:
    mse = np.mean(all_channel_mse[cn])
    print(f"{cn:<10} {mse:>12.6f} {np.sqrt(mse):>12.4f} {np.std(all_channel_mse[cn]):>10.4f}")

# ============================================================
# 3. Step-by-step x_t evolution using hook on sample_fm
# ============================================================
print("\n=== 3. Step-by-step x_t evolution (via hook) ===")

step_outputs = []
def unpatchify_hook(module, input, output):
    step_outputs.append(output.detach().clone())

hook = fm_model.dit.unpatchify.register_forward_hook(unpatchify_hook)

fm_model.eval()
with torch.no_grad():
    for batch_idx, batch in enumerate(test_loader):
        if batch_idx >= 1:
            break
        cond = batch["condition"].to(device)
        target = batch["target"].to(device)
        B, TC, H, W = cond.shape  # TC = 45 (T=5, C=9)

        print(f"cond shape: {cond.shape}, target shape: {target.shape}")
        print(f"z_850: x0_std={cond[:, -C+6].std():.4f}, tgt_std={target[:, 6].std():.4f}")
        print(f"z_500: x0_std={cond[:, -C+7].std():.4f}, tgt_std={target[:, 7].std():.4f}")
        print(f"z_250: x0_std={cond[:, -C+8].std():.4f}, tgt_std={target[:, 8].std():.4f}")

        step_outputs.clear()
        pred = fm_model.sample_fm(cond, device,
            euler_steps=4, euler_mode="midpoint",
            clamp_range=None, z_clamp_range=None)

        print(f"\nPredicted (final): z_850={pred[:, 6].std():.4f}, z_500={pred[:, 7].std():.4f}, z_250={pred[:, 8].std():.4f}")
        print(f"Target:              z_850={target[:, 6].std():.4f}, z_500={target[:, 7].std():.4f}, z_250={target[:, 8].std():.4f}")
        print(f"\nPer-step unpatchify outputs ({len(step_outputs)} steps captured):")
        for i, out in enumerate(step_outputs):
            print(f"  Step {i+1}: z_850={out[:, 6].std():.4f}, z_500={out[:, 7].std():.4f}, z_250={out[:, 8].std():.4f}")

hook.remove()

# ============================================================
# 4. Check: does increasing Euler steps help z?
# ============================================================
print("\n=== 4. Euler steps sweep ===")
fm_model.eval()
for n_steps in [1, 2, 4, 8, 16]:
    all_preds = []
    all_gts_local = []
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            if batch_idx >= 3:
                break
            cond = batch["condition"].to(device)
            target = batch["target"].to(device)
            fm_pred = fm_model.sample_fm(cond, device,
                euler_steps=n_steps, euler_mode="midpoint",
                clamp_range=None, z_clamp_range=None)
            all_preds.append(fm_pred.cpu().numpy())
            all_gts_local.append(target.cpu().numpy())
    preds = np.concatenate(all_preds, axis=0)
    gts = np.concatenate(all_gts_local, axis=0)
    n = len(preds)
    print(f"\nSteps={n_steps}:")
    for ch_idx, cn in enumerate(channel_names):
        rmse = np.sqrt(((preds[:n, ch_idx] - gts[:n, ch_idx])**2).mean())
        std_r = preds[:n, ch_idx].std() / gts[:n, ch_idx].std() if gts[:n, ch_idx].std() > 0 else 0
        print(f"  {cn}: RMSE={rmse:.4f}, StdR={std_r:.3f}")
