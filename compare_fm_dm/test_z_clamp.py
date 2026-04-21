"""
Compare z-clamped vs un-clamped sampling for FM model.
"""
import os, sys, numpy as np, torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

sys.path.insert(0, '.')
from configs.config import DataConfig, ModelConfig, TrainConfig
from data.dataset import build_dataloaders
from models.unified_model import UnifiedModel

def load_checkpoint(model, path, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    sd = ckpt["model_state_dict"]
    model.load_state_dict(sd, strict=False)
    ema_shadow = ckpt.get("ema_state_dict", {}).get("shadow", {})
    for name, param in model.named_parameters():
        if name in ema_shadow:
            param.data.copy_(ema_shadow[name].to(param.device, param.dtype))
    print(f"Loaded: {path}  epoch={ckpt.get('epoch','?')}")
    return model

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
fm_model = load_checkpoint(fm_model, "multi_seed_results_20ep/seed_42/checkpoints_fm/best.pt", device)
fm_model.eval()

# Inference configs
configs = {
    "z_clamp(-2,2)":  dict(clamp_range=(-3.0, 3.0), z_clamp_range=(-2.0, 2.0)),
    "z_clamp(-5,5)":  dict(clamp_range=(-3.0, 3.0), z_clamp_range=(-5.0, 5.0)),
    "no z_clamp":     dict(clamp_range=(-3.0, 3.0), z_clamp_range=None),
    "wider clamp":    dict(clamp_range=(-5.0, 5.0), z_clamp_range=None),
}

all_preds = {}
all_gts = None
batch_count = 0
max_batches = 5

print("Running inference...")
with torch.no_grad():
    for batch_idx, batch in enumerate(test_loader):
        if batch_idx >= max_batches:
            break
        cond = batch["condition"].to(device)
        target = batch["target"].to(device)
        if all_gts is None:
            all_gts = target.cpu().numpy()
        else:
            all_gts = np.concatenate([all_gts, target.cpu().numpy()], axis=0)

        for name, kwargs in configs.items():
            fm_pred = fm_model.sample_fm(cond, device,
                euler_steps=4, euler_mode="midpoint", **kwargs)
            if name not in all_preds:
                all_preds[name] = []
            all_preds[name].append(fm_pred.cpu().numpy())
        batch_count += 1

for name in all_preds:
    all_preds[name] = np.concatenate(all_preds[name], axis=0)

n = len(list(all_preds.values())[0])
print(f"Collected {n} samples from {batch_count} batches")

# Denormalize
std = np.where(norm_std < 1e-8, 1.0, norm_std)
gt_dn = all_gts * std[np.newaxis, :, np.newaxis, np.newaxis] + norm_mean[np.newaxis, :, np.newaxis, np.newaxis]

channel_names = [f"{v}_{l}" for v in ["u", "v", "z"] for l in [850, 500, 250]]

# === Summary table ===
print(f"\n{'='*90}")
print(f"{'Config':<20} " + " ".join([f"{cn:>12}" for cn in channel_names]))
print(f"{'='*90}")

for name, preds in all_preds.items():
    pred_dn = preds * std[np.newaxis, :, np.newaxis, np.newaxis] + norm_mean[np.newaxis, :, np.newaxis, np.newaxis]
    rmse_list = []
    for ch_idx in range(9):
        rmse = np.sqrt(((preds[:n, ch_idx] - all_gts[:n, ch_idx])**2).mean())
        rmse_list.append(rmse)
    row = f"{name:<20} " + " ".join([f"{r:>12.4f}" for r in rmse_list])
    print(row)

print(f"\n{'='*90}")
print(f"{'GT std:':<20} " + " ".join([f"{gt_dn[:n, i].std():>12.4f}" for i in range(9)]))

# === Std ratio table ===
print(f"\n{'='*90}")
print(f"{'Config':<20} " + " ".join([f"{cn:>12}" for cn in channel_names]))
print(f"{'='*90}")
for name, preds in all_preds.items():
    pred_dn = preds * std[np.newaxis, :, np.newaxis, np.newaxis] + norm_mean[np.newaxis, :, np.newaxis, np.newaxis]
    ratio_list = []
    for ch_idx in range(9):
        ratio = pred_dn[:n, ch_idx].std() / gt_dn[:n, ch_idx].std() if gt_dn[:n, ch_idx].std() > 0 else 0
        ratio_list.append(ratio)
    row = f"{name:<20} " + " ".join([f"{r:>12.3f}" for r in ratio_list])
    print(row)
print(f"\nGT std ratio always = 1.000")

# === Plot comparison for z channels ===
fig, axes = plt.subplots(3, len(configs) + 1, figsize=(3.5 * (len(configs) + 1), 10))
gt_sample = gt_dn[0]

for row, (z_name, z_level) in enumerate([("z_850", 6), ("z_500", 7), ("z_250", 8)]):
    ax = axes[row, 0]
    im = ax.imshow(gt_sample[z_level], cmap='RdBu_r', origin='lower')
    ax.set_title(f"GT: {z_name}\nstd={gt_sample[z_level].std():.1f}", fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    for col, (name, preds) in enumerate(all_preds.items(), 1):
        pred_dn = preds * std[np.newaxis, :, np.newaxis, np.newaxis] + norm_mean[np.newaxis, :, np.newaxis, np.newaxis]
        ax = axes[row, col]
        im = ax.imshow(pred_dn[0, z_level], cmap='RdBu_r', origin='lower')
        ax.set_title(f"{name}\nstd={pred_dn[0, z_level].std():.1f}", fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

plt.suptitle("Z-channel predictions: GT vs different clamp settings", fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig("channel_plots/z_clamp_comparison.png", dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("\nSaved: channel_plots/z_clamp_comparison.png")
