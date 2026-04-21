"""
Compare FM vs DM channel performance on test set.
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

C = data_cfg.num_channels
channel_names = [f"{v}_{l}" for v in ["u", "v", "z"] for l in [850, 500, 250]]

# Load FM model
fm_model = UnifiedModel(model_cfg, data_cfg, train_cfg, method="fm").to(device)
ckpt = torch.load("multi_seed_results_20ep/seed_42/checkpoints_fm/best.pt", map_location=device, weights_only=False)
fm_model.load_state_dict(ckpt["model_state_dict"], strict=False)
ema_shadow = ckpt.get("ema_state_dict", {}).get("shadow", {})
for name, param in fm_model.named_parameters():
    if name in ema_shadow:
        param.data.copy_(ema_shadow[name].to(param.device, param.dtype))
fm_model.eval()
print(f"FM best epoch: {ckpt.get('epoch', '?')}")

# Load DM model
dm_paths = [
    "multi_seed_results_zfix/seed_42/checkpoints_dm/best.pt",
    "multi_seed_results_v3/seed_42/checkpoints_dm/best.pt",
]
dm_models = {}
for p in dm_paths:
    if os.path.exists(p):
        dm_model = UnifiedModel(model_cfg, data_cfg, train_cfg, method="dm").to(device)
        ckpt2 = torch.load(p, map_location=device, weights_only=False)
        dm_model.load_state_dict(ckpt2["model_state_dict"], strict=False)
        ema2 = ckpt2.get("ema_state_dict", {}).get("shadow", {})
        for name, param in dm_model.named_parameters():
            if name in ema2:
                param.data.copy_(ema2[name].to(param.device, param.dtype))
        dm_model.eval()
        short = p.split("/")[1] + "/" + p.split("/")[-1]
        dm_models[short] = dm_model
        print(f"DM loaded: {short}, epoch {ckpt2.get('epoch', '?')}")

n_batches = 10
all_fm_preds = []
all_dm_preds = {k: [] for k in dm_models}
all_gts = []
all_x0s = []

with torch.no_grad():
    for batch_idx, batch in enumerate(test_loader):
        if batch_idx >= n_batches:
            break
        cond = batch["condition"].to(device)
        target = batch["target"].to(device)
        x0 = cond[:, -C:, :, :]

        fm_pred = fm_model.sample_fm(cond, device,
            euler_steps=4, euler_mode="midpoint",
            clamp_range=None, z_clamp_range=None)

        all_fm_preds.append(fm_pred.cpu().numpy())
        all_gts.append(target.cpu().numpy())
        all_x0s.append(x0.cpu().numpy())

        for name, dm_model in dm_models.items():
            dm_pred = dm_model.sample_dm(cond, device, ddim_steps=50,
                clamp_range=None, z_clamp_range=None)
            all_dm_preds[name].append(dm_pred.cpu().numpy())

for k in all_dm_preds:
    all_dm_preds[k] = np.concatenate(all_dm_preds[k], axis=0)
all_fm_preds = np.concatenate(all_fm_preds, axis=0)
all_gts = np.concatenate(all_gts, axis=0)
all_x0s = np.concatenate(all_x0s, axis=0)
n = len(all_gts)
print(f"\nTest samples: {n}")
print()

# Table header
header = f"{'Channel':<10} {'x0 RMSE':>10} {'FM RMSE':>10} {'FM StdR':>10}"
for name in dm_models:
    header += f" {name[:15]:>16} {name[:15]:>16}"
print(header)
print("-" * len(header))

for ch_idx, cn in enumerate(channel_names):
    gt = all_gts[:n, ch_idx]
    x0 = all_x0s[:n, ch_idx]
    fm = all_fm_preds[:n, ch_idx]

    x0_rmse = np.sqrt(((x0 - gt)**2).mean())
    fm_rmse = np.sqrt(((fm - gt)**2).mean())
    fm_std_r = fm.std() / gt.std() if gt.std() > 0 else 0

    row = f"{cn:<10} {x0_rmse:>10.4f} {fm_rmse:>10.4f} {fm_std_r:>10.3f}"
    for name, dm_preds in all_dm_preds.items():
        dm = dm_preds[:n, ch_idx]
        dm_rmse = np.sqrt(((dm - gt)**2).mean())
        dm_std_r = dm.std() / gt.std() if gt.std() > 0 else 0
        row += f" {dm_rmse:>10.4f}/{dm_std_r:>5.3f}"
    print(row)

print()
# Improvement over x0
print(f"{'Channel':<10} {'x0->FM imp':>12} {'% imp':>8}")
for ch_idx, cn in enumerate(channel_names):
    gt = all_gts[:n, ch_idx]
    x0 = all_x0s[:n, ch_idx]
    fm = all_fm_preds[:n, ch_idx]
    x0_rmse = np.sqrt(((x0 - gt)**2).mean())
    fm_rmse = np.sqrt(((fm - gt)**2).mean())
    imp = x0_rmse - fm_rmse
    pct = imp / x0_rmse * 100 if x0_rmse > 0 else 0
    print(f"{cn:<10} {imp:>+12.4f} {pct:>+7.1f}%")
