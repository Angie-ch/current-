"""
Run FM vs DM comparison using 20ep FM checkpoint with proper Euler steps (4+).
"""
import os, sys, json, numpy as np, torch
from tqdm import tqdm

sys.path.insert(0, '.')
from configs.config import DataConfig, ModelConfig, TrainConfig
from data.dataset import build_dataloaders
from models.unified_model import UnifiedModel

device = torch.device('cuda')

# ============================================================
# Configs
# ============================================================
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

# ============================================================
# Load data
# ============================================================
print("Loading data...")
_, _, test_loader, norm_mean, norm_std = build_dataloaders(data_cfg, train_cfg)
if isinstance(norm_mean, torch.Tensor): norm_mean = norm_mean.numpy()
if isinstance(norm_std, torch.Tensor): norm_std = norm_std.numpy()

C = data_cfg.num_channels  # 9
channel_names = [f"{v}_{l}" for v in ["u", "v", "z"] for l in [850, 500, 250]]

# ============================================================
# Load FM model (20ep checkpoint)
# ============================================================
print("Loading FM model (20ep)...")
fm_model = UnifiedModel(model_cfg, data_cfg, train_cfg, method="fm").to(device)
ckpt = torch.load("multi_seed_results_20ep/seed_42/checkpoints_fm/best.pt", map_location=device, weights_only=False)
fm_model.load_state_dict(ckpt["model_state_dict"], strict=False)
ema = ckpt.get("ema_state_dict", {}).get("shadow", {})
for name, param in fm_model.named_parameters():
    if name in ema:
        param.data.copy_(ema[name].to(param.device, param.dtype))
fm_model.eval()
print("FM model loaded (EMA applied)")

# ============================================================
# Load DM model
# ============================================================
dm_path = "multi_seed_results_20ep/seed_42/checkpoints_dm/best.pt"
print(f"Loading DM model: {dm_path}...")
dm_model = UnifiedModel(model_cfg, data_cfg, train_cfg, method="dm").to(device)
if os.path.exists(dm_path):
    dm_ckpt = torch.load(dm_path, map_location=device, weights_only=False)
    dm_model.load_state_dict(dm_ckpt["model_state_dict"], strict=False)
    ema_dm = dm_ckpt.get("ema_state_dict", {}).get("shadow", {})
    for name, param in dm_model.named_parameters():
        if name in ema_dm:
            param.data.copy_(ema_dm[name].to(param.device, param.dtype))
    print("DM model loaded (EMA applied)")
else:
    print("ERROR: DM checkpoint not found!")
    sys.exit(1)
dm_model.eval()

# ============================================================
# Evaluate FM with different Euler steps
# ============================================================
print("\n=== Evaluating FM (20ep) with different Euler steps ===")
fm_steps_to_try = [1, 2, 4, 8]

for n_steps in fm_steps_to_try:
    all_preds, all_gts = [], []
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            if batch_idx >= 10:
                break
            cond = batch["condition"].to(device)
            target = batch["target"].to(device)
            pred = fm_model.sample_fm(cond, device,
                euler_steps=n_steps, euler_mode="midpoint",
                clamp_range=None, z_clamp_range=None)
            all_preds.append(pred.cpu().numpy())
            all_gts.append(target.cpu().numpy())

    preds = np.concatenate(all_preds, axis=0)
    gts = np.concatenate(all_gts, axis=0)

    print(f"\nFM Steps={n_steps}:")
    for ch_idx, cn in enumerate(channel_names):
        rmse = np.sqrt(((preds[:len(preds), ch_idx] - gts[:len(preds), ch_idx])**2).mean())
        std_r = preds[:len(preds), ch_idx].std() / gts[:len(preds), ch_idx].std() if gts[:len(preds), ch_idx].std() > 0 else 0
        marker = " **" if (cn.startswith("z_") and abs(std_r - 1.0) > 0.05) else ""
        print(f"  {cn}: RMSE={rmse:.4f}, StdR={std_r:.3f}{marker}")

# ============================================================
# Evaluate DM
# ============================================================
print("\n=== Evaluating DM ===")
all_preds_dm, all_gts_dm = [], []
with torch.no_grad():
    for batch_idx, batch in enumerate(test_loader):
        if batch_idx >= 10:
            break
        cond = batch["condition"].to(device)
        target = batch["target"].to(device)
        pred = dm_model.sample_dm(cond, device, ddim_steps=50)
        all_preds_dm.append(pred.cpu().numpy())
        all_gts_dm.append(target.cpu().numpy())

preds_dm = np.concatenate(all_preds_dm, axis=0)
gts_dm = np.concatenate(all_gts_dm, axis=0)

print(f"\nDM (50 DDIM steps):")
for ch_idx, cn in enumerate(channel_names):
    rmse = np.sqrt(((preds_dm[:len(preds_dm), ch_idx] - gts_dm[:len(preds_dm), ch_idx])**2).mean())
    std_r = preds_dm[:len(preds_dm), ch_idx].std() / gts_dm[:len(preds_dm), ch_idx].std() if gts_dm[:len(preds_dm), ch_idx].std() > 0 else 0
    marker = " **" if (cn.startswith("z_") and abs(std_r - 1.0) > 0.05) else ""
    print(f"  {cn}: RMSE={rmse:.4f}, StdR={std_r:.3f}{marker}")

print("\nDone!")
