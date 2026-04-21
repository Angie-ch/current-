"""
Check: Is the z variance issue specific to FM or universal?
Load both FM and DM from the same experiment (multi_seed_results)
and compare z channel std ratios.
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

def load_model(method, ckpt_path):
    model = UnifiedModel(model_cfg, data_cfg, train_cfg, method=method).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    ema = ckpt.get("ema_state_dict", {}).get("shadow", {})
    for name, param in model.named_parameters():
        if name in ema:
            param.data.copy_(ema[name].to(param.device, param.dtype))
    model.eval()
    return model, ckpt.get("epoch", "?")

# Load both FM and DM from multi_seed_results (the oldest, most trained run)
fm_model, fm_ep = load_model("fm", "multi_seed_results/seed_42/checkpoints_fm/best.pt")
dm_model, dm_ep = load_model("dm", "multi_seed_results/seed_42/checkpoints_dm/best.pt")
print(f"FM epoch: {fm_ep}, DM epoch: {dm_ep}")

# Also load FM from zfix run
fm_zfix, fm_zfix_ep = load_model("fm", "multi_seed_results_zfix/seed_42/checkpoints_fm/best.pt")
print(f"FM_zfix epoch: {fm_zfix_ep}")

n_batches = 10
results = {"FM_v3": [], "DM_v3": [], "FM_zfix": [], "GT": [], "x0": []}

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
        fm_zfix_pred = fm_zfix.sample_fm(cond, device,
            euler_steps=4, euler_mode="midpoint",
            clamp_range=None, z_clamp_range=None)

        results["FM_v3"].append(fm_pred.cpu().numpy())
        results["DM_v3"].append(dm_pred.cpu().numpy())
        results["FM_zfix"].append(fm_zfix_pred.cpu().numpy())
        results["GT"].append(target.cpu().numpy())
        results["x0"].append(x0.cpu().numpy())

for k in results:
    results[k] = np.concatenate(results[k], axis=0)
n = len(results["GT"])

# Stats
print(f"\n{'='*90}")
print(f"{'Channel':<10} {'GT std':>8} | {'FM_v3 StdR':>11} {'DM_v3 StdR':>11} {'FM_zfix StdR':>12} | {'FM_v3 RMSE':>10} {'DM_v3 RMSE':>10} {'FM_zfix RMSE':>12}")
print(f"{'='*90}")
for ch_idx, cn in enumerate(channel_names):
    gt = results["GT"][:n, ch_idx]
    fm_v3 = results["FM_v3"][:n, ch_idx]
    dm_v3 = results["DM_v3"][:n, ch_idx]
    fm_zfix = results["FM_zfix"][:n, ch_idx]
    x0 = results["x0"][:n, ch_idx]

    gt_std = gt.std()
    fm_v3_sr = fm_v3.std() / gt_std if gt_std > 0 else 0
    dm_v3_sr = dm_v3.std() / gt_std if gt_std > 0 else 0
    fm_zfix_sr = fm_zfix.std() / gt_std if gt_std > 0 else 0

    fm_v3_rmse = np.sqrt(((fm_v3 - gt)**2).mean())
    dm_v3_rmse = np.sqrt(((dm_v3 - gt)**2).mean())
    fm_zfix_rmse = np.sqrt(((fm_zfix - gt)**2).mean())

    print(f"{cn:<10} {gt_std:>8.3f} | {fm_v3_sr:>11.3f} {dm_v3_sr:>11.3f} {fm_zfix_sr:>12.3f} | {fm_v3_rmse:>10.4f} {dm_v3_rmse:>10.4f} {fm_zfix_rmse:>12.4f}")
