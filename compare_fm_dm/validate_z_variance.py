"""
Validate z variance for both 20ep and 10ep FM checkpoints.
"""
import os, sys, numpy as np, torch
from tqdm import tqdm

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

channel_names = [f"{v}_{l}" for v in ["u", "v", "z"] for l in [850, 500, 250]]

print("Loading data...")
_, _, test_loader, norm_mean, norm_std = build_dataloaders(data_cfg, train_cfg)

# ============================================================
# Helper
# ============================================================
def load_fm(ckpt_path, label):
    print(f"\nLoading {label}: {ckpt_path}")
    model = UnifiedModel(model_cfg, data_cfg, train_cfg, method="fm").to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    ema = ckpt.get("ema_state_dict", {}).get("shadow", {})
    for name, param in model.named_parameters():
        if name in ema:
            param.data.copy_(ema[name].to(param.device, param.dtype))
    model.eval()
    return model

def eval_fm(model, n_steps, label):
    all_preds, all_gts = [], []
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            if batch_idx >= 5:
                break
            cond = batch["condition"].to(device)
            target = batch["target"].to(device)
            pred = model.sample_fm(cond, device,
                euler_steps=n_steps, euler_mode="midpoint",
                clamp_range=None, z_clamp_range=None)
            all_preds.append(pred.cpu().numpy())
            all_gts.append(target.cpu().numpy())

    preds = np.concatenate(all_preds, axis=0)
    gts = np.concatenate(all_gts, axis=0)
    n = len(preds)

    print(f"\n{label} FM Steps={n_steps}:")
    print(f"{'Channel':<10} {'RMSE':>10} {'StdR':>8} {'Status':>8}")
    print("-" * 40)
    ok_count = 0
    for ch_idx, cn in enumerate(channel_names):
        rmse = np.sqrt(((preds[:n, ch_idx] - gts[:n, ch_idx])**2).mean())
        std_r = preds[:n, ch_idx].std() / gts[:n, ch_idx].std() if gts[:n, ch_idx].std() > 0 else 0
        if cn.startswith("z_"):
            status = "OK" if 0.95 <= std_r <= 1.05 else "FAIL"
            if status == "OK":
                ok_count += 1
        else:
            status = "OK" if 0.95 <= std_r <= 1.05 else "WARN"
        print(f"{cn:<10} {rmse:>10.4f} {std_r:>8.3f} {status:>8}")
    if all(cn.startswith("z_") for cn in channel_names) or any(cn.startswith("z_") for cn in channel_names):
        z_ok = all(0.95 <= preds[:n, ch_idx].std() / gts[:n, ch_idx].std() <= 1.05 for ch_idx in range(6, 9))
        print(f"\nZ variance validated: {'YES' if z_ok else 'NO'}")
    return preds, gts

# ============================================================
# Load both models
# ============================================================
fm_20ep = load_fm("multi_seed_results_20ep/seed_42/checkpoints_fm/best.pt", "20ep FM")
fm_10ep = load_fm("multi_seed_results_fixed/seed_42/checkpoints_fm/best.pt", "10ep FM")

# ============================================================
# Evaluate both with 4 steps (optimal from earlier)
# ============================================================
print("\n" + "="*60)
print("Z VARIANCE VALIDATION (4 Euler steps)")
print("="*60)

preds_20, gts = eval_fm(fm_20ep, 4, "20ep")
preds_10, _ = eval_fm(fm_10ep, 4, "10ep")

# ============================================================
# Summary table
# ============================================================
print("\n" + "="*60)
print("COMPARISON SUMMARY (4 Euler steps)")
print("="*60)
print(f"{'Channel':<10} {'20ep StdR':>12} {'10ep StdR':>12} {'20ep OK':>10} {'10ep OK':>10}")
print("-" * 60)
for ch_idx, cn in enumerate(channel_names):
    std_20 = preds_20[:, ch_idx].std() / gts[:, ch_idx].std()
    std_10 = preds_10[:, ch_idx].std() / gts[:, ch_idx].std()
    ok_20 = "YES" if 0.95 <= std_20 <= 1.05 else "NO"
    ok_10 = "YES" if 0.95 <= std_10 <= 1.05 else "NO"
    print(f"{cn:<10} {std_20:>12.3f} {std_10:>12.3f} {ok_20:>10} {ok_10:>10}")

z_ok_20ep = all(0.95 <= preds_20[:, ch_idx].std() / gts[:, ch_idx].std() <= 1.05 for ch_idx in range(6, 9))
z_ok_10ep = all(0.95 <= preds_10[:, ch_idx].std() / gts[:, ch_idx].std() <= 1.05 for ch_idx in range(6, 9))
print("\n" + "="*60)
print(f"Z VARIANCE VALIDATED: 20ep={z_ok_20ep}, 10ep={z_ok_10ep}")
print("="*60)
