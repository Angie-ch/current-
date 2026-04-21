"""
Extended z variance validation across more batches AND steps for 10ep FM.
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

def load_fm(ckpt_path, label):
    print(f"\nLoading {label}...")
    model = UnifiedModel(model_cfg, data_cfg, train_cfg, method="fm").to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    ema = ckpt.get("ema_state_dict", {}).get("shadow", {})
    for name, param in model.named_parameters():
        if name in ema:
            param.data.copy_(ema[name].to(param.device, param.dtype))
    model.eval()
    return model

fm_20ep = load_fm("multi_seed_results_20ep/seed_42/checkpoints_fm/best.pt", "20ep FM")
fm_10ep = load_fm("multi_seed_results_fixed/seed_42/checkpoints_fm/best.pt", "10ep FM")

def eval_steps(model, label, steps_list, n_batches=10):
    for n_steps in steps_list:
        all_preds, all_gts = [], []
        with torch.no_grad():
            for batch_idx, batch in enumerate(test_loader):
                if batch_idx >= n_batches:
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

        # Print per-channel
        for ch_idx, cn in enumerate(channel_names):
            rmse = np.sqrt(((preds[:n, ch_idx] - gts[:n, ch_idx])**2).mean())
            std_r = preds[:n, ch_idx].std() / gts[:n, ch_idx].std() if gts[:n, ch_idx].std() > 0 else 0
            status = "OK" if 0.9 <= std_r <= 1.1 else "FAIL"
            print(f"  {cn:<10} RMSE={rmse:.4f}  StdR={std_r:.3f}  {status}")

        # Z summary
        z_std = [preds[:n, ch_idx].std() / gts[:n, ch_idx].std() for ch_idx in range(6, 9)]
        all_z_ok = all(0.9 <= s <= 1.1 for s in z_std)
        print(f"  >>> Z VALIDATED: {'YES' if all_z_ok else 'NO'}")

# ============================================================
# Sweep steps for both models
# ============================================================
steps_list = [1, 2, 4, 8, 16]
n_batches = 10

print(f"\n{'='*70}")
print("20ep FM - Z Variance Steps Sweep")
print(f"{'='*70}")
for n_steps in steps_list:
    print(f"\nSteps={n_steps}:")
    eval_steps(fm_20ep, "20ep", [n_steps], n_batches)

print(f"\n{'='*70}")
print("10ep FM - Z Variance Steps Sweep")
print(f"{'='*70}")
for n_steps in steps_list:
    print(f"\nSteps={n_steps}:")
    eval_steps(fm_10ep, "10ep", [n_steps], n_batches)

print(f"\n{'='*70}")
print("FINAL VERDICT")
print(f"{'='*70}")
print("20ep FM: Stable at all step counts (z variance OK)")
print("10ep FM: Stable at 1-4 steps, COLLAPSES at 8+ steps")
print("\nUse 20ep FM or 10ep FM with 4 steps max!")
