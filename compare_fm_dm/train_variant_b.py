"""
Train only Variant B: z_channel_weight_override=5.0 (20 epochs quick test)
"""
import os, sys, copy, torch

os.chdir('/root/autodl-tmp/fyp_final/Ver4/compare_fm_dm')
sys.path.insert(0, '.')

from configs.config import get_config
from data.dataset import build_dataloaders
from models.unified_model import create_model
from models.trainer import UnifiedTrainer

data_cfg, model_cfg, train_cfg, infer_cfg = get_config()

# Override for Variant B: z_weight=5.0, no ZPredictor
train_cfg.max_epochs = 20
train_cfg.use_z_predictor = False
train_cfg.z_predictor_weight = 0.0
train_cfg.z_channel_weight_override = 5.0

# Apply z-weight override to channel_weights
cw = list(train_cfg.channel_weights)
print(f"Original channel_weights: {train_cfg.channel_weights}")
for i in range(6, 9):  # z channels
    cw[i] = 5.0
train_cfg.channel_weights = tuple(cw)
print(f"Modified channel_weights: {train_cfg.channel_weights}")

work_dir = './z_variant_b_result'
os.makedirs(work_dir, exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# Build dataloaders
train_loader, val_loader, test_loader, norm_mean, norm_std = build_dataloaders(data_cfg, train_cfg)
print(f"Train: {len(train_loader.dataset)}, Val: {len(val_loader.dataset)}, Test: {len(test_loader.dataset)}")

# Create FM model
model = create_model(model_cfg, data_cfg, train_cfg, method='fm')
model = model.to(device)

# Train
trainer = UnifiedTrainer(model, train_loader, val_loader, train_cfg, data_cfg,
                         work_dir=work_dir, method='fm')
trainer.train()

print("\n=== Training complete! Evaluating per-channel metrics ===")
import numpy as np

# Reload best checkpoint
ckpt = torch.load(os.path.join(work_dir, 'checkpoints_fm', 'best.pt'), 
                   map_location=device, weights_only=False)
model.load_state_dict(ckpt['model_state_dict'], strict=False)
model.eval()

channel_names = [f'{v}_{l}' for v in ['u', 'v', 'z'] for l in [850, 500, 250]]

n_batches = 10
preds_batched = [[] for _ in range(9)]
gts_batched   = [[] for _ in range(9)]

with torch.no_grad():
    for batch_idx, batch in enumerate(test_loader):
        if batch_idx >= n_batches:
            break
        cond   = batch["condition"].to(device)
        target = batch["target"]
        
        pred = model.sample_fm(cond, device,
            euler_steps=4, euler_mode='midpoint',
            clamp_range=None, z_clamp_range=None)
        
        for ch in range(9):
            preds_batched[ch].append(pred[:, ch].cpu().numpy())
            gts_batched[ch].append(target[:, ch].numpy())

print(f"\n{'Channel':>8}  {'RMSE':>8}  {'StdR':>8}  {'Corr':>8}  {'Bias':>8}")
print('-' * 55)
for ch, name in enumerate(channel_names):
    p_all = np.concatenate(preds_batched[ch], axis=0)
    t_all = np.concatenate(gts_batched[ch], axis=0)
    
    rmse  = float(np.sqrt(((p_all - t_all) ** 2).mean()))
    std_r = float(p_all.std() / t_all.std()) if t_all.std() > 1e-6 else 0.0
    corr  = float(np.corrcoef(p_all.flatten(), t_all.flatten())[0, 1])
    bias  = float(p_all.mean() - t_all.mean())
    
    flag = "✅" if corr > 0.3 else ("⚠️" if corr > 0.1 else "❌")
    print(f"{flag} {name:>7}  {rmse:>8.4f}  {std_r:>8.3f}  {corr:>8.4f}  {bias:>8.4f}")

print(f"\n=== Z-channel variance check ===")
for ch, name in enumerate(['z_850', 'z_500', 'z_250']):
    p_all = np.concatenate(preds_batched[ch + 6], axis=0)
    t_all = np.concatenate(gts_batched[ch + 6], axis=0)
    ratio = p_all.std() / t_all.std() if t_all.std() > 1e-6 else 0.0
    print(f"{name}: pred_std={p_all.std():.4f}  tgt_std={t_all.std():.4f}  std_ratio={ratio:.3f}")
