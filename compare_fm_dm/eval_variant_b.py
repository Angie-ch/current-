"""
Evaluate variant B checkpoint - per-channel metrics
"""
import os, sys, torch, numpy as np

os.chdir('/root/autodl-tmp/fyp_final/Ver4/compare_fm_dm')
sys.path.insert(0, '.')

from configs.config import get_config
from data.dataset import build_dataloaders
from models.unified_model import create_model

data_cfg, model_cfg, train_cfg, infer_cfg = get_config()

# Variant B config
train_cfg.use_z_predictor = False
train_cfg.z_predictor_weight = 0.0
train_cfg.z_channel_weight_override = 5.0
cw = list(train_cfg.channel_weights)
for i in range(6, 9):
    cw[i] = 5.0
train_cfg.channel_weights = tuple(cw)

device = torch.device('cuda')
ckpt_path = 'z_variant_b_result/checkpoints_fm/best.pt'
print(f"Loading: {ckpt_path}")

model = create_model(model_cfg, data_cfg, train_cfg, method='fm').to(device)
ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
model.load_state_dict(ckpt['model_state_dict'], strict=False)
model.eval()

# EMA
ema = ckpt.get('ema_state_dict', {})
if ema:
    ema_shadow = ema.get('shadow', {})
    for name, param in model.named_parameters():
        if name in ema_shadow:
            param.data.copy_(ema_shadow[name].to(param.device, param.dtype))

# Build test loader
_, _, test_loader, _, _ = build_dataloaders(data_cfg, train_cfg)

channel_names = [f'{v}_{l}' for v in ['u', 'v', 'z'] for l in [850, 500, 250]]

n_batches = 20
preds_batched = [[] for _ in range(9)]
gts_batched   = [[] for _ in range(9)]

print(f"Running inference on {n_batches} batches...")
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

print(f"\n{'='*60}")
print(f"VARIANT B: z_channel_weight_override=5.0, no ZPredictor")
print(f"{'='*60}")
print(f"\n{'Channel':>8}  {'RMSE':>8}  {'StdR':>8}  {'Corr':>8}  {'Bias':>8}")
print('-' * 60)
for ch, name in enumerate(channel_names):
    p_all = np.concatenate(preds_batched[ch], axis=0)
    t_all = np.concatenate(gts_batched[ch], axis=0)
    
    rmse  = float(np.sqrt(((p_all - t_all) ** 2).mean()))
    std_r = float(p_all.std() / t_all.std()) if t_all.std() > 1e-6 else 0.0
    corr  = float(np.corrcoef(p_all.flatten(), t_all.flatten())[0, 1])
    bias  = float(p_all.mean() - t_all.mean())
    
    flag = "✅" if corr > 0.3 else ("⚠️" if corr > 0.1 else "❌")
    print(f"{flag} {name:>7}  {rmse:>8.4f}  {std_r:>8.3f}  {corr:>8.4f}  {bias:>8.4f}")

print(f"\n{'='*60}")
print(f"Z-channel variance (StdR) check:")
for ch, name in enumerate(['z_850', 'z_500', 'z_250']):
    p_all = np.concatenate(preds_batched[ch + 6], axis=0)
    t_all = np.concatenate(gts_batched[ch + 6], axis=0)
    ratio = p_all.std() / t_all.std() if t_all.std() > 1e-6 else 0.0
    status = "✅ OK" if 0.3 < ratio < 2.0 else "⚠️ LOW" if ratio < 0.3 else "⚠️ HIGH"
    print(f"  {name}: pred_std={p_all.std():.4f}  tgt_std={t_all.std():.4f}  ratio={ratio:.3f}  {status}")
