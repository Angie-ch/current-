"""
Quick per-channel evaluation of the z_sweep A_baseline checkpoint.
"""
import torch, sys, os
import numpy as np

os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0')
os.chdir('/root/autodl-tmp/fyp_final/Ver4/compare_fm_dm')
sys.path.insert(0, '.')

from data.dataset import build_dataloaders
from models.unified_model import UnifiedModel
from configs.config import ModelConfig, DataConfig, TrainConfig

# Use sweep checkpoint
ckpt_path = 'z_sweep_result/A_baseline/A_baseline_seed42/checkpoints_fm/best.pt'
print(f"Loading checkpoint: {ckpt_path}")

model_cfg = ModelConfig()
data_cfg = DataConfig()
train_cfg = TrainConfig()

# Override for variant A
train_cfg.use_z_predictor = False
train_cfg.z_predictor_weight = 0.0
train_cfg.z_channel_weight_override = 1.0

fm_model = UnifiedModel(model_cfg, data_cfg, train_cfg, method='fm').to('cuda')
ckpt = torch.load(ckpt_path, map_location='cuda', weights_only=False)
fm_model.load_state_dict(ckpt['model_state_dict'], strict=False)

ema = ckpt.get('ema_state_dict', {})
if ema:
    ema_shadow = ema.get('shadow', {})
    for name, param in fm_model.named_parameters():
        if name in ema_shadow:
            param.data.copy_(ema_shadow[name].to(param.device, param.dtype))

fm_model.eval()

# Build test loader
_, _, test_loader, norm_mean, norm_std = build_dataloaders(data_cfg, train_cfg)

channel_names = [f'{v}_{l}' for v in ['u', 'v', 'z'] for l in [850, 500, 250]]

# Collect predictions
n_batches = 20
preds_batched = [[] for _ in range(9)]
gts_batched   = [[] for _ in range(9)]

with torch.no_grad():
    for batch_idx, batch in enumerate(test_loader):
        if batch_idx >= n_batches:
            break
        cond   = batch["condition"].to('cuda')
        target = batch["target"].to('cuda')
        
        pred = fm_model.sample_fm(cond, 'cuda',
            euler_steps=4, euler_mode='midpoint',
            clamp_range=None, z_clamp_range=None)
        
        for ch in range(9):
            preds_batched[ch].append(pred[:, ch].cpu().numpy())
            gts_batched[ch].append(target[:, ch].cpu().numpy())

print(f"\n=== Per-Channel Metrics ({n_batches} batches) ===")
print(f"{'Channel':>8}  {'RMSE':>8}  {'StdR':>8}  {'Corr':>8}  {'Bias':>8}")
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

print(f"\n=== Z-channel statistics ===")
for ch, name in enumerate(['z_850', 'z_500', 'z_250']):
    p_all = np.concatenate(preds_batched[ch + 6], axis=0)
    t_all = np.concatenate(gts_batched[ch + 6], axis=0)
    print(f"{name}: pred_std={p_all.std():.4f}  tgt_std={t_all.std():.4f}  ratio={p_all.std()/t_all.std():.3f}")
