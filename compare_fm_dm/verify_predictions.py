# ── Config: change ckpt_path and variant settings here ──
CKPT_PATH = '/root/autodl-tmp/fyp_final/Ver4/compare_fm_dm/z_variant_b_result/checkpoints_fm/best.pt'
USE_Z_PREDICTOR = False
Z_CHANNEL_WEIGHT_OVERRIDE = 5.0  # Variant B: 5x z weight

import torch, sys, os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # use GPU
os.chdir('/root/autodl-tmp/fyp_final/Ver4/compare_fm_dm')
sys.path.insert(0, '.')

from data.dataset import build_dataloaders
from models.unified_model import create_model
from configs.config import ModelConfig, DataConfig, TrainConfig
import numpy as np

model_cfg = ModelConfig()
data_cfg = DataConfig()
train_cfg = TrainConfig()

# Apply variant B settings
train_cfg.use_z_predictor = USE_Z_PREDICTOR
train_cfg.z_predictor_weight = 0.0
train_cfg.z_channel_weight_override = Z_CHANNEL_WEIGHT_OVERRIDE
if Z_CHANNEL_WEIGHT_OVERRIDE != 1.0:
    cw = list(train_cfg.channel_weights)
    for i in range(6, 9):  # z channels
        cw[i] = Z_CHANNEL_WEIGHT_OVERRIDE
    train_cfg.channel_weights = tuple(cw)

fm_model = create_model(model_cfg, data_cfg, train_cfg, method='fm').to('cuda')
ckpt = torch.load(CKPT_PATH, map_location='cuda', weights_only=False)
fm_model.load_state_dict(ckpt['model_state_dict'], strict=False)
ema = ckpt.get('ema_state_dict', {})
if ema:
    ema_shadow = ema.get('shadow', {})
    for name, param in fm_model.named_parameters():
        if name in ema_shadow:
            param.data.copy_(ema_shadow[name].to(param.device, param.dtype))
fm_model.eval()

print(f"\nLoaded: {CKPT_PATH}")
print(f"Config: use_z_predictor={USE_Z_PREDICTOR}, z_weight_override={Z_CHANNEL_WEIGHT_OVERRIDE}\n")

_, _, test_loader, norm_mean, norm_std = build_dataloaders(data_cfg, train_cfg)
test_loader = torch.utils.data.DataLoader(
    test_loader.dataset,
    batch_size=128, shuffle=False, num_workers=4, pin_memory=True
)
if isinstance(norm_mean, torch.Tensor): norm_mean = norm_mean.numpy()
if isinstance(norm_std, torch.Tensor): norm_std = norm_std.numpy()

channel_names = [f'{v}_{l}' for v in ['u', 'v', 'z'] for l in [850, 500, 250]]

# Collect ALL batches for robust stats
all_pred = []
all_tgt = []
all_cond = []
n_batches = 0
for batch in test_loader:
    cond = batch['condition'].cuda()
    target = batch['target']
    with torch.no_grad():
        pred = fm_model.sample_fm(cond, 'cuda',
            euler_steps=4, euler_mode='midpoint',
            clamp_range=None, z_clamp_range=None)
    all_pred.append(pred.cpu())
    all_tgt.append(target)
    all_cond.append(cond[:, :9].cpu())
    n_batches += 1
    if n_batches >= 20:
        break

pred_np = torch.cat(all_pred, dim=0).numpy()
target_np = torch.cat(all_tgt, dim=0).numpy()
cond_np = torch.cat(all_cond, dim=0).numpy()
print(f"Collected {n_batches} batches, {pred_np.shape[0]} total samples\n")

# --- Normalized space ---
print('=== Normalized space ===')
print(f'{"Channel":>8}  {"Pred Min":>9}  {"Pred Max":>9}  {"Pred Mean":>9}  {"Pred Std":>9}  |  {"Tgt Min":>9}  {"Tgt Max":>9}  {"Tgt Mean":>9}  {"Tgt Std":>9}')
print('-' * 115)
for ch_idx, ch_name in enumerate(channel_names):
    p = pred_np[0, ch_idx]
    t = target_np[0, ch_idx]
    print(f'{ch_name:>8}  {p.min():>9.4f}  {p.max():>9.4f}  {p.mean():>9.4f}  {p.std():>9.4f}  |  {t.min():>9.4f}  {t.max():>9.4f}  {t.mean():>9.4f}  {t.std():>9.4f}')

# --- Denormalized space ---
print()
print('=== Denormalized space ===')
pred_denorm = pred_np * norm_std[np.newaxis, :, np.newaxis, np.newaxis] + norm_mean[np.newaxis, :, np.newaxis, np.newaxis]
tgt_denorm  = target_np * norm_std[np.newaxis, :, np.newaxis, np.newaxis] + norm_mean[np.newaxis, :, np.newaxis, np.newaxis]
print(f'{"Channel":>8}  {"Pred Min":>9}  {"Pred Max":>9}  {"Pred Mean":>9}  |  {"Tgt Min":>9}  {"Tgt Max":>9}  {"Tgt Mean":>9}')
print('-' * 85)
for ch_idx, ch_name in enumerate(channel_names):
    p = pred_denorm[:, ch_idx].mean(axis=0)
    t = tgt_denorm[:, ch_idx].mean(axis=0)
    print(f'{ch_name:>8}  {p.min():>9.2f}  {p.max():>9.2f}  {p.mean():>9.2f}  |  {t.min():>9.2f}  {t.max():>9.2f}  {t.mean():>9.2f}')

# --- Correlation vs x0 (persistence) ---
print()
print(f'=== Correlation & Bias ({pred_np.shape[0]} samples) ===')
pred_flat = pred_np.reshape(pred_np.shape[0], pred_np.shape[1], -1)
tgt_flat  = target_np.reshape(target_np.shape[0], target_np.shape[1], -1)
cond_flat = cond_np.reshape(cond_np.shape[0], 9, -1)
print(f'{"Channel":>8}  {"Corr(pred,tgt)":>14}  {"Corr(x0,tgt)":>12}  {"Bias":>8}  {"Pred/Tgt Std":>12}')
print('-' * 65)
for ch_idx, ch_name in enumerate(channel_names):
    corr_pred = np.corrcoef(pred_flat[:, ch_idx].flatten(), tgt_flat[:, ch_idx].flatten())[0, 1]
    corr_x0   = np.corrcoef(cond_flat[:, ch_idx].flatten(), tgt_flat[:, ch_idx].flatten())[0, 1]
    bias      = float(pred_flat[:, ch_idx].mean() - tgt_flat[:, ch_idx].mean())
    std_ratio = float(pred_flat[:, ch_idx].std() / tgt_flat[:, ch_idx].std())
    flag = '✅' if corr_pred > corr_x0 else '⚠️'
    print(f'{ch_name:>8}  {flag}  {corr_pred:>14.4f}  {corr_x0:>12.4f}  {bias:>8.4f}  {std_ratio:>12.3f}')

# --- Spatial range sanity check ---
print()
print('=== Z-channel range sanity (raw output) ===')
for ch_idx, ch_name in zip([3, 4, 5], ['z_850', 'z_500', 'z_250']):
    vals = pred_np[:, ch_idx].flatten()
    pct_1 = (np.abs(vals) > 1.0).mean() * 100
    pct_2 = (np.abs(vals) > 2.0).mean() * 100
    pct_3 = (np.abs(vals) > 3.0).mean() * 100
    print(f'{ch_name}: |v|>1: {pct_1:5.1f}%  |v|>2: {pct_2:5.1f}%  |v|>3: {pct_3:5.1f}%  (before clamp, model output)')
