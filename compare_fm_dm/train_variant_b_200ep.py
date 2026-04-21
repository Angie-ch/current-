"""
Train Variant B with 200 epochs: both FM and DM models
z_channel_weight_override=5.0, no ZPredictor
"""
import os
import sys
import torch

os.chdir('/root/autodl-tmp/fyp_final/Ver4/compare_fm_dm')
sys.path.insert(0, '.')

from configs.config import get_config
from data.dataset import build_dataloaders
from models.unified_model import create_model
from models.trainer import UnifiedTrainer

data_cfg, model_cfg, train_cfg, infer_cfg = get_config()

# Override for Variant B: z_weight=5.0, no ZPredictor
train_cfg.max_epochs = 200
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

work_dir = './z_variant_b_200ep_result'
os.makedirs(work_dir, exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# Build dataloaders
train_loader, val_loader, test_loader, norm_mean, norm_std = build_dataloaders(data_cfg, train_cfg)
print(f"Train: {len(train_loader.dataset)}, Val: {len(val_loader.dataset)}, Test: {len(test_loader.dataset)}")

# ========== Train FM Model ==========
print("\n" + "="*60)
print("Training FM Model (200 epochs)")
print("="*60)

fm_model = create_model(model_cfg, data_cfg, train_cfg, method='fm')
fm_model = fm_model.to(device)

fm_trainer = UnifiedTrainer(fm_model, train_loader, val_loader, train_cfg, data_cfg,
                           work_dir=os.path.join(work_dir, 'checkpoints_fm'), method='fm')
fm_trainer.train()

print("\nFM Training complete!")

# ========== Train DM Model ==========
print("\n" + "="*60)
print("Training DM Model (200 epochs)")
print("="*60)

dm_model = create_model(model_cfg, data_cfg, train_cfg, method='dm')
dm_model = dm_model.to(device)

dm_trainer = UnifiedTrainer(dm_model, train_loader, val_loader, train_cfg, data_cfg,
                           work_dir=os.path.join(work_dir, 'checkpoints_dm'), method='dm')
dm_trainer.train()

print("\nDM Training complete!")
print("\n" + "="*60)
print("All training complete!")
print("="*60)
