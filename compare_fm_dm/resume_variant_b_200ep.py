"""
Resume FM training from epoch 80 with the latest checkpoint.
Properly restores: model, optimizer, EMA, LR scheduler, epoch, patience.
"""
import os
import sys
import torch
import logging

os.chdir('/root/autodl-tmp/fyp_final/Ver4/compare_fm_dm')
sys.path.insert(0, '.')

from configs.config import get_config
from data.dataset import build_dataloaders
from models.unified_model import create_model
from models.trainer import UnifiedTrainer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

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

# ========== Resume FM Model ==========
print("\n" + "="*60)
print("Resuming FM Model from epoch 80 (200 epochs)")
print("="*60)

fm_model = create_model(model_cfg, data_cfg, train_cfg, method='fm')
fm_model = fm_model.to(device)

fm_trainer = UnifiedTrainer(fm_model, train_loader, val_loader, train_cfg, data_cfg,
                            work_dir=os.path.join(work_dir, 'checkpoints_fm'), method='fm')

# Load checkpoint and restore all state
fm_ckpt_path = os.path.join(work_dir, 'checkpoints_fm', 'checkpoints_fm', 'latest.pt')
logger.info(f"Loading checkpoint: {fm_ckpt_path}")
ckpt = torch.load(fm_ckpt_path, map_location=device, weights_only=False)

fm_trainer.model.load_state_dict(ckpt["model_state_dict"], strict=False)
fm_trainer.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
fm_trainer.ema.load_state_dict(ckpt["ema_state_dict"])
fm_trainer.lr_scheduler.load_state_dict(ckpt["lr_scheduler_state_dict"])
fm_trainer.epoch = ckpt["epoch"] + 1
fm_trainer.global_step = ckpt["global_step"]
fm_trainer.optim_step = ckpt["optim_step"]
fm_trainer.best_val_loss = ckpt["best_val_loss"]
fm_trainer.patience_counter = ckpt["patience_counter"]

logger.info(f"  Resumed from epoch {fm_trainer.epoch}/{train_cfg.max_epochs}")
logger.info(f"  best_val_loss: {fm_trainer.best_val_loss:.6f}")
logger.info(f"  patience_counter: {fm_trainer.patience_counter}")
logger.info(f"  current lr: {fm_trainer.optimizer.param_groups[0]['lr']:.2e}")

# Start training
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
