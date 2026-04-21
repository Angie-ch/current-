#!/bin/bash
# Finetune trajectory predictor using compare_fm_dm FM model
# This script runs the complete pipeline:
#  1. Generate FM-ERA5 cache (if not exists)
#  2. Finetune trajectory model on FM-ERA5
#  3. Evaluate on test set

set -e  # Exit on error

# ========== Configuration ==========
# Adjust these paths to your setup

# Trajectory stage 1 checkpoint (pre-trained on real ERA5)
PRETRAINED_CKPT="/root/autodl-tmp/fyp_final/Ver4/Trajectory/checkpoints/best.pt"

# compare_fm_dm FM checkpoint (trained FM model)
COMPAREFM_CKPT="/root/autodl-tmp/fyp_final/Ver4/compare_fm_dm/multi_seed_results/seed_42/checkpoints_fm/best.pt"

# compare_fm_dm code directory
COMPAREFM_CODE="/root/autodl-tmp/fyp_final/Ver4/compare_fm_dm"

# Normalization stats (from trajectory pre-training)
NORM_STATS="/root/autodl-tmp/fyp_final/Ver4/Trajectory/preprocessed_era5/norm_stats.pt"

# ERA5 data root
DATA_ROOT="/root/autodl-tmp/fyp_final/Typhoon_data_final"

# Track CSV
TRACK_CSV="/root/autodl-tmp/fyp_final/Ver4/Trajectory/processed_typhoon_tracks.csv"

# Output directory for finetuned model
OUTPUT_DIR="/root/autodl-tmp/fyp_final/Ver4/flow_matching/checkpoints_finetune_comparefm"

# Finetuning hyperparameters
FINETUNE_EPOCHS=80
FINETUNE_LR=2e-5
BATCH_SIZE=64
FREEZE_STRATEGY="bridge"
EULER_STEPS=4

# ========== Run ==========
cd /root/autodl-tmp/fyp_final/Ver4/flow_matching

echo "=========================================="
echo "Finetune with compare_fm_dm FM model"
echo "=========================================="
echo "Pretrained ckpt:   $PRETRAINED_CKPT"
echo "CompareFM ckpt:    $COMPAREFM_CKPT"
echo "CompareFM code:    $COMPAREFM_CODE"
echo "Norm stats:        $NORM_STATS"
echo "Data root:         $DATA_ROOT"
echo "Track CSV:         $TRACK_CSV"
echo "Output dir:        $OUTPUT_DIR"
echo "Euler steps:       $EULER_STEPS"
echo "=========================================="

python finetune_train_compare.py \
    --pretrained_ckpt "$PRETRAINED_CKPT" \
    --comparefm_code "$COMPAREFM_CODE" \
    --comparefm_ckpt "$COMPAREFM_CKPT" \
    --norm_stats "$NORM_STATS" \
    --data_root "$DATA_ROOT" \
    --track_csv "$TRACK_CSV" \
    --checkpoint_dir "$OUTPUT_DIR" \
    --finetune_epochs $FINETUNE_EPOCHS \
    --finetune_lr $FINETUNE_LR \
    --batch_size $BATCH_SIZE \
    --freeze_strategy "$FREEZE_STRATEGY" \
    --euler_steps $EULER_STEPS

echo ""
echo "=========================================="
echo "Done! Model saved to: $OUTPUT_DIR/best_finetune_comparefm.pt"
echo "=========================================="
