#!/bin/bash
# 运行 FM vs DM 对比实验 (使用 newtry 的 best_eps.pt)

cd /root/autodl-tmp/fyp_final/Ver4/compare_fm_dm

python run_comparison.py \
    --data_root /root/autodl-tmp/fyp_final/Ver4/Trajectory/preprocessed_era5 \
    --preprocess_dir /root/autodl-tmp/fyp_final/Ver4/Trajectory/preprocessed_era5 \
    --norm_stats /root/autodl-tmp/fyp_final/Ver4/Trajectory/preprocessed_era5/norm_stats.pt \
    --external_dm_ckpt /root/autodl-tmp/fyp_final/Ver4/newtry/checkpoints/best_eps.pt \
    --skip_train \
    --eval_samples 50 \
    --ar_steps 24 \
    --work_dir ./results_newtry_comparison \
    --output_dir ./results_newtry_comparison/figures
