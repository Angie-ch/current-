"""
自定义评估脚本：使用 era5_cache.npz 和 best_finetune.pt 评估 2019-2021 年
生成 Table 2 (ensemble mean) 和 Table 3 (oracle best)
"""
import os
import sys
import argparse
import json
import numpy as np
import torch
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from datetime import datetime

# Add Trajectory dir to path
TRAJ_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, TRAJ_DIR)

from config import model_cfg, data_cfg, train_cfg
from model import LT3PModel
from dataset import normalize_coords, normalize_era5, ERA5_CHANNEL_MEAN, ERA5_CHANNEL_STD
from data_processing import load_typhoon_csv, load_single_tyc_storm
from paper_eval_common import (
    build_report_indices, apply_report_view, aggregate_table2_predictions,
    aggregate_table3_predictions, select_typhoon_ids, _extract_year,
    traj_data_cfg, traj_model_cfg
)


def load_trajectory_model(checkpoint_path, era5_channels, device, bias_path=None):
    """Load the finetuned trajectory model."""
    model = LT3PModel(
        coord_dim=model_cfg.coord_dim,
        output_dim=model_cfg.output_dim,
        era5_channels=era5_channels,
        t_history=model_cfg.t_history,
        t_future=model_cfg.t_future,
        d_model=model_cfg.transformer_dim,
        n_heads=model_cfg.transformer_heads,
        n_layers=model_cfg.transformer_layers,
        ff_dim=model_cfg.transformer_ff_dim,
        dropout=model_cfg.dropout,
    )
    
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_key = 'ema_model_state_dict' if 'ema_model_state_dict' in ckpt else 'model_state_dict'
    state_dict = ckpt[state_key]
    
    # Check for adapter and load accordingly
    has_conv_adapter = any(k.startswith('adapter.') for k in state_dict.keys())
    if has_conv_adapter:
        from finetune_train import ERA5ConvAdaptedModel
        adapted_model = ERA5ConvAdaptedModel(model, era5_channels=era5_channels)
        print("  检测到 Conv 适配器")
    else:
        from finetune_train import ERA5AdaptedModel
        adapted_model = ERA5AdaptedModel(model, era5_channels=era5_channels)
        print("  检测到 Affine 适配器")
    
    adapted_model.load_state_dict(state_dict)
    adapted_model.to(device)
    adapted_model.eval()
    print(f"  已加载微调模型 (epoch {ckpt.get('epoch', '?')})")
    return adapted_model


def run_table2_evaluation(args):
    """Run Table 2 evaluation (ensemble mean)."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"设备: {device}")
    
    # Load era5_cache
    print(f"\n[1] 加载扩散ERA5缓存: {args.era5_cache}")
    cache = np.load(args.era5_cache, allow_pickle=True)
    diffusion_cache = {k: cache[k] for k in cache.files}
    print(f"  缓存台风数: {len(diffusion_cache)}")
    
    # Load trajectory model
    print(f"\n[2] 加载轨迹模型: {args.trajectory_ckpt}")
    era5_channels = 9  # Both models use 9 channels
    traj_model = load_trajectory_model(
        checkpoint_path=args.trajectory_ckpt,
        era5_channels=era5_channels,
        device=device,
        bias_path=args.bias_path,
    )
    
    # Load track CSV and filter by year
    print(f"\n[3] 加载轨迹数据: {args.track_csv}")
    track_df = load_typhoon_csv(args.track_csv)
    # Get unique storm IDs for 2019-2021
    track_df['year'] = track_df['typhoon_id'].str[:4].astype(int)
    year_filtered = track_df[track_df['year'].between(2019, 2021)]
    storm_ids = sorted(year_filtered['storm_id'].unique().tolist())
    print(f"  {len(storm_ids)} 个台风 (2019-2021)")
    
    # Apply exclude list
    if args.exclude_file and os.path.exists(args.exclude_file):
        with open(args.exclude_file, 'r') as f:
            exclude_ids = set(line.strip() for line in f if line.strip())
        storm_ids = [s for s in storm_ids if s not in exclude_ids]
        print(f"  排除后剩余: {len(storm_ids)} 个台风")
    
    # Build report indices
    report_indices = build_report_indices(
        total_steps=traj_model_cfg.t_future,
        base_resolution_hours=traj_data_cfg.time_resolution_hours,
        report_every_hours=args.report_every_hours,
    )
    
    # Evaluate each storm
    results = []
    lat_range = traj_data_cfg.lat_range
    lon_range = traj_data_cfg.lon_range
    
    print(f"\n[4] 开始 Table 2 评估 (ensemble mean)")
    for count, storm_id in enumerate(tqdm(storm_ids, desc="台风"), 1):
        # Check if storm is in cache
        if storm_id not in diffusion_cache:
            print(f"  跳过 {storm_id}: 不在缓存中")
            continue
        
        # Get track data
        track_df_storm = track_df[track_df['storm_id'] == storm_id].copy()
        if len(track_df_storm) < traj_model_cfg.t_history + traj_model_cfg.t_future:
            print(f"  跳过 {storm_id}: 轨迹数据不足 ({len(track_df_storm)} < {traj_model_cfg.t_history + traj_model_cfg.t_future})")
            continue
        
        # Find forecast start index (need at least t_history before)
        forecast_start_idx = None
        for idx in range(traj_model_cfg.t_history, len(track_df_storm) - traj_model_cfg.t_future):
            forecast_start_idx = idx
            break
        if forecast_start_idx is None:
            print(f"  跳过 {storm_id}: 无法找到有效的起报点")
            continue
        
        # Get history and ground truth
        history_lat = track_df_storm['lat'].values[forecast_start_idx - traj_model_cfg.t_history:forecast_start_idx].astype(np.float32)
        history_lon = track_df_storm['lon'].values[forecast_start_idx - traj_model_cfg.t_history:forecast_start_idx].astype(np.float32)
        gt_lat = track_df_storm['lat'].values[forecast_start_idx:forecast_start_idx + traj_model_cfg.t_future].astype(np.float32)
        gt_lon = track_df_storm['lon'].values[forecast_start_idx:forecast_start_idx + traj_model_cfg.t_future].astype(np.float32)
        
        # Build history tensor
        h_lat_n, h_lon_n = normalize_coords(history_lat, history_lon)
        history_coords_t = torch.from_numpy(np.stack([h_lat_n, h_lon_n], axis=-1)).float().unsqueeze(0).to(device)
        
        # Get ERA5 from cache (shape: T_era5, 9, 40, 40)
        # Cache covers full storm - need to extract the forecast window
        era5_full = diffusion_cache[storm_id]  # (T_storm, 9, 40, 40)
        
        # The cache should align with track times. Need to find correct offset.
        # Assuming cache is aligned with track data (1:1 time mapping)
        # The forecast starts at forecast_start_idx, we need t_future steps
        if era5_full.shape[0] < forecast_start_idx + traj_model_cfg.t_future:
            print(f"  跳过 {storm_id}: ERA5缓存长度不足 ({era5_full.shape[0]} < {forecast_start_idx + traj_model_cfg.t_future})")
            continue
            
        future_era5 = era5_full[forecast_start_idx:forecast_start_idx + traj_model_cfg.t_future]
        future_era5_norm = normalize_era5(future_era5)
        future_era5_t = torch.from_numpy(future_era5_norm).float().unsqueeze(0).to(device)  # (1, T_future, C, H, W)
        
        # Generate N samples
        sample_preds = []
        with torch.no_grad():
            for _ in range(args.num_samples):
                pred = traj_model(history_coords_t, future_era5_t)  # (1, T_future, 2)
                pred_np = pred.squeeze(0).cpu().numpy()  # (T_future, 2)
                # Denormalize coordinates
                pred_lat = pred_np[:, 0] * (lat_range[1] - lat_range[0]) + lat_range[0]
                pred_lon = pred_np[:, 1] * (lon_range[1] - lon_range[0]) + lon_range[0]
                sample_preds.append(np.stack([pred_lat, pred_lon], axis=-1))  # (T_future, 2)
        
        # Aggregate: ensemble mean
        ensemble_mean = np.mean(sample_preds, axis=0)  # (T_future, 2)
        
        # Compute error
        error_km = np.sqrt((ensemble_mean[:, 0] - gt_lat)**2 + (ensemble_mean[:, 1] - gt_lon)**2) * 111  # approx km per degree
        
        # Build result dict
        result = {
            'error_km': error_km,
            'report_hours': np.arange(traj_model_cfg.t_future) * 3,  # Every 3 hours
            'sample_count': args.num_samples,
            'report_every_hours': args.report_every_hours,
            'storm_id': storm_id,
            'case_key': f"{storm_id}::forecast_start={forecast_start_idx}",
        }
        results.append(result)
    
    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Compile summary
    all_errors = np.array([r['error_km'] for r in results])
    report_indices = build_report_indices(
        total_steps=traj_model_cfg.t_future,
        base_resolution_hours=traj_data_cfg.time_resolution_hours,
        report_every_hours=args.report_every_hours,
    )
    
    summary = {
        "generated_at": datetime.now().isoformat(),
        "method": "DM_ensemble_mean",
        "num_storms": len(results),
        "num_samples_per_case": args.num_samples,
        "report_every_hours": args.report_every_hours,
        "error_by_hour": {},
    }
    
    for idx, hours in enumerate(results[0]['report_hours']):
        if hours % args.report_every_hours == 0 or hours == results[0]['report_hours'][-1]:
            hour_key = f"{hours}h"
            mean_err = float(all_errors[:, idx].mean())
            std_err = float(all_errors[:, idx].std())
            summary["error_by_hour"][hour_key] = {
                "mean_km": mean_err,
                "std_km": std_err,
            }
    
    summary_path = os.path.join(args.output_dir, "table2_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n[完成] 结果保存至: {args.output_dir}")
    print(f"  台风数: {len(results)}")
    print(f"  平均误差:")
    for hour_key, vals in summary["error_by_hour"].items():
        print(f"    {hour_key}: {vals['mean_km']:.2f} ± {vals['std_km']:.2f} km")
    
    return results


def run_table3_evaluation(args):
    """Run Table 3 evaluation (oracle best)."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"设备: {device}")
    
    # Load era5_cache
    print(f"\n[1] 加载扩散ERA5缓存: {args.era5_cache}")
    cache = np.load(args.era5_cache, allow_pickle=True)
    diffusion_cache = {k: cache[k] for k in cache.files}
    print(f"  缓存台风数: {len(diffusion_cache)}")
    
    # Load trajectory model
    print(f"\n[2] 加载轨迹模型: {args.trajectory_ckpt}")
    era5_channels = 9
    traj_model = load_trajectory_model(
        checkpoint_path=args.trajectory_ckpt,
        era5_channels=era5_channels,
        device=device,
        bias_path=args.bias_path,
    )
    
    # Load track CSV and filter by year
    print(f"\n[3] 加载轨迹数据: {args.track_csv}")
    track_df = load_typhoon_csv(args.track_csv)
    track_df['year'] = track_df['typhoon_id'].str[:4].astype(int)
    year_filtered = track_df[track_df['year'].between(2019, 2021)]
    storm_ids = sorted(year_filtered['storm_id'].unique().tolist())
    print(f"  {len(storm_ids)} 个台风 (2019-2021)")
    
    # Apply exclude list
    if args.exclude_file and os.path.exists(args.exclude_file):
        with open(args.exclude_file, 'r') as f:
            exclude_ids = set(line.strip() for line in f if line.strip())
        storm_ids = [s for s in storm_ids if s not in exclude_ids]
        print(f"  排除后剩余: {len(storm_ids)} 个台风")
    
    report_indices = build_report_indices(
        total_steps=traj_model_cfg.t_future,
        base_resolution_hours=traj_data_cfg.time_resolution_hours,
        report_every_hours=args.report_every_hours,
    )
    
    lat_range = traj_data_cfg.lat_range
    lon_range = traj_data_cfg.lon_range
    
    results = []
    print(f"\n[4] 开始 Table 3 评估 (oracle best)")
    
    for count, storm_id in enumerate(tqdm(storm_ids, desc="台风"), 1):
        if storm_id not in diffusion_cache:
            print(f"  跳过 {storm_id}: 不在缓存中")
            continue
        
        track_df_storm = track_df[track_df['storm_id'] == storm_id].copy()
        if len(track_df_storm) < traj_model_cfg.t_history + traj_model_cfg.t_future:
            print(f"  跳过 {storm_id}: 轨迹数据不足")
            continue
        
        forecast_start_idx = None
        for idx in range(traj_model_cfg.t_history, len(track_df_storm) - traj_model_cfg.t_future):
            forecast_start_idx = idx
            break
        if forecast_start_idx is None:
            print(f"  跳过 {storm_id}: 无法找到有效的起报点")
            continue
        
        history_lat = track_df_storm['lat'].values[forecast_start_idx - traj_model_cfg.t_history:forecast_start_idx].astype(np.float32)
        history_lon = track_df_storm['lon'].values[forecast_start_idx - traj_model_cfg.t_history:forecast_start_idx].astype(np.float32)
        gt_lat = track_df_storm['lat'].values[forecast_start_idx:forecast_start_idx + traj_model_cfg.t_future].astype(np.float32)
        gt_lon = track_df_storm['lon'].values[forecast_start_idx:forecast_start_idx + traj_model_cfg.t_future].astype(np.float32)
        
        h_lat_n, h_lon_n = normalize_coords(history_lat, history_lon)
        history_coords_t = torch.from_numpy(np.stack([h_lat_n, h_lon_n], axis=-1)).float().unsqueeze(0).to(device)
        
        era5_full = diffusion_cache[storm_id]
        if era5_full.shape[0] < forecast_start_idx + traj_model_cfg.t_future:
            print(f"  跳过 {storm_id}: ERA5缓存长度不足")
            continue
            
        future_era5 = era5_full[forecast_start_idx:forecast_start_idx + traj_model_cfg.t_future]
        future_era5_norm = normalize_era5(future_era5)
        future_era5_t = torch.from_numpy(future_era5_norm).float().unsqueeze(0).to(device)
        
        # Generate N samples
        sample_preds = []
        with torch.no_grad():
            for _ in range(args.num_samples):
                pred = traj_model(history_coords_t, future_era5_t)
                pred_np = pred.squeeze(0).cpu().numpy()
                pred_lat = pred_np[:, 0] * (lat_range[1] - lat_range[0]) + lat_range[0]
                pred_lon = pred_np[:, 1] * (lon_range[1] - lon_range[0]) + lon_range[0]
                sample_preds.append(np.stack([pred_lat, pred_lon], axis=-1))
        
        # Oracle best: per_lead_min (minimum error at each lead time)
        best_preds = aggregate_table3_predictions(sample_preds, gt_lat, gt_lon, strategy='per_lead_min')
        
        error_km = np.sqrt((best_preds[:, 0] - gt_lat)**2 + (best_preds[:, 1] - gt_lon)**2) * 111
        
        result = {
            'error_km': error_km,
            'report_hours': np.arange(traj_model_cfg.t_future) * 3,
            'sample_count': args.num_samples,
            'report_every_hours': args.report_every_hours,
            'storm_id': storm_id,
            'case_key': f"{storm_id}::forecast_start={forecast_start_idx}",
        }
        results.append(result)
    
    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    all_errors = np.array([r['error_km'] for r in results])
    
    summary = {
        "generated_at": datetime.now().isoformat(),
        "method": "DM_oracle_best",
        "num_storms": len(results),
        "num_samples_per_case": args.num_samples,
        "report_every_hours": args.report_every_hours,
        "error_by_hour": {},
    }
    
    for idx, hours in enumerate(results[0]['report_hours']):
        if hours % args.report_every_hours == 0 or hours == results[0]['report_hours'][-1]:
            hour_key = f"{hours}h"
            mean_err = float(all_errors[:, idx].mean())
            std_err = float(all_errors[:, idx].std())
            summary["error_by_hour"][hour_key] = {
                "mean_km": mean_err,
                "std_km": std_err,
            }
    
    summary_path = os.path.join(args.output_dir, "table3_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n[完成] 结果保存至: {args.output_dir}")
    print(f"  台风数: {len(results)}")
    print(f"  平均误差:")
    for hour_key, vals in summary["error_by_hour"].items():
        print(f"    {hour_key}: {vals['mean_km']:.2f} ± {vals['std_km']:.2f} km")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="自定义评估: 使用era5_cache评估2019-2021")
    parser.add_argument("--mode", choices=["table2", "table3", "both"], required=True,
                        help="评估模式: table2(ensemble mean), table3(oracle best), both(两者)")
    parser.add_argument("--era5_cache", type=str, required=True,
                        help="扩散ERA5缓存文件 (.npz)")
    parser.add_argument("--trajectory_ckpt", type=str, required=True,
                        help="轨迹模型 checkpoint")
    parser.add_argument("--norm_stats", type=str, required=True,
                       help="归一化统计文件")
    parser.add_argument("--track_csv", type=str, required=True,
                       help="台风轨迹 CSV")
    parser.add_argument("--output_dir", type=str, required=True,
                       help="结果输出目录")
    parser.add_argument("--bias_path", type=str, default=None,
                       help="可选的偏差校正文件")
    parser.add_argument("--exclude_file", type=str, default=None,
                       help="排除台风列表文件")
    parser.add_argument("--num_samples", type=int, default=20,
                       help="每个case采样数")
    parser.add_argument("--report_every_hours", type=int, default=6,
                       help="报告间隔 (小时)")
    parser.add_argument("--min_year", type=int, default=2019,
                       help="最小年份")
    parser.add_argument("--max_year", type=int, default=2021,
                       help="最大年份")
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.mode in ["table2", "both"]:
        print("="*70)
        print("TABLE 2 评估: Ensemble Mean")
        print("="*70)
        run_table2_evaluation(args)
    
    if args.mode in ["table3", "both"]:
        print("\n" + "="*70)
        print("TABLE 3 评估: Oracle Best")
        print("="*70)
        run_table3_evaluation(args)


if __name__ == "__main__":
    main()
