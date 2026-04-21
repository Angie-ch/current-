"""
论文 Table 3 风格评估脚本。

评估定义：
- 对同一 case 生成 N 条轨迹预测（默认 20）。
- 按 oracle best-of-N 规则汇总。

默认策略：
- per_lead_min: 对每个 lead time 取 N 条预测中的最小误差。
  这更接近论文表格逐时效 FDE 的展示方式，但它是 oracle 上界。

可选策略：
- best_72h_traj: 选择 72h 误差最低的一整条轨迹，并返回该轨迹所有时效误差。
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch

from paper_eval_common import (
    add_common_arguments,
    aggregate_table3_predictions,
    apply_report_view,
    build_report_indices,
    collect_saved_prediction_samples,
    heuristic_forecast_start_idx,
    infer_end_to_end_era5_channels,
    infer_era5_channels_from_saved_entries,
    load_diffusion_runtime,
    load_track_data,
    load_trajectory_model,
    prepare_future_era5_for_traj,
    print_summary,
    resolve_case_key_for_saved,
    resolve_track_csv,
    run_trajectory_prediction_once,
    save_results,
    select_typhoon_ids,
    _extract_year,
    traj_data_cfg,
    traj_model_cfg,
    validate_args,
)
from table2_mode import build_history_tensor, select_matching_sample_idx


def run_end_to_end(args) -> List[Dict[str, np.ndarray]]:
    # Set random seeds for reproducible diffusion sampling
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    diff_data_cfg, diff_infer_cfg, predictor, ERA5TyphoonDataset, split_typhoon_ids = load_diffusion_runtime(args, device)

    # Load ALL typhoon IDs from track CSV directly (bypass test split)
    track_csv_path = resolve_track_csv(args.track_csv)
    track_df_full = pd.read_csv(track_csv_path)
    all_typhoon_ids = sorted(track_df_full['typhoon_id'].unique().tolist())
    
    print(f"[数据加载] CSV中共有 {len(all_typhoon_ids)} 个台风")
    
    # Filter by min_year if specified
    if args.min_year is not None:
        all_typhoon_ids = [tid for tid in all_typhoon_ids if _extract_year(tid) is not None and _extract_year(tid) >= args.min_year]
        print(f"[年份过滤] min_year={args.min_year}: 剩余 {len(all_typhoon_ids)} 个台风")

    # Filter to only typhoons with valid data (sufficient length + within bounds)
    # Fast approach: compute bounds via groupby, then only load full data for candidates
    lat_range = traj_data_cfg.lat_range
    lon_range = traj_data_cfg.lon_range
    
    grouped = track_df_full.groupby('typhoon_id').agg({
        'lat': ['min', 'max', 'count'],
        'lon': ['min', 'max']
    })
    grouped.columns = ['lat_min', 'lat_max', 'lat_count', 'lon_min', 'lon_max']
    
    valid_typhoon_ids = []
    skipped_insufficient = []
    skipped_bounds = []
    skipped_invalid_start = []
    
    for storm_id in all_typhoon_ids:
        if storm_id not in grouped.index:
            continue
        row = grouped.loc[storm_id]
        # Convert lon to 0-360 convention
        lon_min = row['lon_min']
        lon_max = row['lon_max']
        if lon_min < 0:  # Data uses -180 to 180 convention
            lon_min = lon_min % 360
            lon_max = row['lon_max'] % 360
        lat_min = row['lat_min']
        lat_max = row['lat_max']
        # Bounds check
        lat_ok = (lat_min >= lat_range[0]) and (lat_max <= lat_range[1])
        lon_ok = (lon_min >= lon_range[0]) and (lon_max <= lon_range[1])
        if not (lat_ok and lon_ok):
            skipped_bounds.append(storm_id)
            continue
        # Length check
        if row['lat_count'] < traj_model_cfg.t_history + traj_model_cfg.t_future:
            skipped_insufficient.append(storm_id)
            continue
        # Load full track for forecast_start_idx check
        track_df = load_track_data(track_csv_path, storm_id)
        if track_df is None:
            continue
        forecast_start_idx = heuristic_forecast_start_idx(track_df, traj_model_cfg.t_history, traj_model_cfg.t_future)
        if forecast_start_idx < traj_model_cfg.t_history:
            skipped_invalid_start.append(storm_id)
            continue
        valid_typhoon_ids.append(storm_id)
    
    print(f"[数据过滤] 有效台风: {len(valid_typhoon_ids)} 个")
    if skipped_insufficient:
        print(f"  跳过 (数据不足): {len(skipped_insufficient)} 个 - {skipped_insufficient[:5]}{'...' if len(skipped_insufficient)>5 else ''}")
    if skipped_bounds:
        print(f"  跳过 (超出范围): {len(skipped_bounds)} 个 - {skipped_bounds[:5]}{'...' if len(skipped_bounds)>5 else ''}")
    if skipped_invalid_start:
        print(f"  跳过 (起报点无效): {len(skipped_invalid_start)} 个")
    
    # Limit to num_typhoons if specified
    if args.num_typhoons and args.num_typhoons > 0:
        valid_typhoon_ids = valid_typhoon_ids[:args.num_typhoons]
        print(f"[限制] 仅评估前 {args.num_typhoons} 个台风")

    stats = torch.load(args.norm_stats, weights_only=True, map_location="cpu")
    test_dataset = ERA5TyphoonDataset(
        typhoon_ids=valid_typhoon_ids,
        data_root=diff_data_cfg.data_root,
        pl_vars=diff_data_cfg.pressure_level_vars,
        sfc_vars=diff_data_cfg.surface_vars,
        pressure_levels=diff_data_cfg.pressure_levels,
        history_steps=diff_data_cfg.history_steps,
        forecast_steps=diff_data_cfg.forecast_steps,
        norm_mean=stats["mean"].numpy(),
        norm_std=stats["std"].numpy(),
        preprocessed_dir=args.preprocess_dir,
    )

    era5_channels = infer_end_to_end_era5_channels(diff_data_cfg)
    traj_model = load_trajectory_model(
        checkpoint_path=args.trajectory_ckpt,
        era5_channels=era5_channels,
        device=device,
        bias_path=args.bias_path,
    )

    typhoon_samples: Dict[str, List[int]] = {}
    for idx in range(len(test_dataset)):
        storm_id = test_dataset[idx]["typhoon_id"]
        typhoon_samples.setdefault(storm_id, []).append(idx)

    selected_typhoons = valid_typhoon_ids  # Already filtered above

    # 排除指定台风
    if args.exclude_file and os.path.exists(args.exclude_file):
        with open(args.exclude_file, 'r') as f:
            exclude_ids = set(line.strip() for line in f if line.strip())
        before = len(selected_typhoons)
        selected_typhoons = [t for t in selected_typhoons if t not in exclude_ids]
        print(f"[排除] 已排除 {before - len(selected_typhoons)} 个台风 (来自 {args.exclude_file}), 剩余 {len(selected_typhoons)}")

    track_csv_path = resolve_track_csv(args.track_csv)
    report_indices = build_report_indices(
        total_steps=traj_model_cfg.t_future,
        base_resolution_hours=traj_data_cfg.time_resolution_hours,
        report_every_hours=args.report_every_hours,
    )

    results: List[Dict[str, np.ndarray]] = []
    lat_range = traj_data_cfg.lat_range
    lon_range = traj_data_cfg.lon_range

    print(
        f"[Table 3] 开始 end_to_end 评估，台风数: {len(selected_typhoons)}，"
        f"每个 case 采样: {args.num_samples}，策略: {args.selection_strategy}"
    )
    for count, storm_id in enumerate(selected_typhoons, 1):
        print(f"\n[{count}/{len(selected_typhoons)}] 台风 {storm_id}")
        track_df = load_track_data(track_csv_path, storm_id)
        if track_df is None or len(track_df) < traj_model_cfg.t_history + traj_model_cfg.t_future:
            print("  跳过：轨迹数据不足")
            continue

        forecast_start_idx = heuristic_forecast_start_idx(track_df, traj_model_cfg.t_history, traj_model_cfg.t_future)
        if forecast_start_idx < traj_model_cfg.t_history:
            print("  跳过：起报点无效")
            continue

        best_sample_idx = select_matching_sample_idx(
            available_samples=typhoon_samples[storm_id],
            forecast_start_idx=forecast_start_idx,
            diff_history_steps=diff_data_cfg.history_steps,
        )
        sample = test_dataset[best_sample_idx]
        print(f"  使用 ERA5 sample index: {best_sample_idx}")

        history_lat = track_df["lat"].values[forecast_start_idx - traj_model_cfg.t_history : forecast_start_idx].astype(np.float32)
        history_lon = track_df["lon"].values[forecast_start_idx - traj_model_cfg.t_history : forecast_start_idx].astype(np.float32)
        if len(history_lat) != traj_model_cfg.t_history:
            print("  跳过：历史窗口不足")
            continue
        history_coords_t = build_history_tensor(history_lat, history_lon, device)

        gt_lat = track_df["lat"].values[forecast_start_idx : forecast_start_idx + traj_model_cfg.t_future].astype(np.float32)
        gt_lon = track_df["lon"].values[forecast_start_idx : forecast_start_idx + traj_model_cfg.t_future].astype(np.float32)
        gt_lat = gt_lat[: traj_model_cfg.t_future]
        gt_lon = gt_lon[: traj_model_cfg.t_future]

        cond = sample["condition"].unsqueeze(0).to(device)
        sample_results = []

        # 批量采样
        sample_batch = min(args.num_samples, getattr(args, 'sample_batch_size', 4))
        num_done = 0
        while num_done < args.num_samples:
            n_this_batch = min(sample_batch, args.num_samples - num_done)
            cond_batched = cond.repeat(n_this_batch, 1, 1, 1)

            with torch.no_grad():
                preds = predictor.predict_autoregressive(
                    cond_batched,
                    num_steps=traj_model_cfg.t_future,
                    noise_sigma=diff_infer_cfg.autoregressive_noise_sigma,
                )

            for b in range(n_this_batch):
                preds_b = [p[b:b+1] for p in preds]
                future_era5 = prepare_future_era5_for_traj(
                    torch.stack(preds_b, dim=1),
                    t_future=traj_model_cfg.t_future,
                    era5_channels=era5_channels,
                    device=device,
                )
                sample_results.append(
                    run_trajectory_prediction_once(
                        traj_model=traj_model,
                        history_coords_t=history_coords_t,
                        future_era5_for_traj=future_era5,
                        gt_lat=gt_lat,
                        gt_lon=gt_lon,
                        lat_range=lat_range,
                        lon_range=lon_range,
                    )
                )

            num_done += n_this_batch
            print(f"  采样 {num_done:02d}/{args.num_samples} 完成 (batch={n_this_batch})")

        aggregated = aggregate_table3_predictions(sample_results, selection_strategy=args.selection_strategy)
        aggregated = apply_report_view(aggregated, report_indices, args.report_every_hours)
        
        # Filter by min_lead_time_hours if specified
        if args.min_lead_time_hours > 0:
            mask = aggregated["report_hours"] >= args.min_lead_time_hours
            aggregated["error_km"] = aggregated["error_km"][mask]
            aggregated["report_hours"] = aggregated["report_hours"][mask]
            aggregated["report_every_hours"] = args.report_every_hours  # Keep original reporting interval
        
        aggregated["storm_id"] = storm_id
        aggregated["case_key"] = f"{storm_id}::forecast_start={forecast_start_idx}"
        results.append(aggregated)
        print(
            f"  [Table 3] mean={aggregated['error_km'].mean():.2f}km | "
            f"+72h={aggregated['error_km'][-1]:.2f}km"
        )

    return results


def run_from_saved(args) -> List[Dict[str, np.ndarray]]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ar_files = sorted(Path(args.diffusion_output_dir).glob("ar_pred_*.pt"))
    if not ar_files:
        raise FileNotFoundError(f"未在 {args.diffusion_output_dir} 找到 ar_pred_*.pt")

    grouped_entries: Dict[str, List[Tuple[Path, dict]]] = {}
    fallback_case_count = 0
    for pt_file in ar_files:
        data = torch.load(pt_file, map_location="cpu", weights_only=False)
        case_key, _, fallback_only_typhoon = resolve_case_key_for_saved(pt_file, data)
        grouped_entries.setdefault(case_key, []).append((pt_file, data))
        if fallback_only_typhoon:
            fallback_case_count += 1

    if fallback_case_count:
        print(
            "[警告] 保存样本缺少 forecast_start_idx/case_id 等字段；"
            "from_saved 将按 typhoon_id 分组。若同一台风含多个起报点文件，请先补元数据。"
        )

    first_entries = next(iter(grouped_entries.values()))
    era5_channels = infer_era5_channels_from_saved_entries(first_entries)
    traj_model = load_trajectory_model(
        checkpoint_path=args.trajectory_ckpt,
        era5_channels=era5_channels,
        device=device,
        bias_path=args.bias_path,
    )

    selected_case_keys = sorted(grouped_entries.keys())
    if args.target_typhoon_ids:
        target_set = set(args.target_typhoon_ids)
        selected_case_keys = [
            case_key for case_key in selected_case_keys
            if any(str(data.get("typhoon_id") or data.get("storm_id") or "") in target_set for _, data in grouped_entries[case_key])
        ]

    # 年份过滤 (from_saved)
    if args.min_year is not None:
        before = len(selected_case_keys)
        filtered = []
        for case_key in selected_case_keys:
            entries = grouped_entries[case_key]
            storm_id = str(entries[0][1].get("typhoon_id") or entries[0][1].get("storm_id") or case_key)
            year = _extract_year(storm_id)
            if year is not None and year >= args.min_year:
                filtered.append(case_key)
        selected_case_keys = filtered
        print(f"[年份过滤] min_year={args.min_year}: {before} -> {len(selected_case_keys)} 个 case (排除 {before - len(selected_case_keys)} 个)")

    track_csv_path = resolve_track_csv(args.track_csv)
    report_indices = build_report_indices(
        total_steps=traj_model_cfg.t_future,
        base_resolution_hours=traj_data_cfg.time_resolution_hours,
        report_every_hours=args.report_every_hours,
    )
    results: List[Dict[str, np.ndarray]] = []
    lat_range = traj_data_cfg.lat_range
    lon_range = traj_data_cfg.lon_range

    print(
        f"[Table 3] 开始 from_saved 评估，case 数: {len(selected_case_keys)}，"
        f"每个 case 采样: {args.num_samples}，策略: {args.selection_strategy}"
    )
    for count, case_key in enumerate(selected_case_keys, 1):
        entries = grouped_entries[case_key]
        first_data = entries[0][1]
        storm_id = str(first_data.get("typhoon_id") or first_data.get("storm_id") or case_key)
        print(f"\n[{count}/{len(selected_case_keys)}] case={case_key}")

        track_df = load_track_data(track_csv_path, storm_id)
        if track_df is None or len(track_df) < traj_model_cfg.t_history + traj_model_cfg.t_future:
            print("  跳过：轨迹数据不足")
            continue

        forecast_start_idx = int(
            first_data.get("forecast_start_idx")
            or first_data.get("start_idx")
            or heuristic_forecast_start_idx(track_df, traj_model_cfg.t_history, traj_model_cfg.t_future)
        )

        history_lat = track_df["lat"].values[forecast_start_idx - traj_model_cfg.t_history : forecast_start_idx].astype(np.float32)
        history_lon = track_df["lon"].values[forecast_start_idx - traj_model_cfg.t_history : forecast_start_idx].astype(np.float32)
        if len(history_lat) != traj_model_cfg.t_history:
            print("  跳过：历史窗口不足")
            continue
        history_coords_t = build_history_tensor(history_lat, history_lon, device)

        gt_lat = track_df["lat"].values[forecast_start_idx : forecast_start_idx + traj_model_cfg.t_future].astype(np.float32)
        gt_lon = track_df["lon"].values[forecast_start_idx : forecast_start_idx + traj_model_cfg.t_future].astype(np.float32)
        gt_lat = gt_lat[: traj_model_cfg.t_future]
        gt_lon = gt_lon[: traj_model_cfg.t_future]

        saved_samples: List[torch.Tensor] = []
        for pt_file, data in entries:
            file_samples = collect_saved_prediction_samples(data)
            saved_samples.extend(file_samples)
            print(f"  从 {pt_file.name} 读取 {len(file_samples)} 个样本")
            if len(saved_samples) >= args.num_samples:
                break

        if not saved_samples:
            print("  跳过：没有可用保存样本")
            continue
        if len(saved_samples) < args.num_samples:
            print(f"  [警告] 仅有 {len(saved_samples)} 个样本，将使用可用子集")

        sample_results = []
        for sample_no, preds_norm in enumerate(saved_samples[: args.num_samples], 1):
            future_era5 = prepare_future_era5_for_traj(
                preds_norm,
                t_future=traj_model_cfg.t_future,
                era5_channels=era5_channels,
                device=device,
            )
            sample_results.append(
                run_trajectory_prediction_once(
                    traj_model=traj_model,
                    history_coords_t=history_coords_t,
                    future_era5_for_traj=future_era5,
                    gt_lat=gt_lat,
                    gt_lon=gt_lon,
                    lat_range=lat_range,
                    lon_range=lon_range,
                )
            )
            print(f"  样本 {sample_no:02d}/{min(args.num_samples, len(saved_samples))} 完成")

        aggregated = aggregate_table3_predictions(sample_results, selection_strategy=args.selection_strategy)
        aggregated = apply_report_view(aggregated, report_indices, args.report_every_hours)
        
        # Filter by min_lead_time_hours if specified
        if args.min_lead_time_hours > 0:
            mask = aggregated["report_hours"] >= args.min_lead_time_hours
            aggregated["error_km"] = aggregated["error_km"][mask]
            aggregated["report_hours"] = aggregated["report_hours"][mask]
            aggregated["report_every_hours"] = args.report_every_hours  # Keep original reporting interval
        
        aggregated["storm_id"] = storm_id
        aggregated["case_key"] = case_key
        results.append(aggregated)
        print(
            f"  [Table 3] mean={aggregated['error_km'].mean():.2f}km | "
            f"+72h={aggregated['error_km'][-1]:.2f}km"
        )

    return results


def parse_args():
    parser = argparse.ArgumentParser(description="论文 Table 3 风格评估：oracle best-of-N 轨迹 FDE")
    add_common_arguments(parser)
    parser.add_argument(
        "--selection_strategy",
        choices=["per_lead_min", "best_72h_traj"],
        default="per_lead_min",
        help="Table 3 的 oracle 聚合策略；默认 per_lead_min。",
    )
    parser.add_argument(
        "--min_lead_time_hours",
        type=int,
        default=0,
        help="最小预报时效（小时）。只报告 >= 此值的误差。例如 48 表示只报告 48h 及以后的误差",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    validate_args(args, needs_table3_strategy=True)

    if args.mode == "end_to_end":
        results = run_end_to_end(args)
    else:
        results = run_from_saved(args)

    print_summary(results, mode_label=f"Table 3 ({args.selection_strategy})")
    save_results(args.output_dir, mode_name="table3", results=results)


if __name__ == "__main__":
    main()
