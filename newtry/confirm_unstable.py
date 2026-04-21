"""
验证扩散模型不稳定性是否会拖差 Table 2 结果。

功能分三层：
1. 审计现有 outputs/ar_pred_*.pt 是否真的是“同一 case 的重复采样”。
2. 对同一扩散 case 重复采样 N 次，量化风场预测 spread / ensemble 效果。
3. 如提供轨迹模型 checkpoint，则继续量化对下游轨迹 Table 2 / best-of-N 的影响。

核心判断思路：
- 如果同一 case 的 20 次扩散采样 spread 很大，
- 且 Table 2 风格的 ensemble-mean trajectory FDE 明显差于单成员平均或 best-of-N，
- 那么“扩散采样不稳定 / 多模态”就是 Table 2 变差的强证据之一。
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from configs import get_config
from data.dataset import ERA5TyphoonDataset, split_typhoon_ids
from inference import ERA5Predictor
from models import ERA5DiffusionModel
from train import EMA


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def denormalize_field(data_norm: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """(C, H, W) 反归一化到物理空间。"""
    safe_std = np.where(std < 1e-8, 1.0, std)
    return data_norm * safe_std.reshape(-1, 1, 1) + mean.reshape(-1, 1, 1)


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2)))


def build_test_dataset(
    data_root: str,
    preprocess_dir: Optional[str],
    norm_mean: np.ndarray,
    norm_std: np.ndarray,
) -> Tuple[object, object, object]:
    """构建扩散模型测试集。"""
    data_cfg, model_cfg, _, infer_cfg = get_config(data_root=data_root)
    _, _, test_ids = split_typhoon_ids(data_cfg.data_root, seed=42)
    test_dataset = ERA5TyphoonDataset(
        typhoon_ids=test_ids,
        data_root=data_cfg.data_root,
        pl_vars=data_cfg.pressure_level_vars,
        sfc_vars=data_cfg.surface_vars,
        pressure_levels=data_cfg.pressure_levels,
        history_steps=data_cfg.history_steps,
        forecast_steps=data_cfg.forecast_steps,
        norm_mean=norm_mean[: data_cfg.num_channels],
        norm_std=norm_std[: data_cfg.num_channels],
        preprocessed_dir=preprocess_dir,
    )
    return data_cfg, model_cfg, infer_cfg, test_dataset


def load_diffusion_predictor(
    checkpoint: str,
    data_root: str,
    preprocess_dir: Optional[str],
    norm_stats_path: str,
    ddim_steps: Optional[int],
    noise_sigma_override: Optional[float],
    ar_ensemble_override: Optional[int],
    device: torch.device,
):
    """加载扩散模型与推理器。"""
    stats = torch.load(norm_stats_path, map_location="cpu", weights_only=True)
    norm_mean = stats["mean"].numpy()
    norm_std = stats["std"].numpy()

    data_cfg, model_cfg, infer_cfg, test_dataset = build_test_dataset(
        data_root=data_root,
        preprocess_dir=preprocess_dir,
        norm_mean=norm_mean,
        norm_std=norm_std,
    )

    if ddim_steps is not None:
        infer_cfg.ddim_steps = ddim_steps
    if noise_sigma_override is not None:
        infer_cfg.autoregressive_noise_sigma = noise_sigma_override
    if ar_ensemble_override is not None:
        infer_cfg.ar_ensemble_per_step = ar_ensemble_override

    model = ERA5DiffusionModel(model_cfg, data_cfg).to(device)
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    if "ema_state_dict" in ckpt:
        ema = EMA(model, decay=0.9999)
        ema.load_state_dict(ckpt["ema_state_dict"])
        ema.apply_shadow(model)
        print("[扩散模型] 已加载 EMA 参数")
    else:
        model.load_state_dict(ckpt["model_state_dict"])
        print("[扩散模型] 已加载基础参数")
    model.eval()

    predictor = ERA5Predictor(model, data_cfg, infer_cfg, norm_mean, norm_std, device)
    return data_cfg, infer_cfg, predictor, test_dataset, norm_mean, norm_std


def build_dataset_position_maps(test_dataset: ERA5TyphoonDataset) -> Tuple[Dict[str, List[int]], Dict[int, int]]:
    """
    构建:
    - tid_to_indices: 每个台风对应哪些 dataset index
    - dataset_idx_to_pos: dataset index 在该台风内部的相对位置
    """
    tid_to_indices: Dict[str, List[int]] = defaultdict(list)
    for dataset_idx in range(len(test_dataset)):
        sample = test_dataset[dataset_idx]
        tid_to_indices[sample["typhoon_id"]].append(dataset_idx)

    dataset_idx_to_pos: Dict[int, int] = {}
    for tid, indices in tid_to_indices.items():
        for pos, dataset_idx in enumerate(indices):
            dataset_idx_to_pos[dataset_idx] = pos
    return tid_to_indices, dataset_idx_to_pos


def extract_start_idx(test_dataset: ERA5TyphoonDataset, dataset_idx: int, dataset_idx_to_pos: Dict[int, int]) -> Optional[int]:
    """
    获取样本对应的时间起点。

    - 预处理模式下直接返回真实 start_idx
    - NC 模式下没有显式 start_idx，只返回该台风内部的相对窗口位置
    """
    sample_meta = test_dataset.samples[dataset_idx]
    if len(sample_meta) >= 2 and isinstance(sample_meta[1], int):
        return int(sample_meta[1])
    return int(dataset_idx_to_pos[dataset_idx])


def audit_saved_outputs(
    output_dir: str,
    test_dataset: ERA5TyphoonDataset,
    dataset_idx_to_pos: Dict[int, int],
    save_dir: str,
) -> Dict[str, object]:
    """
    审计现有 ar_pred_*.pt：
    它们是否真的是同一 case 的重复采样，还是不同 dataset sample 的普通批量输出。
    """
    ar_files = sorted(Path(output_dir).glob("ar_pred_*.pt"))
    if not ar_files:
        print(f"[审计] 在 {output_dir} 中没有找到 ar_pred_*.pt，跳过。")
        return {
            "num_files": 0,
            "unique_cases": 0,
            "same_case_repeated": False,
            "rows": [],
        }

    rows: List[Dict[str, object]] = []
    for pt_file in ar_files:
        match = re.search(r"ar_pred_(\d+)\.pt$", pt_file.name)
        if not match:
            continue
        dataset_idx = int(match.group(1))
        if dataset_idx >= len(test_dataset):
            continue

        payload = torch.load(pt_file, map_location="cpu", weights_only=False)
        dataset_sample = test_dataset[dataset_idx]
        tid = dataset_sample["typhoon_id"]
        row = {
            "file": pt_file.name,
            "dataset_idx": dataset_idx,
            "saved_tid": payload.get("typhoon_id"),
            "dataset_tid": tid,
            "relative_case_pos": dataset_idx_to_pos[dataset_idx],
            "start_idx_or_relative_pos": extract_start_idx(test_dataset, dataset_idx, dataset_idx_to_pos),
        }
        rows.append(row)

    case_keys = {
        (row["dataset_tid"], row["start_idx_or_relative_pos"])
        for row in rows
    }
    same_case_repeated = len(case_keys) == 1 and len(rows) > 1

    os.makedirs(save_dir, exist_ok=True)
    csv_path = os.path.join(save_dir, "saved_output_audit.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(
            fp,
            fieldnames=[
                "file",
                "dataset_idx",
                "saved_tid",
                "dataset_tid",
                "relative_case_pos",
                "start_idx_or_relative_pos",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print("\n[审计现有 outputs/ar_pred_*.pt]")
    print(f"  文件数: {len(rows)}")
    print(f"  唯一 case 数: {len(case_keys)}")
    if same_case_repeated:
        print("  结论: 这些文件看起来是同一 case 的重复采样。")
    else:
        print("  结论: 这些文件不是同一 case 的重复采样，不能直接当作 Table 2/3 的 20 次采样。")
    print(f"  审计表已保存: {csv_path}")

    return {
        "num_files": len(rows),
        "unique_cases": len(case_keys),
        "same_case_repeated": same_case_repeated,
        "rows": rows,
    }


def choose_case_indices(
    test_dataset: ERA5TyphoonDataset,
    tid_to_indices: Dict[str, List[int]],
    requested_indices: Optional[Sequence[int]],
    target_typhoon_ids: Optional[Sequence[str]],
    num_cases: int,
) -> List[int]:
    """选择要做重复采样实验的 dataset indices。"""
    if requested_indices:
        return [idx for idx in requested_indices if 0 <= idx < len(test_dataset)]

    selected: List[int] = []
    target_set = set(target_typhoon_ids or [])

    for tid in sorted(tid_to_indices.keys()):
        if target_set and tid not in target_set:
            continue
        indices = tid_to_indices[tid]
        selected.append(indices[len(indices) // 2])
        if num_cases > 0 and len(selected) >= num_cases:
            break

    return selected


def collect_gt_sequence(
    test_dataset: ERA5TyphoonDataset,
    start_idx: int,
    num_ar_steps: int,
    num_channels: int,
) -> Tuple[List[np.ndarray], str]:
    """
    从 dataset 中收集该起点后续每个 lead time 的真值第一帧。
    """
    start_sample = test_dataset[start_idx]
    tid = start_sample["typhoon_id"]
    gt_list: List[np.ndarray] = []

    for lead_idx in range(num_ar_steps):
        dataset_idx = start_idx + lead_idx
        if dataset_idx >= len(test_dataset):
            break
        sample = test_dataset[dataset_idx]
        if sample["typhoon_id"] != tid:
            break
        gt_list.append(sample["target"][:num_channels].numpy().astype(np.float32))

    return gt_list, tid


def analyze_diffusion_samples(
    pred_trials: np.ndarray,
    gt_norm_seq: np.ndarray,
    norm_mean: np.ndarray,
    norm_std: np.ndarray,
) -> Dict[str, np.ndarray]:
    """
    对重复采样得到的风场进行稳定性分析。

    pred_trials: (N, T, C, H, W) 归一化空间
    gt_norm_seq: (T, C, H, W)     归一化空间
    """
    ensemble_mean = pred_trials.mean(axis=0)  # (T, C, H, W)
    member_gt_rmse_norm = np.sqrt(((pred_trials - gt_norm_seq[None]) ** 2).mean(axis=(2, 3, 4)))  # (N, T)
    ensemble_gt_rmse_norm = np.sqrt(((ensemble_mean - gt_norm_seq) ** 2).mean(axis=(1, 2, 3)))  # (T,)
    spread_to_mean_norm = np.sqrt(((pred_trials - ensemble_mean[None]) ** 2).mean(axis=(2, 3, 4)))  # (N, T)
    pairwise_spread_norm = np.sqrt(np.mean((pred_trials[:, None] - pred_trials[None, :]) ** 2, axis=(2, 3, 4, 5)))  # (N, N, T)

    T = pred_trials.shape[1]
    C = pred_trials.shape[2]
    pred_trials_phys = np.empty_like(pred_trials)
    gt_phys = np.empty_like(gt_norm_seq)
    ensemble_mean_phys = np.empty_like(ensemble_mean)
    for t in range(T):
        gt_phys[t] = denormalize_field(gt_norm_seq[t], norm_mean[:C], norm_std[:C])
        ensemble_mean_phys[t] = denormalize_field(ensemble_mean[t], norm_mean[:C], norm_std[:C])
        for n in range(pred_trials.shape[0]):
            pred_trials_phys[n, t] = denormalize_field(pred_trials[n, t], norm_mean[:C], norm_std[:C])

    member_gt_rmse_phys = np.sqrt(((pred_trials_phys - gt_phys[None]) ** 2).mean(axis=(2, 3, 4)))  # (N, T)
    ensemble_gt_rmse_phys = np.sqrt(((ensemble_mean_phys - gt_phys) ** 2).mean(axis=(1, 2, 3)))  # (T,)
    spread_to_mean_phys = np.sqrt(((pred_trials_phys - ensemble_mean_phys[None]) ** 2).mean(axis=(2, 3, 4)))  # (N, T)

    return {
        "member_gt_rmse_norm": member_gt_rmse_norm.astype(np.float32),
        "ensemble_gt_rmse_norm": ensemble_gt_rmse_norm.astype(np.float32),
        "spread_to_mean_norm": spread_to_mean_norm.astype(np.float32),
        "pairwise_spread_norm": pairwise_spread_norm.astype(np.float32),
        "member_gt_rmse_phys": member_gt_rmse_phys.astype(np.float32),
        "ensemble_gt_rmse_phys": ensemble_gt_rmse_phys.astype(np.float32),
        "spread_to_mean_phys": spread_to_mean_phys.astype(np.float32),
    }


def load_trajectory_runtime(
    trajectory_project_dir: str,
    trajectory_ckpt: str,
    trajectory_bias_path: Optional[str],
):
    """
    动态导入 Trajectory 项目的 helper。
    """
    if trajectory_project_dir not in sys.path:
        sys.path.insert(0, trajectory_project_dir)

    from paper_eval_common import (
        compute_track_error_km,
        load_track_data,
        load_trajectory_model,
        prepare_future_era5_for_traj,
        traj_data_cfg,
        traj_model_cfg,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    traj_model = load_trajectory_model(
        checkpoint_path=trajectory_ckpt,
        era5_channels=9,
        device=device,
        bias_path=trajectory_bias_path,
    )
    return {
        "device": device,
        "traj_model": traj_model,
        "compute_track_error_km": compute_track_error_km,
        "load_track_data": load_track_data,
        "prepare_future_era5_for_traj": prepare_future_era5_for_traj,
        "traj_data_cfg": traj_data_cfg,
        "traj_model_cfg": traj_model_cfg,
    }


def analyze_trajectory_samples(
    pred_trials: np.ndarray,
    trajectory_rt: Dict[str, object],
    track_csv_path: str,
    storm_id: str,
    forecast_start_idx: int,
) -> Optional[Dict[str, np.ndarray]]:
    """
    将多次扩散采样送入轨迹模型，检查 Table 2 是否被拉差。
    """
    traj_model = trajectory_rt["traj_model"]
    traj_model_cfg = trajectory_rt["traj_model_cfg"]
    traj_data_cfg = trajectory_rt["traj_data_cfg"]
    load_track_data = trajectory_rt["load_track_data"]
    prepare_future_era5_for_traj = trajectory_rt["prepare_future_era5_for_traj"]
    compute_track_error_km = trajectory_rt["compute_track_error_km"]
    device = trajectory_rt["device"]

    track_df = load_track_data(track_csv_path, storm_id)
    if track_df is None:
        return None

    t_hist = traj_model_cfg.t_history
    t_fut = min(traj_model_cfg.t_future, pred_trials.shape[1])
    if forecast_start_idx < t_hist or forecast_start_idx + t_fut > len(track_df):
        return None

    history_lat = track_df["lat"].values[forecast_start_idx - t_hist : forecast_start_idx].astype(np.float32)
    history_lon = track_df["lon"].values[forecast_start_idx - t_hist : forecast_start_idx].astype(np.float32)
    if len(history_lat) != t_hist:
        return None

    from dataset import normalize_coords  # Trajectory project import after sys.path adjusted

    h_lat_n, h_lon_n = normalize_coords(history_lat, history_lon)
    history_coords = np.stack([h_lat_n, h_lon_n], axis=-1)
    history_coords_t = torch.from_numpy(history_coords).float().unsqueeze(0).to(device)

    gt_lat = track_df["lat"].values[forecast_start_idx : forecast_start_idx + t_fut].astype(np.float32)
    gt_lon = track_df["lon"].values[forecast_start_idx : forecast_start_idx + t_fut].astype(np.float32)

    lat_range = traj_data_cfg.lat_range
    lon_range = traj_data_cfg.lon_range
    pred_lat_members: List[np.ndarray] = []
    pred_lon_members: List[np.ndarray] = []
    member_errors: List[np.ndarray] = []

    for trial_idx in range(pred_trials.shape[0]):
        future_era5 = prepare_future_era5_for_traj(
            torch.from_numpy(pred_trials[trial_idx]).float().unsqueeze(0),
            t_future=traj_model_cfg.t_future,
            era5_channels=9,
            device=device,
        )
        with torch.no_grad():
            outputs = traj_model.predict(history_coords_t, future_era5)
        pred_coords = outputs["predicted_coords"].detach().cpu().numpy()[0][:t_fut]
        pred_lat = pred_coords[:, 0] * (lat_range[1] - lat_range[0]) + lat_range[0]
        pred_lon = pred_coords[:, 1] * (lon_range[1] - lon_range[0]) + lon_range[0]
        error_km = compute_track_error_km(pred_lat, pred_lon, gt_lat, gt_lon)
        pred_lat_members.append(pred_lat.astype(np.float32))
        pred_lon_members.append(pred_lon.astype(np.float32))
        member_errors.append(error_km.astype(np.float32))

    pred_lat_stack = np.stack(pred_lat_members, axis=0)
    pred_lon_stack = np.stack(pred_lon_members, axis=0)
    member_error_stack = np.stack(member_errors, axis=0)

    ensemble_lat = pred_lat_stack.mean(axis=0)
    ensemble_lon = pred_lon_stack.mean(axis=0)
    ensemble_error = compute_track_error_km(ensemble_lat, ensemble_lon, gt_lat, gt_lon).astype(np.float32)
    best_of_n_error = member_error_stack.min(axis=0).astype(np.float32)

    spread_km_members = []
    for trial_idx in range(pred_lat_stack.shape[0]):
        spread_km_members.append(
            compute_track_error_km(
                pred_lat_stack[trial_idx],
                pred_lon_stack[trial_idx],
                ensemble_lat,
                ensemble_lon,
            )
        )
    spread_km = np.stack(spread_km_members, axis=0).mean(axis=0).astype(np.float32)

    return {
        "member_error_km": member_error_stack.astype(np.float32),
        "ensemble_error_km": ensemble_error,
        "best_of_n_error_km": best_of_n_error,
        "spread_km": spread_km,
        "ensemble_gap_vs_member_mean": (ensemble_error - member_error_stack.mean(axis=0)).astype(np.float32),
        "ensemble_gap_vs_best_of_n": (ensemble_error - best_of_n_error).astype(np.float32),
    }


def summarize_key_hours(values: np.ndarray, hours: Sequence[int], lead_hours: Sequence[int]) -> Dict[str, float]:
    result = {}
    for hour in hours:
        try:
            idx = lead_hours.index(hour)
        except ValueError:
            continue
        result[f"{hour}h"] = float(values[idx])
    return result


def print_case_summary(
    case_name: str,
    lead_hours: List[int],
    diffusion_metrics: Dict[str, np.ndarray],
    trajectory_metrics: Optional[Dict[str, np.ndarray]],
) -> None:
    key_hours = [24, 48, 72]
    print(f"\n[Case] {case_name}")
    diff_member = diffusion_metrics["member_gt_rmse_norm"].mean(axis=0)
    diff_ens = diffusion_metrics["ensemble_gt_rmse_norm"]
    diff_spread = diffusion_metrics["spread_to_mean_norm"].mean(axis=0)
    print("  扩散风场(归一化空间):")
    for hour in key_hours:
        if hour not in lead_hours:
            continue
        idx = lead_hours.index(hour)
        print(
            f"    +{hour:2d}h | member_mean_rmse={diff_member[idx]:.4f} | "
            f"ensemble_rmse={diff_ens[idx]:.4f} | spread={diff_spread[idx]:.4f}"
        )

    if trajectory_metrics is not None:
        member_mean = trajectory_metrics["member_error_km"].mean(axis=0)
        ensemble = trajectory_metrics["ensemble_error_km"]
        best = trajectory_metrics["best_of_n_error_km"]
        spread = trajectory_metrics["spread_km"]
        print("  下游轨迹(km):")
        for hour in key_hours:
            if hour not in lead_hours:
                continue
            idx = lead_hours.index(hour)
            print(
                f"    +{hour:2d}h | member_mean={member_mean[idx]:7.2f} | "
                f"table2={ensemble[idx]:7.2f} | best_of_n={best[idx]:7.2f} | "
                f"track_spread={spread[idx]:7.2f}"
            )


def main():
    parser = argparse.ArgumentParser(description="验证扩散采样不稳定是否拖差 Table 2")
    parser.add_argument("--checkpoint", type=str, required=True, help="扩散模型 checkpoint")
    parser.add_argument("--data_root", type=str, required=True, help="ERA5 数据根目录")
    parser.add_argument("--work_dir", type=str, default=ROOT_DIR, help="包含 norm_stats.pt 的目录")
    parser.add_argument("--norm_stats", type=str, default=None, help="归一化统计文件，默认 work_dir/norm_stats.pt")
    parser.add_argument("--preprocess_dir", type=str, default=None)
    parser.add_argument("--save_dir", type=str, default="instability_check")

    parser.add_argument("--num_cases", type=int, default=3, help="抽查多少个 case")
    parser.add_argument("--sample_indices", type=int, nargs="*", default=None, help="直接指定 dataset index")
    parser.add_argument("--target_typhoon_ids", nargs="*", default=None, help="只分析指定台风")

    parser.add_argument("--num_trials", type=int, default=20, help="同一 case 重复采样次数")
    parser.add_argument("--num_ar_steps", type=int, default=24, help="自回归步数，24=72h")
    parser.add_argument("--ddim_steps", type=int, default=None)
    parser.add_argument("--noise_sigma", type=float, default=None, help="覆盖 autoregressive_noise_sigma")
    parser.add_argument("--ar_ensemble", type=int, default=None, help="覆盖 ar_ensemble_per_step")
    parser.add_argument("--seed_base", type=int, default=1234, help="每次 trial 用 seed_base+trial_idx 固定随机种子")

    parser.add_argument("--audit_output_dir", type=str, default="outputs", help="先审计该目录中的 ar_pred_*.pt")

    parser.add_argument("--trajectory_project_dir", type=str, default=None, help="可选，Trajectory 项目目录")
    parser.add_argument("--trajectory_ckpt", type=str, default=None, help="可选，轨迹模型 checkpoint")
    parser.add_argument("--trajectory_bias_path", type=str, default=None, help="可选，lead_time_bias.pt")
    parser.add_argument("--track_csv", type=str, default=None, help="可选，轨迹 CSV")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    norm_stats_path = args.norm_stats or os.path.join(args.work_dir, "norm_stats.pt")
    data_cfg, infer_cfg, predictor, test_dataset, norm_mean, norm_std = load_diffusion_predictor(
        checkpoint=args.checkpoint,
        data_root=args.data_root,
        preprocess_dir=args.preprocess_dir,
        norm_stats_path=norm_stats_path,
        ddim_steps=args.ddim_steps,
        noise_sigma_override=args.noise_sigma,
        ar_ensemble_override=args.ar_ensemble,
        device=device,
    )

    print("\n[扩散推理配置]")
    print(f"  ddim_steps: {infer_cfg.ddim_steps}")
    print(f"  autoregressive_noise_sigma: {infer_cfg.autoregressive_noise_sigma}")
    print(f"  ar_ensemble_per_step: {infer_cfg.ar_ensemble_per_step}")
    print(f"  num_trials: {args.num_trials}")

    tid_to_indices, dataset_idx_to_pos = build_dataset_position_maps(test_dataset)

    audit_dir = args.audit_output_dir
    if not os.path.isabs(audit_dir):
        audit_dir = os.path.join(ROOT_DIR, audit_dir)
    audit_summary = audit_saved_outputs(
        output_dir=audit_dir,
        test_dataset=test_dataset,
        dataset_idx_to_pos=dataset_idx_to_pos,
        save_dir=args.save_dir,
    )

    selected_indices = choose_case_indices(
        test_dataset=test_dataset,
        tid_to_indices=tid_to_indices,
        requested_indices=args.sample_indices,
        target_typhoon_ids=args.target_typhoon_ids,
        num_cases=args.num_cases,
    )
    if not selected_indices:
        raise RuntimeError("没有找到可分析的 case，请检查 sample_indices/target_typhoon_ids 设置。")

    trajectory_rt = None
    if args.trajectory_ckpt:
        trajectory_project_dir = args.trajectory_project_dir or os.path.join(os.path.dirname(ROOT_DIR), "Trajectory")
        track_csv_path = args.track_csv or os.path.join(trajectory_project_dir, "processed_typhoon_tracks.csv")
        trajectory_rt = load_trajectory_runtime(
            trajectory_project_dir=trajectory_project_dir,
            trajectory_ckpt=args.trajectory_ckpt,
            trajectory_bias_path=args.trajectory_bias_path,
        )
    else:
        track_csv_path = None

    case_rows: List[Dict[str, object]] = []
    summary_rows: Dict[int, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))

    for case_order, dataset_idx in enumerate(selected_indices, 1):
        gt_seq_list, tid = collect_gt_sequence(
            test_dataset=test_dataset,
            start_idx=dataset_idx,
            num_ar_steps=args.num_ar_steps,
            num_channels=data_cfg.num_channels,
        )
        if not gt_seq_list:
            print(f"[跳过] dataset_idx={dataset_idx} 没有可用真值序列")
            continue

        valid_steps = len(gt_seq_list)
        lead_hours = [(step + 1) * data_cfg.time_interval_hours for step in range(valid_steps)]
        start_idx = extract_start_idx(test_dataset, dataset_idx, dataset_idx_to_pos)
        case_name = f"{tid}::dataset_idx={dataset_idx}::start={start_idx}"
        sample = test_dataset[dataset_idx]
        cond = sample["condition"].unsqueeze(0).to(device)

        pred_trials: List[np.ndarray] = []
        for trial_idx in range(args.num_trials):
            set_global_seed(args.seed_base + trial_idx)
            with torch.no_grad():
                preds = predictor.predict_autoregressive(
                    cond,
                    num_steps=valid_steps,
                    noise_sigma=infer_cfg.autoregressive_noise_sigma,
                    ensemble_per_step=infer_cfg.ar_ensemble_per_step,
                )
            pred_trials.append(torch.stack(preds, dim=1).cpu().numpy()[0])

        pred_trials_np = np.stack(pred_trials, axis=0).astype(np.float32)  # (N, T, C, H, W)
        gt_seq_np = np.stack(gt_seq_list, axis=0).astype(np.float32)        # (T, C, H, W)

        diffusion_metrics = analyze_diffusion_samples(
            pred_trials=pred_trials_np,
            gt_norm_seq=gt_seq_np,
            norm_mean=norm_mean,
            norm_std=norm_std,
        )

        trajectory_metrics = None
        if trajectory_rt is not None and track_csv_path is not None and start_idx is not None:
            forecast_start_idx = int(start_idx) + int(data_cfg.history_steps)
            trajectory_metrics = analyze_trajectory_samples(
                pred_trials=pred_trials_np,
                trajectory_rt=trajectory_rt,
                track_csv_path=track_csv_path,
                storm_id=tid,
                forecast_start_idx=forecast_start_idx,
            )

        print_case_summary(case_name, lead_hours, diffusion_metrics, trajectory_metrics)

        member_mean_norm = diffusion_metrics["member_gt_rmse_norm"].mean(axis=0)
        ensemble_norm = diffusion_metrics["ensemble_gt_rmse_norm"]
        spread_norm = diffusion_metrics["spread_to_mean_norm"].mean(axis=0)
        member_mean_phys = diffusion_metrics["member_gt_rmse_phys"].mean(axis=0)
        ensemble_phys = diffusion_metrics["ensemble_gt_rmse_phys"]
        spread_phys = diffusion_metrics["spread_to_mean_phys"].mean(axis=0)

        row: Dict[str, object] = {
            "case_name": case_name,
            "storm_id": tid,
            "dataset_idx": dataset_idx,
            "start_idx": start_idx,
            "valid_steps": valid_steps,
        }
        for hour in (24, 48, 72):
            if hour not in lead_hours:
                continue
            idx = lead_hours.index(hour)
            row[f"diff_member_mean_norm_{hour}h"] = float(member_mean_norm[idx])
            row[f"diff_ensemble_norm_{hour}h"] = float(ensemble_norm[idx])
            row[f"diff_spread_norm_{hour}h"] = float(spread_norm[idx])
            row[f"diff_member_mean_phys_{hour}h"] = float(member_mean_phys[idx])
            row[f"diff_ensemble_phys_{hour}h"] = float(ensemble_phys[idx])
            row[f"diff_spread_phys_{hour}h"] = float(spread_phys[idx])

            summary_rows[hour]["diff_member_mean_norm"].append(float(member_mean_norm[idx]))
            summary_rows[hour]["diff_ensemble_norm"].append(float(ensemble_norm[idx]))
            summary_rows[hour]["diff_spread_norm"].append(float(spread_norm[idx]))
            summary_rows[hour]["diff_member_mean_phys"].append(float(member_mean_phys[idx]))
            summary_rows[hour]["diff_ensemble_phys"].append(float(ensemble_phys[idx]))
            summary_rows[hour]["diff_spread_phys"].append(float(spread_phys[idx]))

        if trajectory_metrics is not None:
            member_mean_track = trajectory_metrics["member_error_km"].mean(axis=0)
            ensemble_track = trajectory_metrics["ensemble_error_km"]
            best_track = trajectory_metrics["best_of_n_error_km"]
            spread_track = trajectory_metrics["spread_km"]
            gap_vs_member = trajectory_metrics["ensemble_gap_vs_member_mean"]
            gap_vs_best = trajectory_metrics["ensemble_gap_vs_best_of_n"]

            for hour in (24, 48, 72):
                if hour not in lead_hours:
                    continue
                idx = lead_hours.index(hour)
                row[f"track_member_mean_{hour}h_km"] = float(member_mean_track[idx])
                row[f"track_table2_{hour}h_km"] = float(ensemble_track[idx])
                row[f"track_best_of_n_{hour}h_km"] = float(best_track[idx])
                row[f"track_spread_{hour}h_km"] = float(spread_track[idx])
                row[f"track_gap_vs_member_{hour}h_km"] = float(gap_vs_member[idx])
                row[f"track_gap_vs_best_{hour}h_km"] = float(gap_vs_best[idx])

                summary_rows[hour]["track_member_mean"].append(float(member_mean_track[idx]))
                summary_rows[hour]["track_table2"].append(float(ensemble_track[idx]))
                summary_rows[hour]["track_best_of_n"].append(float(best_track[idx]))
                summary_rows[hour]["track_spread"].append(float(spread_track[idx]))
                summary_rows[hour]["track_gap_vs_member"].append(float(gap_vs_member[idx]))
                summary_rows[hour]["track_gap_vs_best"].append(float(gap_vs_best[idx]))

        case_rows.append(row)

    csv_path = os.path.join(args.save_dir, "case_metrics.csv")
    if case_rows:
        fieldnames = sorted({key for row in case_rows for key in row.keys()})
        with open(csv_path, "w", newline="", encoding="utf-8") as fp:
            writer = csv.DictWriter(fp, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(case_rows)
        print(f"\n[保存] case metrics: {csv_path}")

    summary = {
        "audit_saved_outputs": {
            "num_files": audit_summary["num_files"],
            "unique_cases": audit_summary["unique_cases"],
            "same_case_repeated": audit_summary["same_case_repeated"],
        },
        "diffusion_config": {
            "ddim_steps": int(infer_cfg.ddim_steps),
            "autoregressive_noise_sigma": float(infer_cfg.autoregressive_noise_sigma),
            "ar_ensemble_per_step": int(infer_cfg.ar_ensemble_per_step),
            "num_trials": int(args.num_trials),
        },
        "aggregate": {},
    }

    print("\n" + "=" * 72)
    print("聚合结论")
    print("=" * 72)
    for hour in sorted(summary_rows.keys()):
        hour_key = f"{hour}h"
        summary["aggregate"][hour_key] = {}
        print(f"\n[{hour_key}]")
        for metric_name, values in summary_rows[hour].items():
            mean_value = float(np.mean(values)) if values else math.nan
            summary["aggregate"][hour_key][metric_name] = mean_value
            print(f"  {metric_name}: {mean_value:.4f}")

    if "72h" in summary["aggregate"]:
        agg_72h = summary["aggregate"]["72h"]
        if "track_gap_vs_member" in agg_72h:
            gap = agg_72h["track_gap_vs_member"]
            spread = agg_72h.get("track_spread", math.nan)
            if gap > 20 and spread > 20:
                verdict = "支持该假设：同一 case 的扩散采样 spread 较大，并显著拉高了 Table 2 风格轨迹误差。"
            elif gap > 0:
                verdict = "部分支持该假设：Table 2 风格轨迹误差高于成员均值，但幅度有限。"
            else:
                verdict = "不支持该假设：Table 2 风格轨迹误差没有明显高于成员均值。"
            summary["verdict"] = verdict
            print(f"\n[判断] {verdict}")
        else:
            diff_spread = agg_72h.get("diff_spread_norm", math.nan)
            diff_member = agg_72h.get("diff_member_mean_norm", math.nan)
            if diff_spread > 0.5 * diff_member:
                verdict = "扩散风场存在明显采样 spread，可能会传导到 Table 2，但当前未做轨迹层验证。"
            else:
                verdict = "从风场层面看 spread 不算极端，Table 2 变差未必主要由扩散不稳定导致。"
            summary["verdict"] = verdict
            print(f"\n[判断] {verdict}")

    summary_path = os.path.join(args.save_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as fp:
        json.dump(summary, fp, ensure_ascii=False, indent=2)
    print(f"[保存] summary: {summary_path}")


if __name__ == "__main__":
    main()
