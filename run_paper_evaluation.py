"""
Unified Paper Evaluation Runner — Complete Pipeline for FM vs DM Paper

This script orchestrates the complete evaluation pipeline, running all
scripts in sequence and generating all tables and figures for the paper.

Usage:
    python run_paper_evaluation.py \
        --fm_ckpt checkpoints_finetune_vpred/flow_matching_best.pt \
        --dm_ckpt checkpoints_finetune_vpred/diffusion_best.pt \
        --era5_dir Trajectory/preprocessed_era5/ \
        --track_csv Trajectory/processed_typhoon_tracks.csv \
        --output_dir paper_results/

Workflow:
1. Generate Table 2 (ensemble mean) for DM and FM
2. Generate Table 3 (oracle best) for DM
3. Run NFE sweep for both methods
4. Compute spectral fidelity
5. Compute geostrophic balance
6. Compute intensity metrics
7. Generate all paper figures
8. Compile final summary tables
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def run_command(cmd: List[str], desc: str, timeout: int = 3600) -> bool:
    """Run a shell command and report status."""
    print(f"\n{'='*70}")
    print(f"[RUN] {desc}")
    print(f"[CMD] {' '.join(str(x) for x in cmd)}")
    print(f"{'='*70}")
    start = time.time()
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=os.path.dirname(os.path.abspath(__file__)),
        )
        elapsed = time.time() - start
        if result.returncode == 0:
            print(f"[OK] {desc} completed in {elapsed:.1f}s")
            return True
        else:
            print(f"[FAIL] {desc} failed (exit {result.returncode})")
            if result.stdout:
                print("STDOUT:", result.stdout[-1000:])
            if result.stderr:
                print("STDERR:", result.stderr[-1000:])
            return False
    except subprocess.TimeoutExpired:
        print(f"[TIMEOUT] {desc} exceeded {timeout}s")
        return False
    except Exception as e:
        print(f"[ERROR] {desc}: {e}")
        return False


def run_table2_evaluation(
    output_dir: str,
    trajectory_ckpt: str,
    norm_stats: str,
    track_csv: str,
    diffusion_output_dir: str = "",
    diffusion_code: str = "",
    diffusion_ckpt: str = "",
    method: str = "dm",
    num_samples: int = 20,
    ddim_steps: int = 50,
    euler_steps: int = 4,
) -> Tuple[bool, str]:
    """Run Table 2 evaluation (ensemble mean)."""
    table2_dir = os.path.join(output_dir, f"table2_{method}")
    os.makedirs(table2_dir, exist_ok=True)

    cmd = [
        sys.executable,
        "Trajectory/table2_mode.py",
        "--mode", "from_saved" if diffusion_output_dir else "end_to_end",
        "--trajectory_ckpt", trajectory_ckpt,
        "--norm_stats", norm_stats,
        "--track_csv", track_csv,
        "--output_dir", table2_dir,
        "--num_samples", str(num_samples),
        "--report_every_hours", "6",
    ]

    if diffusion_output_dir:
        cmd += ["--diffusion_output_dir", diffusion_output_dir]
    else:
        cmd += [
            "--mode", "end_to_end",
            "--diffusion_code", diffusion_code,
            "--diffusion_ckpt", diffusion_ckpt,
            "--data_root", os.path.dirname(norm_stats.rstrip("/")),
        ]

    if method == "fm":
        cmd += ["--method", "fm"]
        cmd += ["--euler_steps", str(euler_steps)]
    else:
        cmd += ["--ddim_steps", str(ddim_steps)]

    success = run_command(cmd, f"Table 2 ({method.upper()})", timeout=7200)
    return success, table2_dir


def run_table3_evaluation(
    output_dir: str,
    trajectory_ckpt: str,
    norm_stats: str,
    track_csv: str,
    diffusion_output_dir: str = "",
    method: str = "dm",
    num_samples: int = 20,
    selection_strategy: str = "per_lead_min",
) -> Tuple[bool, str]:
    """Run Table 3 evaluation (oracle best)."""
    table3_dir = os.path.join(output_dir, f"table3_{method}")
    os.makedirs(table3_dir, exist_ok=True)

    cmd = [
        sys.executable,
        "Trajectory/table3_mode.py",
        "--mode", "from_saved",
        "--diffusion_output_dir", diffusion_output_dir,
        "--trajectory_ckpt", trajectory_ckpt,
        "--norm_stats", norm_stats,
        "--track_csv", track_csv,
        "--output_dir", table3_dir,
        "--num_samples", str(num_samples),
        "--report_every_hours", "6",
        "--selection_strategy", selection_strategy,
    ]

    success = run_command(cmd, f"Table 3 Oracle ({method.upper()})", timeout=7200)
    return success, table3_dir


def run_nfe_sweep(
    output_dir: str,
    fm_ckpt: str = "",
    dm_ckpt: str = "",
    device: str = "cuda",
    num_samples: int = 100,
) -> Tuple[bool, str]:
    """Run NFE sweep for both FM and DM."""
    nfe_dir = os.path.join(output_dir, "nfe_sweep")
    os.makedirs(nfe_dir, exist_ok=True)

    fm_success = True
    dm_success = True

    if fm_ckpt:
        cmd = [
            sys.executable,
            "compare_fm_dm/run_nfe_sweep.py",
            "--method", "fm",
            "--fm_ckpt", fm_ckpt,
            "--fm_euler_steps", "1", "4", "8", "16",
            "--num_eval_samples", str(num_samples),
            "--device", device,
            "--output_dir", os.path.join(nfe_dir, "fm"),
        ]
        fm_success = run_command(cmd, "NFE Sweep (FM)", timeout=3600)

    if dm_ckpt:
        cmd = [
            sys.executable,
            "compare_fm_dm/run_nfe_sweep.py",
            "--method", "dm",
            "--dm_ckpt", dm_ckpt,
            "--dm_ddim_steps", "10", "25", "50", "100",
            "--num_eval_samples", str(num_samples),
            "--device", device,
            "--output_dir", os.path.join(nfe_dir, "dm"),
        ]
        dm_success = run_command(cmd, "NFE Sweep (DM)", timeout=3600)

    return (fm_success and dm_success), nfe_dir


def run_spectral_fidelity(
    output_dir: str,
    pred_dir: str,
    method: str = "dm",
) -> Tuple[bool, str]:
    """Run spectral fidelity evaluation."""
    spectral_dir = os.path.join(output_dir, f"spectral_{method}")
    os.makedirs(spectral_dir, exist_ok=True)

    cmd = [
        sys.executable,
        "compare_fm_dm/evaluation/spectral_fidelity.py",
        "--pred_dir", pred_dir,
        "--output_dir", spectral_dir,
        "--method", method,
        "--z_channel_idx", "7",
        "--k_min", "5",
        "--k_max", "15",
        "--k_threshold", "15",
        "--lead_times", "24", "48", "72",
    ]

    success = run_command(cmd, f"Spectral Fidelity ({method.upper()})", timeout=3600)
    return success, spectral_dir


def run_geostrophic_evaluation(
    output_dir: str,
    pred_dir: str,
    method: str = "dm",
) -> Tuple[bool, str]:
    """Run geostrophic balance evaluation."""
    geo_dir = os.path.join(output_dir, f"geostrophic_{method}")
    os.makedirs(geo_dir, exist_ok=True)

    cmd = [
        sys.executable,
        "compare_fm_dm/evaluation/geostrophic.py",
        "--pred_dir", pred_dir,
        "--output_dir", geo_dir,
        "--method", method,
        "--z_channel_idx", "7",
        "--lead_times", "24", "48", "72",
    ]

    success = run_command(cmd, f"Geostrophic Balance ({method.upper()})", timeout=3600)
    return success, geo_dir


def run_intensity_evaluation(
    output_dir: str,
    pred_dir: str,
    method: str = "dm",
) -> Tuple[bool, str]:
    """Run intensity evaluation."""
    int_dir = os.path.join(output_dir, f"intensity_{method}")
    os.makedirs(int_dir, exist_ok=True)

    cmd = [
        sys.executable,
        "compare_fm_dm/evaluation/intensity.py",
        "--pred_dir", pred_dir,
        "--output_dir", int_dir,
        "--method", method,
        "--z_channel_idx", "7",
        "--lead_times", "24", "48", "72",
    ]

    success = run_command(cmd, f"Intensity Evaluation ({method.upper()})", timeout=3600)
    return success, int_dir


def generate_all_figures(
    output_dir: str,
    table2_dm_dir: str = "",
    table2_fm_dir: str = "",
    nfe_sweep_dir: str = "",
) -> Tuple[bool, str]:
    """Generate all paper figures."""
    fig_dir = os.path.join(output_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    cmd = [
        sys.executable,
        "compare_fm_dm/generate_paper_figures.py",
        "--output_dir", fig_dir,
    ]

    if nfe_sweep_dir:
        cmd += ["--nfe_results", nfe_sweep_dir]
    if table2_dm_dir:
        cmd += ["--table2_dm", table2_dm_dir]
    if table2_fm_dir:
        cmd += ["--table2_fm", table2_fm_dir]

    success = run_command(cmd, "Generate Paper Figures", timeout=1800)
    return success, fig_dir


def compile_final_summary(
    output_dir: str,
    table2_dm_summary: Dict = None,
    table2_fm_summary: Dict = None,
    nfe_results: Dict = None,
    spectral_results: Dict = None,
    geostrophic_results: Dict = None,
    intensity_results: Dict = None,
) -> str:
    """Compile the final summary JSON with all results."""
    summary = {
        "generated_at": datetime.now().isoformat(),
        "report_version": "1.0",
        "methods": {
            "DM": {"description": "Diffusion Model (DDIM, NFE=50)", "stochastic": True},
            "FM": {"description": "Flow Matching (Euler, NFE=4)", "stochastic": False},
        },
    }

    if table2_dm_summary:
        summary["table2_dm"] = table2_dm_summary
    if table2_fm_summary:
        summary["table2_fm"] = table2_fm_summary
    if nfe_results:
        summary["nfe_sweep"] = nfe_results
    if spectral_results:
        summary["spectral_fidelity"] = spectral_results
    if geostrophic_results:
        summary["geostrophic_balance"] = geostrophic_results
    if intensity_results:
        summary["intensity"] = intensity_results

    summary_path = os.path.join(output_dir, "paper_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\n[Summary] Saved: {summary_path}")
    return summary_path


def print_final_report(output_dir: str):
    """Print a formatted final report."""
    print("\n" + "="*80)
    print("FINAL EVALUATION REPORT — Flow Matching vs Diffusion for Typhoon Forecasting")
    print("="*80)

    summary_path = os.path.join(output_dir, "paper_summary.json")
    if not os.path.exists(summary_path):
        print("[Warning] No summary found. Run evaluation first.")
        return

    with open(summary_path) as f:
        summary = json.load(f)

    print(f"\nGenerated: {summary.get('generated_at', 'N/A')}")

    print("\n--- Track Error (Table 2) ---")
    for method in ["table2_dm", "table2_fm"]:
        if method in summary:
            s = summary[method]
            print(f"\n{method.upper().replace('TABLE2_', '')}:")
            if "error_by_hour" in s:
                for hour_key in ["24h", "48h", "72h"]:
                    if hour_key in s["error_by_hour"]:
                        v = s["error_by_hour"][hour_key]
                        print(f"  {hour_key}: {v['mean_km']:.1f} ± {v['std_km']:.1f} km")

    print("\n--- NFE Sweep ---")
    nfe_dir = os.path.join(output_dir, "nfe_sweep")
    for sub in ["fm", "dm"]:
        sub_path = os.path.join(nfe_dir, sub, f"nfe_sweep_{sub}.json")
        if os.path.exists(sub_path):
            with open(sub_path) as f:
                data = json.load(f)
            print(f"\n{sub.upper()}:")
            for r in data:
                print(f"  NFE={r['nfe']:3d}: RMSE={r['rmse_mean']:.4f}, Time={r['inference_time_ms_mean']:.1f}ms")

    print("\n--- Spectral Fidelity ---")
    for method in ["spectral_dm", "spectral_fm"]:
        spectral_dir = os.path.join(output_dir, method)
        spectral_path = os.path.join(spectral_dir, f"spectral_{method.split('_')[1]}_summary.json")
        if os.path.exists(spectral_path):
            with open(spectral_path) as f:
                s = json.load(f)
            agg = s.get("aggregated", {})
            print(f"\n{method.upper()}:")
            for lead in ["24h", "48h", "72h"]:
                beta_key = f"beta_{lead}"
                if beta_key in agg:
                    v = agg[beta_key]
                    if isinstance(v, dict):
                        print(f"  {lead}: beta = {v.get('mean', 'N/A'):.2f}")

    print("\n--- Figures Generated ---")
    fig_dir = os.path.join(output_dir, "figures")
    if os.path.exists(fig_dir):
        import glob
        figs = sorted(glob.glob(os.path.join(fig_dir, "fig*.png")))
        for fig in figs:
            print(f"  {os.path.basename(fig)}")

    print("\n" + "="*80)
    print(f"Results saved to: {output_dir}")
    print("="*80)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Unified Paper Evaluation: Complete FM vs DM evaluation pipeline"
    )

    group = parser.add_argument_group("Paths")
    group.add_argument("--output_dir", type=str, default="paper_results",
                      help="Output directory for all results")
    group.add_argument("--trajectory_ckpt", type=str, required=True,
                      help="LT3P trajectory model checkpoint")
    group.add_argument("--norm_stats", type=str, required=True,
                      help="Normalization statistics (.pt file)")
    group.add_argument("--track_csv", type=str, default="Trajectory/processed_typhoon_tracks.csv",
                      help="Typhoon track CSV")

    group = parser.add_argument_group("Model Checkpoints")
    group.add_argument("--fm_ckpt", type=str, default="",
                      help="Flow Matching checkpoint")
    group.add_argument("--dm_ckpt", type=str, default="",
                      help="Diffusion Model checkpoint")

    group = parser.add_argument_group("Pre-generated Samples (alternative to checkpoints)")
    group.add_argument("--dm_samples_dir", type=str, default="",
                      help="Directory with pre-generated DM samples (ar_pred_*.pt)")
    group.add_argument("--fm_samples_dir", type=str, default="",
                      help="Directory with pre-generated FM samples (ar_pred_*.pt)")

    group = parser.add_argument_group("Evaluation Settings")
    group.add_argument("--num_samples", type=int, default=20,
                      help="Ensemble size for Table 2/3 (default: 20)")
    group.add_argument("--ddim_steps", type=int, default=50,
                      help="DDIM steps for DM (default: 50)")
    group.add_argument("--euler_steps", type=int, default=4,
                      help="Euler steps for FM (default: 4)")
    group.add_argument("--device", type=str, default="cuda",
                      help="Device (cuda or cpu)")
    group.add_argument("--skip_steps", type=str, default="",
                      help="Comma-separated steps to skip (e.g., 'table3,nfe,spectral')")

    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print("="*80)
    print("PAPER EVALUATION PIPELINE — FM vs DM Typhoon Forecasting")
    print("="*80)
    print(f"Output: {args.output_dir}")
    print(f"Device: {args.device}")
    print(f"DM checkpoint: {args.dm_ckpt or args.dm_samples_dir or 'NOT SET'}")
    print(f"FM checkpoint: {args.fm_ckpt or args.fm_samples_dir or 'NOT SET'}")
    print(f"Trajectory checkpoint: {args.trajectory_ckpt}")
    print(f"Num samples: {args.num_samples}")
    print("="*80)

    skip_steps = set(args.skip_steps.split(",")) if args.skip_steps else set()
    results = {}

    if "table2" not in skip_steps:
        if args.dm_samples_dir or args.dm_ckpt:
            print("\n[Step 1] Table 2 — DM Ensemble Mean")
            diffusion_dir = args.dm_samples_dir or ""
            success, table2_dm_dir = run_table2_evaluation(
                output_dir=args.output_dir,
                trajectory_ckpt=args.trajectory_ckpt,
                norm_stats=args.norm_stats,
                track_csv=args.track_csv,
                diffusion_output_dir=diffusion_dir,
                method="dm",
                num_samples=args.num_samples,
                ddim_steps=args.ddim_steps,
            )
            results["table2_dm"] = {"success": success, "dir": table2_dm_dir}

        if args.fm_samples_dir or args.fm_ckpt:
            print("\n[Step 2] Table 2 — FM Deterministic")
            diffusion_dir = args.fm_samples_dir or ""
            success, table2_fm_dir = run_table2_evaluation(
                output_dir=args.output_dir,
                trajectory_ckpt=args.trajectory_ckpt,
                norm_stats=args.norm_stats,
                track_csv=args.track_csv,
                diffusion_output_dir=diffusion_dir,
                method="fm",
                num_samples=1,
                euler_steps=args.euler_steps,
            )
            results["table2_fm"] = {"success": success, "dir": table2_fm_dir}

    if "table3" not in skip_steps and args.dm_samples_dir:
        print("\n[Step 3] Table 3 — Oracle Best-of-N (DM)")
        success, table3_dm_dir = run_table3_evaluation(
            output_dir=args.output_dir,
            trajectory_ckpt=args.trajectory_ckpt,
            norm_stats=args.norm_stats,
            track_csv=args.track_csv,
            diffusion_output_dir=args.dm_samples_dir,
            method="dm",
            num_samples=args.num_samples,
            selection_strategy="per_lead_min",
        )
        results["table3_dm"] = {"success": success, "dir": table3_dm_dir}

    if "nfe" not in skip_steps:
        print("\n[Step 4] NFE Sweep")
        success, nfe_dir = run_nfe_sweep(
            output_dir=args.output_dir,
            fm_ckpt=args.fm_ckpt,
            dm_ckpt=args.dm_ckpt,
            device=args.device,
            num_samples=100,
        )
        results["nfe_sweep"] = {"success": success, "dir": nfe_dir}

    if "spectral" not in skip_steps:
        print("\n[Step 5] Spectral Fidelity")
        if args.dm_samples_dir:
            success, spectral_dm_dir = run_spectral_fidelity(
                output_dir=args.output_dir,
                pred_dir=args.dm_samples_dir,
                method="dm",
            )
            results["spectral_dm"] = {"success": success, "dir": spectral_dm_dir}
        if args.fm_samples_dir:
            success, spectral_fm_dir = run_spectral_fidelity(
                output_dir=args.output_dir,
                pred_dir=args.fm_samples_dir,
                method="fm",
            )
            results["spectral_fm"] = {"success": success, "dir": spectral_fm_dir}

    if "geostrophic" not in skip_steps:
        print("\n[Step 6] Geostrophic Balance")
        if args.dm_samples_dir:
            success, geo_dm_dir = run_geostrophic_evaluation(
                output_dir=args.output_dir,
                pred_dir=args.dm_samples_dir,
                method="dm",
            )
            results["geostrophic_dm"] = {"success": success, "dir": geo_dm_dir}
        if args.fm_samples_dir:
            success, geo_fm_dir = run_geostrophic_evaluation(
                output_dir=args.output_dir,
                pred_dir=args.fm_samples_dir,
                method="fm",
            )
            results["geostrophic_fm"] = {"success": success, "dir": geo_fm_dir}

    if "intensity" not in skip_steps:
        print("\n[Step 7] Intensity Evaluation")
        if args.dm_samples_dir:
            success, int_dm_dir = run_intensity_evaluation(
                output_dir=args.output_dir,
                pred_dir=args.dm_samples_dir,
                method="dm",
            )
            results["intensity_dm"] = {"success": success, "dir": int_dm_dir}
        if args.fm_samples_dir:
            success, int_fm_dir = run_intensity_evaluation(
                output_dir=args.output_dir,
                pred_dir=args.fm_samples_dir,
                method="fm",
            )
            results["intensity_fm"] = {"success": success, "dir": int_fm_dir}

    if "figures" not in skip_steps:
        print("\n[Step 8] Generate Paper Figures")
        table2_dm_dir = results.get("table2_dm", {}).get("dir", "")
        table2_fm_dir = results.get("table2_fm", {}).get("dir", "")
        nfe_dir = results.get("nfe_sweep", {}).get("dir", "")

        success, fig_dir = generate_all_figures(
            output_dir=args.output_dir,
            table2_dm_dir=table2_dm_dir,
            table2_fm_dir=table2_fm_dir,
            nfe_sweep_dir=nfe_dir,
        )
        results["figures"] = {"success": success, "dir": fig_dir}

    print("\n[Step 9] Compile Final Summary")
    compile_final_summary(args.output_dir)

    print("\n[Step 10] Print Final Report")
    print_final_report(args.output_dir)

    print("\n" + "="*80)
    print("EVALUATION PIPELINE COMPLETE")
    print("="*80)
    success_count = sum(1 for r in results.values() if r.get("success", False))
    print(f"Completed: {success_count}/{len(results)} steps")
    print(f"Results: {args.output_dir}")


if __name__ == "__main__":
    main()
