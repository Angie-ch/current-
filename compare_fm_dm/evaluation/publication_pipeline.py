"""
Comprehensive Evaluation Pipeline — Publication-Ready FM vs DM Comparison

This module orchestrates the complete evaluation pipeline integrating:
1. Deterministic metrics (RMSE, MAE, ACC, Bias)
2. Spectral analysis (PSD, E(k), spectral slope)
3. Physics consistency (divergence, geostrophic balance)
4. Ensemble evaluation (CRPS, spread-skill, reliability)
5. Path straightness analysis
6. Baseline comparisons (persistence, climatology)
7. Statistical significance tests
8. Spatial structure metrics (SEDI, FSS, pattern correlation)
9. Multi-seed evaluation

Usage:
    pipeline = PublicationPipeline(configs)
    results = pipeline.run_full_evaluation()
    pipeline.generate_all_figures()
"""
import os
import json
import copy
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import torch
from tqdm import tqdm

logger = logging.getLogger(__name__)

from .metrics import (
    ComparisonEvaluator,
    compute_rmse, compute_mae, compute_lat_weighted_rmse,
    compute_acc, compute_channel_bias,
    compute_2d_psd, compute_kinetic_energy_spectrum,
    compute_spectral_slope,
    compute_divergence, compute_divergence_rmse,
    compute_geostrophic_balance, compute_vorticity,
    compute_temporal_coherence,
    compute_nfe_efficiency,
)
from .compute_climatology import ClimatologyComputer, quick_compute_climatology
from .crps_metric import (
    compute_crps_vectorized, compute_spread_skill_ratio,
    compute_reliability_diagram, compute_crps_per_channel,
)
from .path_straightness import (
    compute_path_straightness_fm, compute_diffusion_path_curvature,
    visualize_path_comparison, compute_path_curvature_batch,
)
from .baselines import BaselineForecaster
from .stat_tests import comprehensive_statistical_test, print_statistical_report
from .spatial_metrics import compute_all_spatial_metrics

# Visualization - 使用绝对导入
from compare_fm_dm.visualization.plots import (
    plot_psd_comparison, plot_nfe_efficiency_curve,
    plot_channel_rmse_comparison, plot_physics_consistency,
    plot_summary_table, visualize_comparison_results,
)
from compare_fm_dm.visualization.typhoon_case_study import TyphoonCaseStudy


class PublicationPipeline:
    """
    Full publication-ready evaluation pipeline for FM vs DM comparison.

    This pipeline generates the complete set of metrics and figures
    needed for a peer-reviewed publication.
    """

    def __init__(
        self,
        data_cfg,
        model_cfg,
        train_cfg,
        infer_cfg,
        fm_model=None,
        dm_model=None,
        work_dir: str = ".",
        device: str = "cuda",
    ):
        self.data_cfg = data_cfg
        self.model_cfg = model_cfg
        self.train_cfg = train_cfg
        self.infer_cfg = infer_cfg
        self.fm_model = fm_model
        self.dm_model = dm_model
        self.work_dir = work_dir
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        self.results = {}
        self.climatology = None
        self.baseline_forecaster = None

        os.makedirs(work_dir, exist_ok=True)
        os.makedirs(os.path.join(work_dir, "figures"), exist_ok=True)
        os.makedirs(os.path.join(work_dir, "tables"), exist_ok=True)
        os.makedirs(os.path.join(work_dir, "stats"), exist_ok=True)

    def compute_climatology(self, dataset, typhoon_ids: List[str]):
        """Step 1: Compute climatological mean for ACC."""
        logger.info("Computing climatological mean...")
        clim_path = os.path.join(self.work_dir, "climatology.npy")

        if os.path.exists(clim_path):
            self.climatology = np.load(clim_path)
            logger.info(f"Loaded existing climatology: {clim_path}")
        else:
            computer = ClimatologyComputer(self.data_cfg)
            result = computer.compute_from_dataset(
                dataset, typhoon_ids, save_path=clim_path
            )
            self.climatology = result["overall"]
            logger.info(f"Climatology computed and saved: {clim_path}")

        # Setup baseline forecaster
        self.baseline_forecaster = BaselineForecaster(
            climatology_mean=self.climatology,
            device=self.device,
        )

    def evaluate_deterministic(
        self,
        predictions: List[torch.Tensor],
        ground_truth: List[torch.Tensor],
        method_name: str,
    ) -> Dict:
        """Step 2: Deterministic accuracy metrics."""
        logger.info(f"Evaluating deterministic metrics for {method_name}...")

        evaluator = ComparisonEvaluator(
            data_cfg=self.data_cfg,
            device=self.device,
            norm_mean=None,  # Already in physical units
            norm_std=None,
        )

        results = evaluator.evaluate_single(predictions, ground_truth, method_name)

        # ACC requires climatology
        if self.climatology is not None:
            clim_t = torch.from_numpy(self.climatology).float().to(self.device)
            all_preds = torch.cat([p.cpu() for p in predictions], dim=0)
            all_gts = torch.cat([t.cpu() for t in ground_truth], dim=0)

            acc = compute_acc(all_preds, all_gts, clim_t)
            results["acc_per_channel"] = {
                self.data_cfg.var_names[i]: float(acc[i])
                for i in range(len(self.data_cfg.var_names))
            }
            results["acc_mean"] = float(acc.mean().item())

        return results

    def evaluate_spectral(
        self,
        predictions: List[torch.Tensor],
        ground_truth: List[torch.Tensor],
        method_name: str,
    ) -> Dict:
        """Step 3: Spectral analysis."""
        logger.info(f"Evaluating spectral metrics for {method_name}...")

        all_preds = torch.cat([p.cpu() for p in predictions], dim=0)
        all_gts = torch.cat([t.cpu() for t in ground_truth], dim=0)

        results = {}

        # PSD
        k, psd_pred = compute_2d_psd(all_preds)
        _, psd_gt = compute_2d_psd(all_gts)

        results["k"] = k.tolist()
        results["psd_pred"] = psd_pred.tolist()
        results["psd_gt"] = psd_gt.tolist()

        # Spectral slope
        slope_pred = compute_spectral_slope(k, psd_pred)
        slope_gt = compute_spectral_slope(k, psd_gt)
        results["spectral_slope_pred"] = float(slope_pred)
        results["spectral_slope_gt"] = float(slope_gt)

        # High-frequency energy ratio
        hf_mask = k > 0.2
        if hf_mask.any():
            results["high_freq_ratio_pred"] = float(psd_pred[hf_mask].sum() / (psd_pred.sum() + 1e-10))
            results["high_freq_ratio_gt"] = float(psd_gt[hf_mask].sum() / (psd_gt.sum() + 1e-10))

        logger.info(f"  {method_name} spectral slope: {slope_pred:.3f} (GT: {slope_gt:.3f})")

        return results

    def evaluate_physics(
        self,
        predictions: List[torch.Tensor],
        ground_truth: List[torch.Tensor],
        method_name: str,
    ) -> Dict:
        """Step 4: Physics consistency."""
        logger.info(f"Evaluating physics consistency for {method_name}...")

        all_preds = torch.cat([p.cpu() for p in predictions], dim=0)
        all_gts = torch.cat([t.cpu() for t in ground_truth], dim=0)

        results = {}

        # Divergence RMSE
        pred_div, gt_div = compute_divergence_rmse(
            all_preds, all_gts,
            u_channel=0, v_channel=3,
        )
        results["divergence_rmse_pred"] = float(pred_div)
        results["divergence_rmse_gt"] = float(gt_div)

        # Geostrophic balance
        if self.data_cfg.num_channels >= 9:
            u_pred = all_preds[:, 0]
            v_pred = all_preds[:, 3]
            z_pred = all_preds[:, 6]

            geo_u, geo_v = compute_geostrophic_balance(u_pred, v_pred, z_pred)
            results["geostrophic_residual_u"] = float(geo_u.item())
            results["geostrophic_residual_v"] = float(geo_v.item())

        logger.info(f"  {method_name} divergence RMSE: {pred_div:.4f} (GT: {gt_div:.4f})")

        return results

    def evaluate_ensemble(
        self,
        ensemble_preds: List[np.ndarray],  # (K, C, H, W) per time step
        ground_truth: List[torch.Tensor],
        method_name: str,
    ) -> Dict:
        """Step 5: Ensemble/probabilistic metrics."""
        logger.info(f"Evaluating ensemble metrics for {method_name}...")

        results = {}

        all_crps = []
        all_spread_skill = []

        for ens, gt in zip(ensemble_preds, ground_truth):
            if isinstance(gt, torch.Tensor):
                gt = gt.cpu().numpy()

            crps = compute_crps_vectorized(ens, gt)
            all_crps.append(crps.mean())

            if ens.shape[0] > 1:
                ssr = compute_spread_skill_ratio(ens, gt)
                all_spread_skill.append(ssr)

        results["crps_mean"] = float(np.mean(all_crps))
        results["crps_std"] = float(np.std(all_crps))

        if all_spread_skill:
            results["spread_skill_ratio_mean"] = float(np.mean(all_spread_skill))
            results["spread_skill_ratio_std"] = float(np.std(all_spread_skill))

        # CRPS per channel
        ens_mean = np.stack([e.mean(axis=0) for e in ensemble_preds], axis=0).mean(axis=0)
        gt_mean = np.stack([g.cpu().numpy() if isinstance(g, torch.Tensor) else g
                           for g in ground_truth], axis=0).mean(axis=0)

        crps_per_ch = compute_crps_per_channel(
            np.stack([e.mean(axis=0) for e in ensemble_preds], axis=0),
            gt_mean,
            self.data_cfg.var_names,
        )
        results["crps_per_channel"] = crps_per_ch

        logger.info(f"  {method_name} CRPS: {results['crps_mean']:.4f} ± {results['crps_std']:.4f}")
        if all_spread_skill:
            logger.info(f"  {method_name} spread-skill ratio: {results['spread_skill_ratio_mean']:.4f}")

        return results

    def evaluate_path_straightness(
        self,
        model,
        dataloader,
        method: str,
        n_samples: int = 50,
    ) -> Dict:
        """Step 6: Path straightness/curvature analysis."""
        logger.info(f"Evaluating path straightness for {method}...")

        result = compute_path_curvature_batch(
            model, dataloader,
            n_samples=n_samples,
            method=method,
            device=self.device,
        )

        logger.info(f"  {method} straightness: {result['straightness_mean']:.4f} ± {result['straightness_std']:.4f}")

        return result

    def evaluate_baselines(
        self,
        ground_truth: List[torch.Tensor],
        conditions: List[torch.Tensor],
    ) -> Dict:
        """Step 7: Baseline comparisons."""
        logger.info("Evaluating baseline methods...")

        if self.baseline_forecaster is None:
            logger.warning("Climatology not computed. Skipping baselines.")
            return {}

        results = self.baseline_forecaster.evaluate_baselines(
            ground_truth, conditions,
            climatology_mean=self.climatology,
        )

        for method, metrics in results.items():
            logger.info(f"  {method} RMSE: {metrics['rmse_mean']:.4f}")

        return results

    def evaluate_spatial(
        self,
        predictions: List[torch.Tensor],
        ground_truth: List[torch.Tensor],
        method_name: str,
    ) -> Dict:
        """Step 8: Spatial structure metrics."""
        logger.info(f"Evaluating spatial metrics for {method_name}...")

        all_preds = torch.cat([p.cpu() for p in predictions], dim=0)
        all_gts = torch.cat([t.cpu() for t in ground_truth], dim=0)

        results = compute_all_spatial_metrics(
            all_preds, all_gts,
            channel_names=self.data_cfg.var_names,
            device="cpu",
        )

        logger.info(f"  {method_name} pattern correlation: {results['pattern_correlation_mean']:.4f}")
        logger.info(f"  {method_name} SEDI: {results['sedi_mean']:.4f}")

        return results

    def run_statistical_tests(
        self,
        fm_errors: np.ndarray,
        dm_errors: np.ndarray,
    ) -> Dict:
        """Step 9: Statistical significance tests."""
        logger.info("Running statistical significance tests...")

        results = comprehensive_statistical_test(fm_errors, dm_errors)
        print_statistical_report(results)

        return results

    def generate_all_figures(
        self,
        results: Dict,
        output_dir: str = None,
    ):
        """Generate all publication-ready figures."""
        if output_dir is None:
            output_dir = os.path.join(self.work_dir, "figures")

        os.makedirs(output_dir, exist_ok=True)

        logger.info("Generating publication figures...")

        # Figure 1: PSD comparison
        if "FM_spectral" in results or "DM_spectral" in results:
            plot_psd_comparison(
                results,
                os.path.join(output_dir, "fig1_psd_comparison.png"),
                title="Kinetic Energy Spectrum: Flow Matching vs Diffusion",
            )

        # Figure 2: NFE efficiency
        if "NFE_efficiency" in results:
            plot_nfe_efficiency_curve(
                results["NFE_efficiency"],
                os.path.join(output_dir, "fig2_nfe_efficiency.png"),
            )

        # Figure 3: Per-channel RMSE
        plot_channel_rmse_comparison(
            results,
            os.path.join(output_dir, "fig3_channel_rmse.png"),
            var_names=self.data_cfg.var_names,
        )

        # Figure 4: Physics consistency
        plot_physics_consistency(
            results,
            os.path.join(output_dir, "fig4_physics_consistency.png"),
        )

        # Figure 5: Summary table
        plot_summary_table(
            results,
            os.path.join(output_dir, "summary_table.png"),
        )

        # Figure 6: Path straightness (if available)
        if "FM_path" in results and "DM_path" in results:
            visualize_path_comparison(
                results["FM_path"].get("straightness_mean", 0.9),
                results["DM_path"].get("straightness_mean", 0.5),
                results["FM_path"].get("path_length_ratio_mean", 1.0),
                results["DM_path"].get("path_length_ratio_mean", 1.5),
                save_path=os.path.join(output_dir, "fig5_path_comparison.png"),
            )

        # Generate LaTeX tables
        self._generate_latex_tables(results, output_dir)

        logger.info(f"All figures saved to: {output_dir}")

    def _generate_latex_tables(self, results: Dict, output_dir: str):
        """Generate publication-ready LaTeX tables."""
        os.makedirs(os.path.join(output_dir, "..", "tables"), exist_ok=True)

        # Table 1: Main accuracy metrics
        self._write_table1(results, output_dir)

        # Table 2: Per-lead-time breakdown
        self._write_table2(results, output_dir)

        # Table 3: Statistical significance
        self._write_table3(results, output_dir)

    def _write_table1(self, results: Dict, output_dir: str):
        """Main accuracy comparison table."""
        lines = [
            "% Table 1: Deterministic Forecast Accuracy",
            "\\begin{table}[htbp]",
            "\\centering",
            "\\caption{Deterministic forecast accuracy comparison. Best results in bold.}",
            "\\begin{tabular}{lccccccc}",
            "\\hline",
            "\\textbf{Method} & \\textbf{RMSE} & \\textbf{Lat-W RMSE} & \\textbf{MAE} & \\textbf{ACC} & "
            "\\textbf{Div RMSE} & \\textbf{Spec Slope} & \\textbf{NFE} \\\\",
            "\\hline",
        ]

        for method, color in [("FM", "blue!20"), ("DM", "red!20")]:
            if method in results:
                r = results[method]
                rmse = r.get("rmse_mean", 0)
                lat_rmse = r.get("lat_weighted_rmse_mean", 0)
                mae = r.get("mae_per_channel", {})
                mae_mean = np.mean(list(mae.values())) if mae else 0
                acc = r.get("acc_mean", 0)
                div = r.get("divergence_rmse_pred", 0)
                nfe = results.get("NFE_efficiency", {}).get(method, {}).get(4, {}).get("rmse_mean", "N/A")
                spec = results.get(f"{method}_spectral", {}).get("spectral_slope_pred", 0)

                lines.append(
                    f"{method} & {rmse:.4f} & {lat_rmse:.4f} & {mae_mean:.4f} & "
                    f"{acc:.4f} & {div:.4f} & {spec:.3f} & {nfe} \\\\"
                )

        lines.extend([
            "\\hline",
            "\\end{tabular}",
            "\\end{table}",
        ])

        path = os.path.join(output_dir, "..", "tables", "table1_accuracy.tex")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            f.write('\n'.join(lines))
        logger.info(f"Table 1 saved: {path}")

    def _write_table2(self, results: Dict, output_dir: str):
        """Per-lead-time breakdown table."""
        lines = [
            "% Table 2: Per-Lead-Time RMSE (Hours)",
            "\\begin{table}[htbp]",
            "\\centering",
            "\\caption{RMSE (m/s) at different forecast lead times.}",
            "\\begin{tabular}{lcccccccc}",
            "\\hline",
            "\\textbf{Method} & \\textbf{+24h} & \\textbf{+48h} & \\textbf{+72h} & "
            "\\textbf{+96h} & \\textbf{+120h} & \\textbf{+144h} & \\textbf{+168h} \\\\",
            "\\hline",
        ]

        for method in ["FM", "DM"]:
            if method not in results:
                continue
            # Extract per-lead-time from results if available
            row = [method]
            for lt in [24, 48, 72, 96, 120, 144, 168]:
                # Placeholder — would need per-lead-time tracking
                row.append("-")
            lines.append(" & ".join(row) + " \\\\")

        lines.extend(["\\hline", "\\end{tabular}", "\\end{table}"])

        path = os.path.join(output_dir, "..", "tables", "table2_leadtime.tex")
        with open(path, 'w') as f:
            f.write('\n'.join(lines))
        logger.info(f"Table 2 saved: {path}")

    def _write_table3(self, results: Dict, output_dir: str):
        """Statistical significance table."""
        if "statistical_tests" not in results:
            return

        st = results["statistical_tests"]
        tt = st.get("paired_ttest", {})
        cd = st.get("cohens_d", {})

        lines = [
            "% Table 3: Statistical Significance Tests",
            "\\begin{table}[htbp]",
            "\\centering",
            "\\caption{Statistical significance of FM vs DM difference.}",
            "\\begin{tabular}{lcccccc}",
            "\\hline",
            "\\textbf{Test} & \\textbf{Statistic} & \\textbf{p-value} & "
            "\\textbf{Mean Diff} & \\textbf{95\\% CI} & \\textbf{Cohen's d} & \\textbf{Conclusion} \\\\",
            "\\hline",
            f"Paired t-test & t={tt.get('t_statistic', 0):.3f} & p={tt.get('p_value', 1):.4f} & "
            f"{tt.get('mean_diff', 0):+.4f} & [{tt.get('ci_95_lower', 0):.4f}, {tt.get('ci_95_upper', 0):.4f}] & "
            f"{cd.get('cohens_d', 0):.3f} & {st.get('summary', {}).get('final_conclusion', '')[:30]} \\\\",
            f"Wilcoxon & W={st.get('wilcoxon', {}).get('statistic', 0):.1f} & "
            f"p={st.get('wilcoxon', {}).get('p_value', 1):.4f} & - & - & - & - \\\\",
            "\\hline",
            "\\end{tabular}",
            "\\end{table}",
        ]

        path = os.path.join(output_dir, "..", "tables", "table3_statistics.tex")
        with open(path, 'w') as f:
            f.write('\n'.join(lines))
        logger.info(f"Table 3 saved: {path}")

    def save_results(self, results: Dict, filename: str = "full_results.json"):
        """Save all results to JSON."""
        path = os.path.join(self.work_dir, filename)

        # Convert numpy arrays to lists for JSON serialization
        def make_serializable(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, dict):
                return {k: make_serializable(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [make_serializable(v) for v in obj]
            return obj

        serializable_results = make_serializable(results)

        with open(path, 'w') as f:
            json.dump(serializable_results, f, indent=2)

        logger.info(f"Results saved: {path}")

    def run_full_evaluation(
        self,
        fm_predictions: List[torch.Tensor],
        dm_predictions: List[torch.Tensor],
        ground_truth: List[torch.Tensor],
        fm_ensemble: Optional[List[np.ndarray]] = None,
        dm_ensemble: Optional[List[np.ndarray]] = None,
        fm_dataloader=None,
        dm_dataloader=None,
    ) -> Dict:
        """
        Run the complete evaluation pipeline.

        Args:
            fm_predictions: List of (B, C, H, W) FM predictions per time step
            dm_predictions: List of (B, C, H, W) DM predictions per time step
            ground_truth: List of (B, C, H, W) ground truth per time step
            fm_ensemble: Optional list of (K, C, H, W) ensemble arrays
            dm_ensemble: Optional list of (K, C, H, W) ensemble arrays
            fm_dataloader: DataLoader for FM path straightness evaluation
            dm_dataloader: DataLoader for DM path straightness evaluation

        Returns:
            Complete results dictionary
        """
        results = {}

        # 1. Deterministic metrics
        if fm_predictions:
            results["FM"] = self.evaluate_deterministic(
                fm_predictions, ground_truth, "FM"
            )
        if dm_predictions:
            results["DM"] = self.evaluate_deterministic(
                dm_predictions, ground_truth, "DM"
            )

        # 2. Spectral analysis
        if fm_predictions:
            results["FM_spectral"] = self.evaluate_spectral(
                fm_predictions, ground_truth, "FM"
            )
        if dm_predictions:
            results["DM_spectral"] = self.evaluate_spectral(
                dm_predictions, ground_truth, "DM"
            )

        # 3. Physics consistency
        if fm_predictions:
            results["FM_physics"] = self.evaluate_physics(
                fm_predictions, ground_truth, "FM"
            )
        if dm_predictions:
            results["DM_physics"] = self.evaluate_physics(
                dm_predictions, ground_truth, "DM"
            )

        # 4. Ensemble/probabilistic metrics
        if fm_ensemble:
            results["FM_ensemble"] = self.evaluate_ensemble(
                fm_ensemble, ground_truth, "FM"
            )
        if dm_ensemble:
            results["DM_ensemble"] = self.evaluate_ensemble(
                dm_ensemble, ground_truth, "DM"
            )

        # 5. Path straightness
        if fm_dataloader and self.fm_model:
            results["FM_path"] = self.evaluate_path_straightness(
                self.fm_model, fm_dataloader, "fm"
            )
        if dm_dataloader and self.dm_model:
            results["DM_path"] = self.evaluate_path_straightness(
                self.dm_model, dm_dataloader, "dm"
            )

        # 6. Spatial structure metrics
        if fm_predictions:
            results["FM_spatial"] = self.evaluate_spatial(
                fm_predictions, ground_truth, "FM"
            )
        if dm_predictions:
            results["DM_spatial"] = self.evaluate_spatial(
                dm_predictions, ground_truth, "DM"
            )

        # 7. Statistical tests (if both models evaluated)
        if "FM" in results and "DM" in results:
            fm_errors = np.array([
                results["FM"]["rmse_per_channel"].get(v, 0)
                for v in self.data_cfg.var_names
            ])
            dm_errors = np.array([
                results["DM"]["rmse_per_channel"].get(v, 0)
                for v in self.data_cfg.var_names
            ])
            results["statistical_tests"] = self.run_statistical_tests(fm_errors, dm_errors)

        # Save results
        self.save_results(results)
        self.results = results

        return results


def quick_run_full_pipeline(
    fm_predictions: List[torch.Tensor],
    dm_predictions: List[torch.Tensor],
    ground_truth: List[torch.Tensor],
    data_cfg,
    output_dir: str = "./pipeline_results",
    climatology: np.ndarray = None,
) -> Dict:
    """
    Quick runner for the full evaluation pipeline.

    Minimal setup — just pass predictions and get all metrics.

    Args:
        fm_predictions: List of FM prediction tensors
        dm_predictions: List of DM prediction tensors
        ground_truth: List of ground truth tensors
        data_cfg: DataConfig instance
        output_dir: Where to save results and figures
        climatology: Optional climatological mean for ACC

    Returns:
        Complete results dictionary
    """
    pipeline = PublicationPipeline(
        data_cfg=data_cfg,
        model_cfg=None,
        train_cfg=None,
        infer_cfg=None,
        work_dir=output_dir,
    )

    if climatology is not None:
        pipeline.climatology = climatology

    results = pipeline.run_full_evaluation(
        fm_predictions=fm_predictions,
        dm_predictions=dm_predictions,
        ground_truth=ground_truth,
    )

    pipeline.generate_all_figures(results)

    return results
