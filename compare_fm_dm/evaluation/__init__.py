"""
评估包 — Publication-ready evaluation for FM vs DM comparison
"""
from .metrics import (
    ComparisonEvaluator,
    compute_rmse, compute_mae, compute_lat_weighted_rmse,
    compute_acc, compute_channel_bias,
    compute_2d_psd, compute_kinetic_energy_spectrum,
    compute_spectral_slope,
    compute_divergence, compute_divergence_rmse,
    compute_geostrophic_balance,
    compute_temporal_coherence,
    compute_nfe_efficiency,
)

from .compute_climatology import (
    ClimatologyComputer,
    quick_compute_climatology,
)

from .crps_metric import (
    compute_crps_ensemble,
    compute_crps_vectorized,
    compute_crps_spatial,
    compute_crps_per_channel,
    compute_spread_skill_ratio,
    compute_reliability_diagram,
    compute_ensemble_entropy,
)

from .path_straightness import (
    compute_path_straightness_fm,
    compute_diffusion_path_curvature,
    visualize_path_comparison,
    compute_path_curvature_batch,
)

from .baselines import (
    persistence_forecast,
    climatology_forecast,
    linear_trend_forecast,
    BaselineForecaster,
)

from .stat_tests import (
    paired_ttest,
    wilcoxon_signed_rank,
    cohens_d,
    bootstrap_ci,
    comprehensive_statistical_test,
    print_statistical_report,
)

from .spatial_metrics import (
    compute_pattern_correlation,
    compute_sedi,
    compute_fss,
    compute_all_spatial_metrics,
)

from .publication_pipeline import (
    PublicationPipeline,
    quick_run_full_pipeline,
)

from .spectral_fidelity import (
    compute_radial_psd_torch,
    fit_power_law,
    compute_high_freq_ratio,
    spectral_analysis_single_field,
    aggregate_spectral_results,
    run_spectral_eval_from_predictions,
    generate_spectral_comparison_table,
)

from .geostrophic import (
    compute_geostrophic_wind_from_z,
    compute_geostrophic_imbalance_torch,
    compute_divergence_torch,
    geostrophic_eval_from_predictions_torch,
    aggregate_geostrophic_results,
    generate_geostrophic_comparison_table,
)

from .intensity import (
    z_to_pressure,
    find_typhoon_center,
    compute_intensity_metrics_single_case,
    intensity_eval_from_predictions,
    aggregate_intensity_results,
)
