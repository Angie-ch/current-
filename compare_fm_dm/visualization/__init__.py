"""
可视化包 — Publication-ready figures for FM vs DM comparison
"""
from .plots import (
    plot_psd_comparison,
    plot_nfe_efficiency_curve,
    plot_channel_rmse_comparison,
    plot_lat_weighted_rmse_heatmap,
    plot_temporal_evolution,
    plot_physics_consistency,
    plot_summary_table,
    visualize_comparison_results,
)

from .typhoon_case_study import (
    TyphoonCaseStudy,
    plot_typhoon_case_study,
    plot_typhoon_intensity_evolution,
    find_typhoon_center,
)
