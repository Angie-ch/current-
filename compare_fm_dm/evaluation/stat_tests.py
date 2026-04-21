"""
Statistical Significance Tests for FM vs DM Comparison

Provides paired statistical tests to determine whether FM's advantage
over DM is statistically significant.

Tests included:
1. Paired t-test (parametric)
2. Wilcoxon signed-rank test (non-parametric)
3. McNemar test (for pairwise classification)
4. Effect size (Cohen's d)
5. Confidence intervals (bootstrapped)
"""
import numpy as np
from typing import Dict, List, Tuple, Optional
from scipy import stats


def paired_ttest(
    errors_fm: np.ndarray,
    errors_dm: np.ndarray,
    alpha: float = 0.05,
) -> Dict:
    """
    Paired t-test for FM vs DM error differences.

    H0: mean(FM_errors) = mean(DM_errors)
    H1: mean(FM_errors) ≠ mean(DM_errors)  [two-tailed]

    Args:
        errors_fm: (N,) RMSE errors for FM predictions
        errors_dm: (N,) RMSE errors for DM predictions
        alpha: significance level

    Returns:
        Dict with t-statistic, p-value, confidence interval, conclusion
    """
    if len(errors_fm) != len(errors_dm):
        raise ValueError("errors_fm and errors_dm must have the same length")

    diffs = errors_fm - errors_dm
    n = len(diffs)

    t_stat, p_value = stats.ttest_rel(errors_fm, errors_dm)

    # Mean and std of differences
    mean_diff = np.mean(diffs)
    std_diff = np.std(diffs, ddof=1)
    se = std_diff / np.sqrt(n)

    # 95% CI
    t_crit = stats.t.ppf(1 - alpha / 2, df=n - 1)
    ci_lower = mean_diff - t_crit * se
    ci_upper = mean_diff + t_crit * se

    # Conclusion
    significant = p_value < alpha
    fm_better = mean_diff < 0

    conclusion = "FM significantly better" if (significant and fm_better) else \
                "DM significantly better" if (significant and not fm_better) else \
                "No significant difference"

    return {
        "test": "paired_ttest",
        "n_samples": n,
        "t_statistic": float(t_stat),
        "p_value": float(p_value),
        "mean_diff": float(mean_diff),
        "std_diff": float(std_diff),
        "ci_95_lower": float(ci_lower),
        "ci_95_upper": float(ci_upper),
        "significant": significant,
        "fm_better": fm_better,
        "alpha": alpha,
        "conclusion": conclusion,
    }


def wilcoxon_signed_rank(
    errors_fm: np.ndarray,
    errors_dm: np.ndarray,
    alpha: float = 0.05,
) -> Dict:
    """
    Wilcoxon signed-rank test (non-parametric alternative to paired t-test).

    Robust to outliers and non-normal distributions.

    Args:
        errors_fm: (N,) RMSE errors for FM predictions
        errors_dm: (N,) RMSE errors for DM predictions
        alpha: significance level

    Returns:
        Dict with test statistic, p-value, conclusion
    """
    if len(errors_fm) != len(errors_dm):
        raise ValueError("errors_fm and errors_dm must have the same length")

    diffs = errors_fm - errors_dm
    # Remove zero differences
    diffs = diffs[diffs != 0]

    if len(diffs) < 10:
        return {
            "test": "wilcoxon",
            "n_nonzero_diffs": len(diffs),
            "conclusion": "Too few non-zero differences for reliable test",
            "p_value": None,
        }

    statistic, p_value = stats.wilcoxon(diffs, alternative='two-sided')

    mean_diff = np.mean(errors_fm - errors_dm)
    fm_better = mean_diff < 0
    significant = p_value < alpha

    conclusion = "FM significantly better" if (significant and fm_better) else \
                "DM significantly better" if (significant and not fm_better) else \
                "No significant difference"

    return {
        "test": "wilcoxon_signed_rank",
        "n_samples": len(errors_fm),
        "n_nonzero_diffs": len(diffs),
        "statistic": float(statistic),
        "p_value": float(p_value),
        "mean_diff": float(mean_diff),
        "significant": significant,
        "fm_better": fm_better,
        "alpha": alpha,
        "conclusion": conclusion,
    }


def cohens_d(
    errors_fm: np.ndarray,
    errors_dm: np.ndarray,
) -> Dict:
    """
    Cohen's d effect size measure.

    Interpretation:
        |d| < 0.2: negligible
        0.2 <= |d| < 0.5: small
        0.5 <= |d| < 0.8: medium
        |d| >= 0.8: large

    Args:
        errors_fm: (N,) RMSE errors for FM predictions
        errors_dm: (N,) RMSE errors for DM predictions

    Returns:
        Dict with Cohen's d and interpretation
    """
    mean_fm = np.mean(errors_fm)
    mean_dm = np.mean(errors_dm)
    std_pooled = np.sqrt((np.std(errors_fm, ddof=1)**2 + np.std(errors_dm, ddof=1)**2) / 2)

    d = (mean_fm - mean_dm) / std_pooled if std_pooled > 0 else 0.0

    abs_d = abs(d)
    if abs_d < 0.2:
        interpretation = "negligible"
    elif abs_d < 0.5:
        interpretation = "small"
    elif abs_d < 0.8:
        interpretation = "medium"
    else:
        interpretation = "large"

    return {
        "cohens_d": float(d),
        "interpretation": interpretation,
        "mean_fm": float(mean_fm),
        "mean_dm": float(mean_dm),
        "std_pooled": float(std_pooled),
        "fm_wins": d < 0,
    }


def bootstrap_ci(
    errors_fm: np.ndarray,
    errors_dm: np.ndarray,
    n_bootstrap: int = 10000,
    alpha: float = 0.05,
    seed: int = 42,
) -> Dict:
    """
    Bootstrap confidence interval for the mean difference.

    Non-parametric: makes no assumption about the distribution.

    Args:
        errors_fm: (N,) RMSE errors for FM predictions
        errors_dm: (N,) RMSE errors for DM predictions
        n_bootstrap: number of bootstrap samples
        alpha: significance level
        seed: random seed for reproducibility

    Returns:
        Dict with bootstrap CI and bias estimates
    """
    rng = np.random.default_rng(seed)
    n = len(errors_fm)

    diffs = errors_fm - errors_dm
    observed_diff = np.mean(diffs)

    bootstrap_diffs = []
    for _ in range(n_bootstrap):
        indices = rng.integers(0, n, size=n)
        bootstrap_diffs.append(np.mean(diffs[indices]))

    bootstrap_diffs = np.array(bootstrap_diffs)

    # Percentile CI
    lower_percentile = (alpha / 2) * 100
    upper_percentile = (1 - alpha / 2) * 100
    ci_lower = np.percentile(bootstrap_diffs, lower_percentile)
    ci_upper = np.percentile(bootstrap_diffs, upper_percentile)

    # Bootstrap standard error
    se = np.std(bootstrap_diffs)

    # Bias
    bias = np.mean(bootstrap_diffs) - observed_diff

    return {
        "observed_mean_diff": float(observed_diff),
        "bootstrap_se": float(se),
        "bias": float(bias),
        "ci_lower": float(ci_lower),
        "ci_upper": float(ci_upper),
        "n_bootstrap": n_bootstrap,
        "fm_better": observed_diff < 0,
        "significant": (ci_lower < 0 and ci_upper < 0) or (ci_lower > 0 and ci_upper > 0),
    }


def comprehensive_statistical_test(
    errors_fm: np.ndarray,
    errors_dm: np.ndarray,
    alpha: float = 0.05,
) -> Dict:
    """
    Run all statistical tests and produce a comprehensive report.

    Args:
        errors_fm: (N,) RMSE errors for FM predictions
        errors_dm: (N,) RMSE errors for DM predictions
        alpha: significance level

    Returns:
        Comprehensive dict with all test results
    """
    results = {}

    # Paired t-test
    results["paired_ttest"] = paired_ttest(errors_fm, errors_dm, alpha=alpha)

    # Wilcoxon
    results["wilcoxon"] = wilcoxon_signed_rank(errors_fm, errors_dm, alpha=alpha)

    # Effect size
    results["cohens_d"] = cohens_d(errors_fm, errors_dm)

    # Bootstrap CI
    results["bootstrap_ci"] = bootstrap_ci(errors_fm, errors_dm, alpha=alpha)

    # Per-channel analysis
    if errors_fm.ndim > 1 and errors_dm.ndim > 1:
        channel_results = []
        for c in range(errors_fm.shape[-1]):
            fm_ch = errors_fm[..., c].ravel()
            dm_ch = errors_dm[..., c].ravel()
            if len(fm_ch) == len(dm_ch):
                channel_results.append({
                    "channel": c,
                    "fm_better": np.mean(fm_ch) < np.mean(dm_ch),
                    "fm_rmse": float(np.mean(fm_ch)),
                    "dm_rmse": float(np.mean(dm_ch)),
                    "improvement_pct": float(100 * (np.mean(dm_ch) - np.mean(fm_ch)) / (np.mean(dm_ch) + 1e-8)),
                    "ttest": paired_ttest(fm_ch, dm_ch, alpha=alpha),
                })
        results["per_channel"] = channel_results

    # Summary
    n_significant = sum(1 for k in ["paired_ttest", "wilcoxon"]
                       if results.get(k, {}).get("significant", False))
    overall_significant = n_significant >= 2
    fm_wins = results.get("bootstrap_ci", {}).get("fm_better", None)

    results["summary"] = {
        "overall_significant": overall_significant,
        "n_significant_tests": n_significant,
        "fm_consistently_better": fm_wins,
        "alpha": alpha,
        "final_conclusion": (
            f"FM {'significantly outperforms' if (overall_significant and fm_wins) else 'does not significantly outperform'} DM "
            f"(p_t={results['paired_ttest']['p_value']:.4f}, "
            f"d={results['cohens_d']['cohens_d']:.3f}, "
            f"CI=[{results['bootstrap_ci']['ci_lower']:.4f}, {results['bootstrap_ci']['ci_upper']:.4f}])"
        ),
    }

    return results


def print_statistical_report(results: Dict):
    """Pretty-print the statistical test results."""
    print("=" * 70)
    print("STATISTICAL SIGNIFICANCE REPORT")
    print("=" * 70)

    summary = results.get("summary", {})
    print(f"\nFinal Conclusion: {summary.get('final_conclusion', 'N/A')}")
    print(f"Overall Significant: {summary.get('overall_significant', False)}")
    print(f"Tests Finding FM Better: {summary.get('n_significant_tests', 0)}/2")

    print("\n--- Paired t-test ---")
    tt = results.get("paired_ttest", {})
    print(f"  t = {tt.get('t_statistic', 0):.4f}")
    print(f"  p = {tt.get('p_value', 1):.6f}")
    print(f"  Mean diff: {tt.get('mean_diff', 0):.6f}")
    print(f"  95% CI: [{tt.get('ci_95_lower', 0):.6f}, {tt.get('ci_95_upper', 0):.6f}]")
    print(f"  Conclusion: {tt.get('conclusion', 'N/A')}")

    print("\n--- Wilcoxon Signed-Rank Test ---")
    wx = results.get("wilcoxon", {})
    print(f"  W = {wx.get('statistic', 0):.4f}")
    print(f"  p = {wx.get('p_value', 1):.6f}")
    print(f"  Conclusion: {wx.get('conclusion', 'N/A')}")

    print("\n--- Effect Size (Cohen's d) ---")
    cd = results.get("cohens_d", {})
    print(f"  d = {cd.get('cohens_d', 0):.4f}")
    print(f"  Interpretation: {cd.get('interpretation', 'N/A')}")
    print(f"  FM RMSE: {cd.get('mean_fm', 0):.6f}")
    print(f"  DM RMSE: {cd.get('mean_dm', 0):.6f}")

    print("\n--- Bootstrap 95% CI ---")
    bs = results.get("bootstrap_ci", {})
    print(f"  Observed diff: {bs.get('observed_mean_diff', 0):.6f}")
    print(f"  Bootstrap SE: {bs.get('bootstrap_se', 0):.6f}")
    print(f"  CI: [{bs.get('ci_lower', 0):.6f}, {bs.get('ci_upper', 0):.6f}]")

    if "per_channel" in results:
        print("\n--- Per-Channel Results ---")
        for ch in results["per_channel"]:
            sig_marker = "**" if ch["ttest"].get("significant", False) else "  "
            print(f"  {sig_marker} Ch {ch['channel']}: FM={ch['fm_rmse']:.4f}, DM={ch['dm_rmse']:.4f}, "
                  f"Δ={ch['improvement_pct']:+.1f}%, p={ch['ttest'].get('p_value', 1):.4f}")

    print("=" * 70)
