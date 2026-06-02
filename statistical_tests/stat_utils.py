"""
Core statistical utilities for hypothesis testing across the MAB framework.

All confidence intervals are frequentist and assume 30 seeds (CLT applies).
All hypothesis tests are non-parametric (Mann-Whitney U) since we cannot
assume normality of aggregated simulation metrics.

Multiple-comparison correction uses Holm-Bonferroni (uniformly more powerful
than Bonferroni, valid under arbitrary dependence).
"""

import numpy as np
from scipy import stats as sp_stats


def ci_95(samples):
    """Compute mean and 95% CI half-width via t-distribution.

    Uses t_{n-1, 0.025} * SE, which is exact for normal data and
    asymptotically correct for n >= 30 via CLT.

    Returns (mean, ci_half, se).
    """
    a = np.asarray(samples, dtype=float)
    n = len(a)
    if n < 2:
        return float(a[0]), 0.0, 0.0
    mean = float(a.mean())
    se = float(a.std(ddof=1) / np.sqrt(n))
    t_crit = sp_stats.t.ppf(0.975, df=n - 1)
    ci_half = t_crit * se
    return mean, ci_half, se


def mann_whitney(x, y, alternative="two-sided"):
    """Two-sample Mann-Whitney U test (non-parametric).

    Returns dict with U statistic, z-score, p-value, and rank-biserial
    effect size r = 1 - 2U/(n1*n2).

    |r| interpretation (Kerby 2014):
        < 0.10  negligible
        0.10-0.30  small
        0.30-0.50  medium
        > 0.50  large
    """
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    result = sp_stats.mannwhitneyu(x, y, alternative=alternative)
    n1, n2 = len(x), len(y)
    r = 1.0 - (2.0 * result.statistic) / (n1 * n2)
    # z-score via normal approximation
    mu_U = n1 * n2 / 2.0
    sigma_U = np.sqrt(n1 * n2 * (n1 + n2 + 1) / 12.0)
    z = (result.statistic - mu_U) / sigma_U if sigma_U > 0 else 0.0
    return {
        "U": float(result.statistic),
        "z": float(z),
        "p": float(result.pvalue),
        "r_effect": float(r),
        "n1": n1,
        "n2": n2,
    }


def kruskal_wallis(*groups):
    """Kruskal-Wallis H test (non-parametric one-way ANOVA).

    Use as an omnibus test before pairwise Mann-Whitney comparisons.
    Returns dict with H statistic, p-value, df.
    """
    result = sp_stats.kruskal(*groups)
    return {
        "H": float(result.statistic),
        "p": float(result.pvalue),
        "df": len(groups) - 1,
    }


def holm_bonferroni(p_values):
    """Holm-Bonferroni correction for multiple comparisons.

    Input: list of (label, p_value) tuples.
    Returns: list of (label, original_p, adjusted_p, significant_at_005) tuples,
    sorted by original p-value ascending.
    """
    m = len(p_values)
    sorted_pvals = sorted(p_values, key=lambda x: x[1])
    adjusted = []
    max_adj = 0.0
    for i, (label, p) in enumerate(sorted_pvals):
        adj = p * (m - i)
        adj = max(adj, max_adj)
        adj = min(adj, 1.0)
        max_adj = adj
        adjusted.append((label, float(p), float(adj), adj < 0.05))
    return adjusted


def pairwise_mann_whitney(groups_dict, metric_name="metric", alternative="two-sided"):
    """Run all pairwise Mann-Whitney U tests with Holm-Bonferroni correction.

    groups_dict: {group_name: array of per-seed values}
    Returns list of comparison dicts.
    """
    names = sorted(groups_dict.keys())
    raw_results = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            mw = mann_whitney(groups_dict[names[i]], groups_dict[names[j]],
                              alternative=alternative)
            label = f"{names[i]} vs {names[j]}"
            raw_results.append({"label": label, **mw})

    p_list = [(r["label"], r["p"]) for r in raw_results]
    corrected = holm_bonferroni(p_list)
    correction_map = {label: (adj_p, sig) for label, _, adj_p, sig in corrected}

    for r in raw_results:
        adj_p, sig = correction_map[r["label"]]
        r["p_adjusted"] = adj_p
        r["significant"] = sig
        r["metric"] = metric_name

    return raw_results


def format_p(p):
    """Format p-value for scientific reporting."""
    if p < 0.0001:
        return "p < 0.0001"
    elif p < 0.001:
        return f"p = {p:.4f}"
    elif p < 0.01:
        return f"p = {p:.3f}"
    else:
        return f"p = {p:.3f}"


def format_ci(mean, ci_half):
    """Format mean ± CI for tables."""
    return f"{mean:.2f} ± {ci_half:.2f}"


def describe_effect(r):
    """Verbal descriptor for rank-biserial effect size."""
    ar = abs(r)
    if ar < 0.10:
        return "negligible"
    elif ar < 0.30:
        return "small"
    elif ar < 0.50:
        return "medium"
    else:
        return "large"
