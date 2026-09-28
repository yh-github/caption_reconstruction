"""
src/shared_target/stats.py

Statistical inference for paired shared-target evaluation:
- Within-video paired difference aggregation
- Stratified cluster bootstrap over videos (B=10000)
- Cluster bootstrap over channels (robustness check)
- Holm-Bonferroni primary family multiplicity correction
- Outcome classification under equivalence margin (delta=0.05)
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Sequence
import numpy as np
import pandas as pd


@dataclass
class PairedTestResult:
    test_name: str
    mean_diff: float
    ci_lower: float
    ci_upper: float
    std_err: float
    p_value: float
    p_adjusted: float = 1.0
    outcome: str = "inconclusive"  # 'superior', 'inferior', 'equivalent', 'inconclusive'


def classify_outcome(
    ci_lower: float,
    ci_upper: float,
    equivalence_margin: float = 0.05
) -> str:
    """
    Classifies outcome relative to equivalence margin delta:
    - superior: ci_lower > 0
    - inferior: ci_upper < 0
    - equivalent: ci_lower >= -delta and ci_upper <= delta
    - inconclusive: otherwise
    """
    if ci_lower > 0.0:
        return "superior"
    elif ci_upper < 0.0:
        return "inferior"
    elif ci_lower >= -equivalence_margin and ci_upper <= equivalence_margin:
        return "equivalent"
    else:
        return "inconclusive"


def stratified_cluster_bootstrap(
    video_diffs: pd.DataFrame,
    diff_col: str = "diff",
    domain_col: str = "category",
    b_resamples: int = 10000,
    alpha: float = 0.05,
    seed: int = 42
) -> tuple[float, float, float, float, float]:
    """
    Performs stratified cluster bootstrap over videos.
    Returns: (mean_diff, ci_lower, ci_upper, std_err, p_value)
    """
    rng = np.random.default_rng(seed)
    
    # Stratified by domain
    domains = video_diffs[domain_col].unique()
    domain_groups = {d: video_diffs[video_diffs[domain_col] == d][diff_col].values for d in domains}
    
    observed_mean = float(video_diffs[diff_col].mean())
    
    boot_means = np.empty(b_resamples, dtype=np.float64)
    for b in range(b_resamples):
        resampled_vals = []
        for d, vals in domain_groups.items():
            n = len(vals)
            if n > 0:
                idx = rng.integers(0, n, size=n)
                resampled_vals.append(vals[idx])
        all_resampled = np.concatenate(resampled_vals)
        boot_means[b] = np.mean(all_resampled)
        
    ci_lower = float(np.percentile(boot_means, 100.0 * (alpha / 2.0)))
    ci_upper = float(np.percentile(boot_means, 100.0 * (1.0 - alpha / 2.0)))
    std_err = float(np.std(boot_means, ddof=1))
    
    # Two-tailed bootstrap p-value
    # H0: mean = 0
    # Center bootstrap distribution at 0
    centered = boot_means - observed_mean
    p_val = float(np.mean(np.abs(centered) >= np.abs(observed_mean)))
    # Clip p-value to [1 / b_resamples, 1.0]
    p_val = max(p_val, 1.0 / float(b_resamples))
    
    return observed_mean, ci_lower, ci_upper, std_err, p_val


def channel_cluster_bootstrap(
    video_diffs: pd.DataFrame,
    diff_col: str = "diff",
    channel_col: str = "channel",
    b_resamples: int = 10000,
    alpha: float = 0.05,
    seed: int = 42
) -> tuple[float, float, float, float]:
    """
    Robustness check: Cluster bootstrap resampling YouTube channels with replacement.
    Returns: (mean_diff, ci_lower, ci_upper, std_err)
    """
    rng = np.random.default_rng(seed)
    channels = video_diffs[channel_col].unique()
    channel_dict = {c: video_diffs[video_diffs[channel_col] == c][diff_col].values for c in channels}
    
    observed_mean = float(video_diffs[diff_col].mean())
    n_channels = len(channels)
    
    boot_means = np.empty(b_resamples, dtype=np.float64)
    for b in range(b_resamples):
        sampled_c = rng.choice(channels, size=n_channels, replace=True)
        vals = np.concatenate([channel_dict[c] for c in sampled_c])
        boot_means[b] = np.mean(vals)
        
    ci_lower = float(np.percentile(boot_means, 100.0 * (alpha / 2.0)))
    ci_upper = float(np.percentile(boot_means, 100.0 * (1.0 - alpha / 2.0)))
    std_err = float(np.std(boot_means, ddof=1))
    
    return observed_mean, ci_lower, ci_upper, std_err


def holm_bonferroni_correction(results: list[PairedTestResult]) -> list[PairedTestResult]:
    """
    Applies step-down Holm-Bonferroni multiplicity correction to primary confirmatory family.
    """
    m = len(results)
    if m == 0:
        return results
        
    # Sort by unadjusted p-value
    indexed = sorted(enumerate(results), key=lambda x: x[1].p_value)
    
    adj_p = [0.0] * m
    running_max = 0.0
    for rank, (orig_idx, res) in enumerate(indexed):
        multiplier = m - rank
        raw_adj = multiplier * res.p_value
        running_max = max(running_max, raw_adj)
        adj_p[orig_idx] = min(1.0, running_max)
        
    for idx, res in enumerate(results):
        res.p_adjusted = adj_p[idx]
        
    return results
