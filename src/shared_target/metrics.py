"""
src/shared_target/metrics.py

Metrics for shared-target retrieval evaluation:
- Mid-rank with exact tie-breaking
- Calibrated score: c = (N + 1 - 2r) / (N - 1) = 2*AUC - 1
- MRR: 1 / r
- Top-k retrieval indicators
- Headline score: 0.5 * (c_near + c_far)
"""

from __future__ import annotations
import numpy as np


def compute_mid_rank(
    target_score: float,
    distractor_scores: np.ndarray | list[float],
    tol: float = 1e-9
) -> tuple[float, bool]:
    """
    Computes the mid-rank of the target among candidates (higher similarity = lower rank = better).
    Rank 1 is the best possible rank.
    
    Args:
        target_score: similarity of the query prediction to the target candidate
        distractor_scores: similarities of the query prediction to distractor candidates
        tol: absolute tolerance for detecting score ties
        
    Returns:
        (mid_rank, has_tie):
            mid_rank: 1.0 + n_greater + 0.5 * n_equal
            has_tie: True if target tied with at least one distractor
    """
    d_scores = np.asarray(distractor_scores, dtype=np.float64)
    if len(d_scores) == 0:
        return 1.0, False
        
    diffs = d_scores - target_score
    n_greater = int(np.sum(diffs > tol))
    n_equal = int(np.sum(np.abs(diffs) <= tol))
    
    mid_rank = 1.0 + n_greater + 0.5 * n_equal
    has_tie = n_equal > 0
    return float(mid_rank), bool(has_tie)


def compute_calibrated_score(mid_rank: float, pool_size: int) -> float:
    """
    Calibrates retrieval rank to chance:
        c = (N + 1 - 2r) / (N - 1) = 2 * AUC - 1
        
    Properties:
        r = 1 (perfect)    -> c = +1.0
        r = (N+1)/2 (chance) -> c = 0.0
        r = N (worst)      -> c = -1.0
        
    Args:
        mid_rank: mid-rank of the target (1 <= mid_rank <= pool_size)
        pool_size: total candidates in the pool (N = 1 target + distractors)
    """
    if pool_size <= 1:
        return 0.0
    c = (pool_size + 1.0 - 2.0 * mid_rank) / (pool_size - 1.0)
    # Clamp to [-1.0, 1.0] to guard against tiny float precision drift
    return float(np.clip(c, -1.0, 1.0))


def compute_mrr(mid_rank: float) -> float:
    """Computes Mean Reciprocal Rank from mid-rank: 1 / r."""
    if mid_rank <= 0.0:
        return 0.0
    return float(1.0 / mid_rank)


def compute_top_k(mid_rank: float, k: int = 1) -> float:
    """
    Top-k retrieval indicator.
    Returns 1.0 if target mid-rank <= k, else 0.0.
    """
    return 1.0 if mid_rank <= float(k) else 0.0


def compute_headline_score(c_near: float, c_far: float) -> float:
    """
    Computes headline score:
        c_headline = 0.5 * (c_near + c_far)
    """
    return 0.5 * (c_near + c_far)
