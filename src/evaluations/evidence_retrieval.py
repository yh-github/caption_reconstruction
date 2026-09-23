from __future__ import annotations
import logging
from typing import Any
import numpy as np

logger = logging.getLogger(__name__)


def calculate_evidence_retrieval_metrics(
    query_vector: np.ndarray,
    index_vectors: np.ndarray,
    target_indices: set[int] | list[int],
    top_k: int = 5
) -> dict[str, Any]:
    """
    Evaluates how effectively a query vector retrieves the target evidence frames
    from a video's temporal index.

    Args:
        query_vector: 1D array of shape (D,) representing the embedded query (question).
        index_vectors: 2D array of shape (T, D) representing the temporal video index (T seconds).
        target_indices: Set or list of frame indices corresponding to the ground-truth evidence interval.
        top_k: Cutoff for recall@k (default 5).

    Returns:
        Dictionary containing:
        - 'rank': Rank of the best-scoring target frame (1-indexed, lower is better).
        - 'mrr': Reciprocal rank (1.0 / rank, higher is better).
        - 'recall_at_1': 1.0 if the top-ranked frame in the index is inside target_indices, else 0.0.
        - 'recall_at_k': 1.0 if rank <= top_k, else 0.0.
        - 'best_target_sim': Cosine similarity of the highest-scoring target frame.
        - 'mean_target_sim': Average cosine similarity across all target frames.
        - 'mean_distractor_sim': Average cosine similarity across all non-target frames.
        - 'contrast_margin': Difference between best target similarity and mean distractor similarity.
    """
    query = np.asarray(query_vector, dtype=np.float32)
    index = np.asarray(index_vectors, dtype=np.float32)

    if query.ndim != 1:
        raise ValueError(f"query_vector must be 1D, got shape {query.shape}")
    if index.ndim != 2:
        raise ValueError(f"index_vectors must be 2D, got shape {index.shape}")
    if index.shape[1] != query.shape[0]:
        raise ValueError(
            f"Dimension mismatch: query has dim {query.shape[0]} but index has dim {index.shape[1]}"
        )

    t_total = index.shape[0]
    valid_targets = set(int(i) for i in target_indices if 0 <= int(i) < t_total)
    if not valid_targets:
        logger.warning(f"No target indices fell within index bounds [0, {t_total}).")
        return {
            "rank": t_total,
            "mrr": 1.0 / t_total if t_total > 0 else 0.0,
            "recall_at_1": 0.0,
            f"recall_at_{top_k}": 0.0,
            "best_target_sim": -1.0,
            "mean_target_sim": -1.0,
            "mean_distractor_sim": 0.0,
            "contrast_margin": -1.0,
            "num_targets": 0,
            "total_frames": t_total,
        }

    # Normalize vectors to ensure cosine similarity
    q_norm = np.linalg.norm(query)
    q_unit = query / (q_norm + 1e-12)

    i_norms = np.linalg.norm(index, axis=1, keepdims=True)
    # Handle zero vectors (e.g. masked slots)
    i_norms = np.where(i_norms == 0, 1.0, i_norms)
    i_unit = index / i_norms

    # Cosine similarities for all T frames
    sims = np.dot(i_unit, q_unit)  # Shape: (T,)

    # Extract target and distractor scores
    target_sims = [sims[i] for i in valid_targets]
    best_target_sim = float(np.max(target_sims))
    mean_target_sim = float(np.mean(target_sims))

    distractor_indices = set(range(t_total)) - valid_targets
    if distractor_indices:
        distractor_sims = [sims[i] for i in distractor_indices]
        mean_distractor_sim = float(np.mean(distractor_sims))
        better_distractors = sum(1 for s in distractor_sims if s > best_target_sim + 1e-6)
    else:
        mean_distractor_sim = 0.0
        better_distractors = 0

    rank = better_distractors + 1
    mrr = float(1.0 / rank)
    r1 = 1.0 if rank == 1 else 0.0
    rk = 1.0 if rank <= top_k else 0.0
    contrast = float(best_target_sim - mean_distractor_sim)

    return {
        "rank": rank,
        "mrr": mrr,
        "recall_at_1": r1,
        f"recall_at_{top_k}": rk,
        "best_target_sim": round(best_target_sim, 4),
        "mean_target_sim": round(mean_target_sim, 4),
        "mean_distractor_sim": round(mean_distractor_sim, 4),
        "contrast_margin": round(contrast, 4),
        "num_targets": len(valid_targets),
        "total_frames": t_total,
    }


def build_evidence_retrieval_index(
    original_embeddings: np.ndarray,
    target_indices: set[int] | list[int],
    condition: str,
    reconstructed_embeddings: np.ndarray | None = None
) -> np.ndarray:
    """
    Constructs a video embedding index for a specific experimental condition.

    Args:
        original_embeddings: Full unmasked embeddings of shape (T, D).
        target_indices: Frame indices of the evidence segment.
        condition: One of 'oracle', 'masked', 'baseline_repeat', 'reconstructed'.
        reconstructed_embeddings: Embeddings for the target segment (shape (len(targets), D)).

    Returns:
        Array of shape (T, D) representing the condition's index.
    """
    orig = np.asarray(original_embeddings, dtype=np.float32).copy()
    t_total, dim = orig.shape
    valid_targets = sorted(set(int(i) for i in target_indices if 0 <= int(i) < t_total))

    if not valid_targets or condition == "oracle":
        return orig

    if condition == "masked":
        # Zero out the masked target frames (black hole in semantic index)
        orig[valid_targets] = 0.0
        return orig

    elif condition == "baseline_repeat":
        # Repeat the nearest known boundary frame before or after the gap
        min_idx = valid_targets[0]
        max_idx = valid_targets[-1]
        boundary_idx = None
        if min_idx > 0:
            boundary_idx = min_idx - 1
        elif max_idx + 1 < t_total:
            boundary_idx = max_idx + 1

        if boundary_idx is not None:
            boundary_vec = orig[boundary_idx]
            orig[valid_targets] = boundary_vec
        else:
            orig[valid_targets] = 0.0
        return orig

    elif condition == "baseline_lerp":
        # Linear interpolation weighted by temporal distance from boundary frames
        min_idx = valid_targets[0]
        max_idx = valid_targets[-1]
        has_left = min_idx > 0
        has_right = max_idx + 1 < t_total

        if has_left and has_right:
            v_left = orig[min_idx - 1]
            v_right = orig[max_idx + 1]
            total_span = (max_idx + 1) - (min_idx - 1)
            for t in valid_targets:
                alpha = (t - (min_idx - 1)) / float(total_span)
                v_t = (1.0 - alpha) * v_left + alpha * v_right
                v_norm = np.linalg.norm(v_t)
                if v_norm > 1e-12:
                    v_t = v_t / v_norm
                orig[t] = v_t
        elif has_left:
            orig[valid_targets] = orig[min_idx - 1]
        elif has_right:
            orig[valid_targets] = orig[max_idx + 1]
        else:
            orig[valid_targets] = 0.0
        return orig

    elif condition == "baseline_mean":
        # Flat average of pre-gap and post-gap boundary vectors
        min_idx = valid_targets[0]
        max_idx = valid_targets[-1]
        has_left = min_idx > 0
        has_right = max_idx + 1 < t_total

        if has_left and has_right:
            v_mean = 0.5 * (orig[min_idx - 1] + orig[max_idx + 1])
            v_norm = np.linalg.norm(v_mean)
            if v_norm > 1e-12:
                v_mean = v_mean / v_norm
            orig[valid_targets] = v_mean
        elif has_left:
            orig[valid_targets] = orig[min_idx - 1]
        elif has_right:
            orig[valid_targets] = orig[max_idx + 1]
        else:
            orig[valid_targets] = 0.0
        return orig

    elif condition == "reconstructed":
        if reconstructed_embeddings is None:
            raise ValueError("reconstructed_embeddings must be provided for condition='reconstructed'")
        recon = np.asarray(reconstructed_embeddings, dtype=np.float32)
        if len(recon) != len(valid_targets):
            raise ValueError(
                f"Reconstructed shape {recon.shape} does not match target count {len(valid_targets)}"
            )
        orig[valid_targets] = recon
        return orig

    else:
        raise ValueError(
            f"Unknown condition '{condition}'. Expected 'oracle', 'masked', 'baseline_repeat', 'baseline_lerp', 'baseline_mean', or 'reconstructed'"
        )
