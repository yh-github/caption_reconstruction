from __future__ import annotations

import numpy as np
import logging
from dataclasses import dataclass

@dataclass
class VideoSurprisalResult:
    avg_cosine_distance: float
    max_cosine_distance: float
    variance_cosine_distance: float
    p95_cosine_distance: float
    effective_rank: float
    tortuosity: float
    apcs_v: float
    combined_dynamism: float

class VideoSurprisalScorer:
    """
    Calculates surprisal/complexity of a video based on its embeddings.
    """
    def __init__(self):
        pass

    def calculate_surprisal(self, embeddings: np.ndarray) -> VideoSurprisalResult:
        """
        embeddings: (T, D) numpy array
        """
        if embeddings.shape[0] < 2:
            return VideoSurprisalResult(0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0)
            
        # 1. Normalize (just in case)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / (norms + 1e-9)
        
        # 2. Compute Cosine Sim between t and t+1
        sims = np.sum(embeddings[:-1] * embeddings[1:], axis=1)
        
        # 3. Convert to Distance (1 - Sim)
        dists = 1.0 - sims
        
        avg_dist = float(np.mean(dists))
        max_dist = float(np.max(dists))
        var_dist = float(np.var(dists))
        p95_dist = float(np.percentile(dists, 95))
        
        # 4. Effective Rank
        try:
            _, S, _ = np.linalg.svd(embeddings, full_matrices=False)
            p = S / np.sum(S)
            H = -np.sum(p * np.log(p + 1e-9))
            eff_rank = float(np.exp(H))
        except np.linalg.LinAlgError:
            eff_rank = 1.0

        # 5. Tortuosity (path length / displacement)
        diffs = embeddings[1:] - embeddings[:-1]
        path_length = np.sum(np.linalg.norm(diffs, axis=1))
        displacement = np.linalg.norm(embeddings[-1] - embeddings[0])
        tortuosity = float(path_length / (displacement + 1e-9))
        
        # 6. APCS_V (Average Pairwise Cosine Similarity)
        m = embeddings.shape[0]
        if m > 1:
            all_sims = np.dot(embeddings, embeddings.T)
            j, k = np.triu_indices(m, k=1)
            pairwise_sims = all_sims[j, k]
            apcs_v = float(np.mean(pairwise_sims))
        else:
            apcs_v = 0.0

        # 7. Combined Dynamism
        combined_dynamism = 100.0 * (0.5 * avg_dist + 0.5 * p95_dist)

        return VideoSurprisalResult(
            avg_cosine_distance=avg_dist,
            max_cosine_distance=max_dist,
            variance_cosine_distance=var_dist,
            p95_cosine_distance=p95_dist,
            effective_rank=eff_rank,
            tortuosity=tortuosity,
            apcs_v=apcs_v,
            combined_dynamism=combined_dynamism
        )

