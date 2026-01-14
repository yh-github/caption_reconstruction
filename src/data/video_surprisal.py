
import numpy as np
import logging
from dataclasses import dataclass

@dataclass
class VideoSurprisalResult:
    avg_cosine_distance: float
    max_cosine_distance: float
    variance_cosine_distance: float
    # We could add more complex stats if needed

class VideoSurprisalScorer:
    """
    Calculates surprisal/complexity of a video based on its embeddings.
    Surprisal here is proxied by the magnitude of change (1 - cosine_similarity) 
    between consecutive frame embeddings.
    High change = High Surprisal/Dynamic Video.
    Low change = Low Surprisal/Static Video.
    """
    def __init__(self):
        pass

    def calculate_surprisal(self, embeddings: np.ndarray) -> VideoSurprisalResult:
        """
        embeddings: (T, D) numpy array
        """
        if embeddings.shape[0] < 2:
            return VideoSurprisalResult(0.0, 0.0, 0.0)
            
        # 1. Normalize (just in case)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / (norms + 1e-9)
        
        # 2. Compute Cosine Sim between t and t+1
        # Dot product of row i and row i+1
        # shape (T-1, D) * (T-1, D) -> (T-1,)
        sims = np.sum(embeddings[:-1] * embeddings[1:], axis=1)
        
        # 3. Convert to Distance (1 - Sim)
        # Range: 0 (identical) to 2 (opposite)
        dists = 1.0 - sims
        
        avg_dist = float(np.mean(dists))
        max_dist = float(np.max(dists))
        var_dist = float(np.var(dists))
        
        return VideoSurprisalResult(
            avg_cosine_distance=avg_dist,
            max_cosine_distance=max_dist,
            variance_cosine_distance=var_dist
        )
