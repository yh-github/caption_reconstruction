import numpy as np

def get_video_complexity(gt_vectors: np.ndarray) -> float:
    """
    Calculates how dynamic a video's narrative is.
    Returns 0.0 (Static/Repetitive) to 1.0 (Chaotic/Unique).
    """
    if len(gt_vectors) < 2:
        return 0.0
        
    # 1. Normalize vectors just in case
    norms = np.linalg.norm(gt_vectors, axis=1, keepdims=True)
    gt_norm = gt_vectors / (norms + 1e-8)
    
    # 2. Compute Cosine Sim between t and t+1
    # Shape: (T-1,)
    adjacent_sims = np.sum(gt_norm[:-1] * gt_norm[1:], axis=1)
    
    # 3. Complexity is the inverse of similarity
    # If frames are identical (sim=1.0), complexity is 0.
    mean_similarity = np.mean(adjacent_sims)
    
    return 1.0 - mean_similarity

def get_extended_video_complexity(gt_vectors: np.ndarray) -> dict:
    from data.video_surprisal import VideoSurprisalScorer
    scorer = VideoSurprisalScorer()
    res = scorer.calculate_surprisal(gt_vectors)
    return {
        "mean_surprisal": res.avg_cosine_distance,
        "max_surprisal": res.max_cosine_distance,
        "variance_surprisal": res.variance_cosine_distance,
        "p95_surprisal": res.p95_cosine_distance,
        "effective_rank": res.effective_rank,
        "tortuosity": res.tortuosity
    }
def get_narrative_complexity(captions: list[str]) -> float:
    """
    Calculates Type-Token Ratio (TTR) as a proxy for 
    Semantic Volatility, independent of embedding models.
    """
    all_words = " ".join(captions).lower().split()
    if not all_words:
        return 0.0
    
    # Structural Volatility: How unique is the vocabulary?
    unique_words = set(all_words)
    ttr = len(unique_words) / len(all_words)
    
    return ttr

def get_temporal_density(captions: list[str], duration_seconds: float) -> float:
    """
    Calculates events per second.
    """
    if duration_seconds <= 0: return 0.0
    return len(set(captions)) / duration_seconds