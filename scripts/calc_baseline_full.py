
import os
import numpy as np
import pandas as pd
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

EMB_DIR = Path("local/wild_videos_embs/")
OUTPUT_CSV = "results/baseline_full_metrics.csv"
WIDTHS = [3, 6, 9, 12, 15, 18, 21, 24, 27, 30]
INDICES = [0, 29, 59] # Start, Middle, End (approx)

def load_embeddings(path):
    return np.load(path)

def reconstruct_mean_closest(vectors, mask_indices):
    """
    Reconstructs vectors at mask_indices using MeanClosest strategy.
    vectors: Full ID array (ground truth).
    mask_indices: List of indices to mask.
    Returns: Matrix of reconstructed vectors (len = len(mask_indices))
    """
    n = len(vectors)
    reconstructed = []
    
    # Identify known indices (all except mask)
    known_indices = sorted(list(set(range(n)) - set(mask_indices)))
    known_indices = np.array(known_indices)
    
    if len(known_indices) == 0:
        # Edge case: All masked? Should not happen in this setup
        return np.zeros((len(mask_indices), vectors.shape[1]))

    for i in mask_indices:
        # Find closest before
        before = known_indices[known_indices < i]
        closest_before = before[-1] if len(before) > 0 else None
        
        # Find closest after
        after = known_indices[known_indices > i]
        closest_after = after[0] if len(after) > 0 else None
        
        if closest_before is not None and closest_after is not None:
            val = (vectors[closest_before] + vectors[closest_after]) / 2.0
        elif closest_before is not None:
            val = vectors[closest_before] # Repeat last known
        elif closest_after is not None:
            val = vectors[closest_after] # Repeat next known
        else:
            val = np.zeros(vectors.shape[1]) # Should not happen
            
        reconstructed.append(val)
        
    return np.array(reconstructed)

def calculate_metrics(pred_matrix, truth_matrix, mask_indices, full_matrix):
    """
    Compute retrieval metrics for the reconstructed segment.
    pred_matrix: (W, Dim)
    truth_matrix: (W, Dim) - The ground truth for the masked region
    mask_indices: [idx1, idx2, ...] corresponding to rows
    full_matrix: The distractor pool (usually the full video or just the masked region? 
                 In eval_vectors.py it uses "distractor_pool=truth_vecs" which matches "true_vecs" argument.
                 If evaluating a sub-segment, usually we retrieve against the full video? 
                 Wait, eval_vectors.py context: 
                 reconstructed_vectors=pred_vecs (subset), true_vecs (subset), distractor_pool=true_vecs (subset).
                 So it only ranks among the masked candidates? 
                 Step 519 output shows "retrieval_total_queries": 3. 
                 Steps 570 eval_vectors.py: distractor_pool=np.array(true_vecs) passed to calculate_retrieval_metrics.
                 And true_vecs comes from reference clips.
                 So yes, the distractor pool IS ONLY THE MASKED SEGMENT.
    """
    
    # Distractor pool is the truth_matrix (the subset)
    distractor_pool = truth_matrix
    
    # 1. Similarity Matrix
    # Sim(Pred i, Truth j)
    # Norms
    norm_pred = np.linalg.norm(pred_matrix, axis=1, keepdims=True)
    norm_truth = np.linalg.norm(distractor_pool, axis=1, keepdims=True)
    
    sim_matrix = np.dot(pred_matrix, distractor_pool.T) / (norm_pred * norm_truth.T + 1e-9)
    # Shape (W, W)
    
    n = len(mask_indices)
    ranks = []
    
    # Standard Metrics
    for i in range(n):
        true_score = sim_matrix[i, i]
        # Count how many are better
        better = np.sum(sim_matrix[i, :] > true_score)
        rank = better + 1
        ranks.append(rank)
        
    ranks = np.array(ranks)
    mean_rank = np.mean(ranks)
    median_rank = np.median(ranks)
    mrr = np.mean(1.0 / ranks)
    r1 = np.mean(ranks == 1)
    
    # Temporal Metrics
    # Keys for temporal are simply the mask_indices relative to the window?
    # No, mask_indices are global. But the matrix columns correspond to mask_indices[0], mask_indices[1]...
    # So column j corresponds to mask_indices[j].
    # Pred i corresponds to mask_indices[i].
    # Distance = |mask_indices[i] - mask_indices[p_idx]|
    
    preds = np.argmax(sim_matrix, axis=1) # Indices into the subset (0..W-1)
    
    success_w1 = 0
    success_w2 = 0
    ndcg_sum = 0
    
    for i in range(n):
        # R@1 Windowed
        p_idx_subset = preds[i]
        true_idx_global = mask_indices[i]
        pred_idx_global = mask_indices[p_idx_subset]
        
        dist = abs(true_idx_global - pred_idx_global)
        if dist <= 1: success_w1 += 1
        if dist <= 2: success_w2 += 1
        
        # NDCG
        # Relevance of column j for query i: 1 / (1 + |Global_i - Global_j|)
        sims = sim_matrix[i]
        ranked_subset_indices = np.argsort(sims)[::-1]
        
        dcg = 0.0
        for rank, sub_idx in enumerate(ranked_subset_indices):
            g_idx = mask_indices[sub_idx]
            d = abs(true_idx_global - g_idx)
            # Stricter: Exp decay
            rel = np.exp(-d / 2.0)
            dcg += rel / np.log2(rank + 2)
            
        # IDCG
        # Ideal distances: |Global_i - Global_k| for all k in subset
        dists_ideal = [abs(true_idx_global - mask_indices[k]) for k in range(n)]
        rels_ideal = sorted([np.exp(-d / 2.0) for d in dists_ideal], reverse=True)
        
        idcg = 0.0
        for rank, rel in enumerate(rels_ideal):
            idcg += rel / np.log2(rank + 2)
            
        if idcg > 0:
            ndcg_sum += dcg / idcg
            
    return {
        "mean_rank": mean_rank,
        "median_rank": median_rank,
        "mrr": mrr,
        "recall_at_1": r1,
        "temporal_recall_at_1_w1": success_w1 / n if n > 0 else 0,
        "temporal_recall_at_1_w2": success_w2 / n if n > 0 else 0,
        "temporal_ndcg": ndcg_sum / n if n > 0 else 0
    }

def process_video(npy_path):
    try:
        vid_id = npy_path.stem
        vectors = load_embeddings(npy_path)
        
        results = []
        
        for w in WIDTHS:
            for start_idx in INDICES:
                # Define mask
                end_idx = start_idx + w
                if end_idx > len(vectors):
                    continue # Skip if out of bounds
                    
                mask_indices = list(range(start_idx, end_idx))
                
                # Reconstruct
                reconstructed_matrix = reconstruct_mean_closest(vectors, mask_indices)
                truth_matrix = vectors[mask_indices]
                
                # Check NaNs in truth (some videos might have nans? unlikely for embeddings)
                if np.isnan(truth_matrix).any():
                    continue
                    
                metrics = calculate_metrics(reconstructed_matrix, truth_matrix, mask_indices, vectors)
                
                metrics.update({
                    "video_id": vid_id,
                    "width": w,
                    "index": start_idx
                })
                results.append(metrics)
                
        return results
    except Exception as e:
        # print(f"Error {npy_path}: {e}")
        return []

def main():
    print("Listing embedding files...")
    files = list(EMB_DIR.glob("*.npy"))
    print(f"Found {len(files)} files.")
    
    all_results = []
    
    with ProcessPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(process_video, f) for f in files]
        for f in as_completed(futures):
            res = f.result()
            all_results.extend(res)
            
    df = pd.DataFrame(all_results)
    print(f"computed {len(df)} records.")
    
    # Add Categories
    try:
        import json
        with open("results/video_categories.json", 'r') as f:
            cats = json.load(f)
        df['category'] = df['video_id'].apply(lambda x: cats.get(x, {}).get("category", "Unknown") if isinstance(cats.get(x), dict) else "Unknown")
    except:
        print("Could not load categories")
        
    df.to_csv(OUTPUT_CSV, index=False)
    print(df.groupby('width')[['mrr', 'temporal_ndcg']].mean())

if __name__ == "__main__":
    main()
