
import os
import re
import json
import base64
import io
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed

import argparse

DEFAULT_PHI3_DIR = "results/recon/manual_download/reconstruction/wild_dev_sim_text"
OUTPUT_CSV = "results/temporal_metrics_final.csv"

def matrix_from_b64(b64_str: str):
    if not b64_str: return None
    try:
        bytes_data = base64.b64decode(b64_str)
        with io.BytesIO(bytes_data) as f:
            return np.load(f, allow_pickle=True)
    except:
        return None

def calculate_temporal_ndcg(sim_matrix, keys):
    """
    NDCG where relevance is based on temporal proximity.
    Matrix: Rows=Preds, Cols=Truth Candidates.
    Keys: Indices corresponding to rows/cols.
    Relevance = 1 / (1 + |TrueIndex - CandidateIndex|)
    """
    ndcg_sum = 0
    n = len(keys)
    if n == 0: return 0.0
    
    for i in range(n):
        true_index = keys[i]
        
        # Predicted similarities for this query
        sims = sim_matrix[i]
        
        # Sort candidates by similarity (descending)
        ranked_indices = np.argsort(sims)[::-1] # indices into 'keys'
        
        # Calculate DCG
        dcg = 0.0
        for rank, cand_idx_ptr in enumerate(ranked_indices):
            cand_idx = keys[cand_idx_ptr]
            dist = abs(true_index - cand_idx)
            # Stricter Decay: Exponential
            # sigma = 2.0. dist=0->1.0, dist=2->0.36, dist=5->0.08
            rel = np.exp(-dist / 2.0)
            dcg += rel / np.log2(rank + 2) # rank is 0-based
            
        # Calculate IDCG (Ideal ordering by distance)
        # Ideal order is sorted by distance (0, 1, 1, 2, 2...)
        # Distances: |true - keys[j]| for all j
        distances = [abs(true_index - k) for k in keys]
        ideal_rels = sorted([np.exp(-d / 2.0) for d in distances], reverse=True)
        
        idcg = 0.0
        for rank, rel in enumerate(ideal_rels):
            idcg += rel / np.log2(rank + 2)
            
        if idcg > 0:
            ndcg_sum += dcg / idcg
            
    return ndcg_sum / n

def process_file(filepath, width, index, temp, rp, video_id):
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
            
        metrics = data.get("metrics", {})
        b64 = metrics.get("similarity_matrix_b64")
        
        if not b64:
            return None
            
        sim_matrix = matrix_from_b64(b64)
        if sim_matrix is None:
            return None
            
        # Get indices
        keys = sorted([int(k) for k in data.get('reconstructed_captions', {}).keys() if k.isdigit()])
        
        if len(keys) != sim_matrix.shape[0]:
            # Mismatch, skip
            return None
            
        # 1. Temporal R@1 (Window 1 & 2)
        preds = np.argmax(sim_matrix, axis=1) # indices into 'keys'
        
        success_w1 = 0
        success_w2 = 0
        n = len(keys)
        
        for i, p_ptr in enumerate(preds):
            true_idx = keys[i]
            pred_idx = keys[p_ptr]
            dist = abs(true_idx - pred_idx)
            
            if dist <= 1: success_w1 += 1
            if dist <= 2: success_w2 += 1
            
        r1_w1 = success_w1 / n if n > 0 else 0
        r1_w2 = success_w2 / n if n > 0 else 0
        
        # 2. Temporal NDCG
        ndcg_temp = calculate_temporal_ndcg(sim_matrix, keys)
        
        # Return row for dataframe
        # Include original metrics too for context
        return {
            "video_id": video_id,
            "width": width,
            "index": index,
            "phi_mrr": metrics.get("mrr"),
            "phi_recall_at_1": metrics.get("recall_at_1"),
            "temporal_recall_at_1_w1": r1_w1,
            "temporal_recall_at_1_w2": r1_w2,
            "temporal_ndcg": ndcg_temp,
            "temperature": temp,
            "repetition_penalty": rp
        }
        
    except Exception as e:
        # print(f"Error {filepath}: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description="Recalculate temporal metrics from experiment results.")
    parser.add_argument("--dir", type=str, default=DEFAULT_PHI3_DIR, help="Directory containing experiment results (JSON files).")
    args = parser.parse_args()
    phi3_dir = args.dir

    tasks = []
    # Folder pattern: phi-3__t=0.1_rp=1.2__fixed_fill(w=12, i=0)
    folder_re = re.compile(r"t=(?P<temp>[\d\.]+)_rp=(?P<rp>[\d\.]+).*w=(?P<width>\d+), i=(?P<index>\d+)\)")
    
    print(f"Collecting files from {phi3_dir}...")
    for root, dirs, files in os.walk(phi3_dir):
        match = folder_re.search(root)
        if match:
            width = int(match.group('width'))
            index = int(match.group('index'))
            temp = float(match.group('temp'))
            rp = float(match.group('rp'))
            
            for file in files:
                if file.endswith(".json"):
                    vid = file.rsplit('.', 1)[0]
                    path = os.path.join(root, file)
                    tasks.append((path, width, index, temp, rp, vid))
                    
    print(f"Processing {len(tasks)} files...")
    
    results = []
    with ProcessPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(process_file, *t) for t in tasks]
        for f in as_completed(futures):
            res = f.result()
            if res:
                results.append(res)
                
    df = pd.DataFrame(results)
    
    # Add Categories
    with open("results/video_categories.json", 'r') as f:
        cats = json.load(f)
        
    def get_cat(vid):
        val = cats.get(vid)
        if isinstance(val, dict):
            return val.get("category", "Unknown")
        return "Unknown"
        
    df['category'] = df['video_id'].apply(get_cat)
    
    print(f"Saving {len(df)} records to {OUTPUT_CSV}")
    df.to_csv(OUTPUT_CSV, index=False)
    print(df.describe())

if __name__ == "__main__":
    main()
