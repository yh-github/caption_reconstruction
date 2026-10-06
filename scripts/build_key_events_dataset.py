"""Find each wild4 video's "climax" second (peak SigLIP-text distance from the mean caption) and write results/wild_key_events.csv."""
import json
import glob
import pandas as pd
import numpy as np
from scipy.ndimage import gaussian_filter1d
import os
import sys

# Ensure src is in path
sys.path.append('src')
from eval_shared_target import get_benchmark_meta
from llm.local_embedder import SiglipTextEmbedder

def main():
    print("1. Loading benchmark metadata...")
    meta_dict_all, _ = get_benchmark_meta("results/unified_benchmark_master.csv")
    w4_meta = {k: v for k, v in meta_dict_all.items() if v.dataset == "Wild4"}
    
    embedder = SiglipTextEmbedder()
    
    records = []
    
    print(f"2. Processing {len(w4_meta)} videos for Climax Detection...")
    for vid in w4_meta.keys():
        caps = [] # IMPORTANT FIX: Reset caps
        gt_file = [f"datasets/wildQA/captions__wild4/{vid}.json"]
        if not gt_file:
            continue
            
        try:
            with open(gt_file[0]) as f:
                data = json.load(f)
                if isinstance(data, list):
                    caps = data
                elif isinstance(data, dict):
                    caps = data.get('data', data.get('captions', data.get('clips', [])))
        except Exception:
            continue
                
        if not caps or 'caption' not in caps[0]:
            continue
            
        captions = [x['caption'] for x in caps]
        if len(captions) < 40:
            continue
            
        embs = embedder.get_embeddings(f"siglip_climax_{vid}", captions)
        embs = np.array(embs)
        
        norms = np.linalg.norm(embs, axis=-1, keepdims=True)
        norms[norms == 0] = 1.0
        embs = embs / norms
        
        mu = np.mean(embs, axis=0)
        mu = mu / np.linalg.norm(mu)
        
        distances = 1 - np.dot(embs, mu)
        smoothed = gaussian_filter1d(distances, sigma=1.0)
        
        start_idx = 15
        end_idx = len(captions) - 15
        if end_idx <= start_idx:
            continue
            
        valid_smoothed = smoothed[start_idx:end_idx]
        local_climax = np.argmax(valid_smoothed)
        climax_t = start_idx + local_climax
        
        records.append({
            "video_id": vid,
            "climax_start": climax_t,
            "climax_distance": distances[climax_t],
            "smoothed_distance": smoothed[climax_t],
            "caption": captions[climax_t]
        })
        
    df = pd.DataFrame(records)
    os.makedirs("results", exist_ok=True)
    df.to_csv("results/wild_key_events.csv", index=False)
    
    print(f"\nSuccessfully curated {len(df)} Key Events.")
    print("Sample of highest anomaly events:")
    print(df.sort_values("smoothed_distance", ascending=False).head(5)[['video_id', 'climax_start', 'caption']])

if __name__ == "__main__":
    main()
