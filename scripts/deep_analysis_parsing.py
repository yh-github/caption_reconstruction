
import re
import json
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Local path where we downloaded HF results
PHI3_DIR = "results/recon/manual_download/reconstruction/wild_dev_sim_text"
VEC_VID_CSV = "results/recon/wild_dev_sim_vec_vid__09-59_11_01_2026/wild_dev_sim_vec_vid.csv"
CATEGORIES_FILE = "results/video_categories.json"

def parse_phi3_json_dir(root_dir):
    data = []
    # flexible regex matching width and index
    # Folder example: phi-3__t=0.1_rp=1.2__fixed_fill(w=3, i=0)
    folder_re = re.compile(r"w=(\d+), i=(\d+)\)")
    
    for root, dirs, files in os.walk(root_dir):
        # Check if current folder matches config pattern
        match = folder_re.search(root)
        if match:
            width = int(match.group(1))
            index = int(match.group(2))
            
            for file in files:
                if file.endswith(".json"):
                    # video_id from filename (remove .json)
                    video_id = file.rsplit('.', 1)[0]
                    
                    try:
                        with open(os.path.join(root, file), 'r') as f:
                            content = json.load(f)
                            metrics = content.get("metrics", {})
                            
                            row = {
                                "video_id": video_id,
                                "width": width,
                                "index": index,
                                "phi_mean_rank": metrics.get("mean_rank"),
                                "phi_mrr": metrics.get("mrr"),
                                "phi_recall_at_1": metrics.get("recall_at_1"),
                                "phi_recall_at_5": metrics.get("recall_at_5"),
                            }
                            
                            # Compute Median Rank if ranks are available
                            ranks = metrics.get("ranks")
                            if ranks and isinstance(ranks, list) and len(ranks) > 0:
                                row["phi_median_rank"] = float(np.median(ranks))
                            else:
                                # Fallback if ranks missing (shouldn't happen for t=0.1)
                                row["phi_median_rank"] = row["phi_mean_rank"]
                                
                            # Filter empty metrics
                            if row["phi_mean_rank"] is not None:
                                data.append(row)
                    except Exception as e:
                        print(f"Error parsing {file}: {e}")
                        
    return pd.DataFrame(data)

def load_categories():
    with open(CATEGORIES_FILE, 'r') as f:
        return json.load(f)

def main():
    # 1. Parse Phi-3 JSONs
    print(f"Parsing Phi-3 results from {PHI3_DIR}...")
    phi_df = parse_phi3_json_dir(PHI3_DIR)
    print(f"Phi-3 entries found: {len(phi_df)}")
    if len(phi_df) == 0:
        print("No Phi-3 data found. Check path.")
        return
        
    print(phi_df.head())
    
    # 2. Load Vec-Vid
    print("\nLoading Vec-Vid CSV...")
    vec_df = pd.read_csv(VEC_VID_CSV)
    vec_df = vec_df.rename(columns={
        "mean_rank_mean": "vec_mean_rank",
        "mrr_mean": "vec_mrr",
        "recall_at_1_mean": "vec_recall_at_1",
        "recall_at_5_mean": "vec_recall_at_5"
    })
    # Aggregate vec_df (should be 100 rows if one run)
    numeric_cols = ["vec_mean_rank", "vec_mrr", "vec_recall_at_1", "vec_recall_at_5"]
    vec_df = vec_df.groupby("video_id", as_index=False)[numeric_cols].mean()
    print(f"Vec-Vid unique videos: {len(vec_df)}")
    
    # 3. Merge
    print("\nMerging...")
    # Phi-3 has many rows per video (different w, i). Vec-Vid has 1.
    # We want to keep all Phi-3 rows and attach Vec-Vid baseline to each.
    merged = pd.merge(phi_df, vec_df, on="video_id", how="inner")
    print(f"Merged entries: {len(merged)}")
    
    # 4. Add Categories
    cats = load_categories()
    def get_cat(vid):
        val = cats.get(vid)
        if isinstance(val, dict):
            return val.get("category", "Unknown")
        return "Unknown"

    merged['category'] = merged['video_id'].apply(get_cat)
    
    # 5. Analysis
    merged['mrr_delta'] = merged['phi_mrr'] - merged['vec_mrr']
    merged['rank_delta'] = merged['vec_mean_rank'] - merged['phi_mean_rank'] # Pos = Phi better
    
    # --- Deep Analysis Output ---
    
    # A. Performance by Width
    print("\n--- MRR by Width (Phi-3) ---")
    print(merged.groupby('width')['phi_mrr'].mean())
    print("\n--- Mean Rank by Width (Phi-3) ---")
    print(merged.groupby('width')['phi_mean_rank'].mean())
    print("\n--- Median Rank by Width (Phi-3) ---")
    print(merged.groupby('width')['phi_median_rank'].mean()) # Average of medians across videos
    
    # B. Performance by Index
    print("\n--- MRR by Position Index (Phi-3) ---")
    print(merged.groupby('index')['phi_mrr'].mean())
    
    # C. By Category (Aggregated across all w/i)
    # This tells us generally which categories Phi-3 is better at
    print("\n--- Mean Delta by Category (All Configs) ---")
    cat_stats = merged.groupby('category')[['mrr_delta', 'rank_delta', 'phi_mrr', 'vec_mrr']].mean()
    print(cat_stats)

    # D. Head to Head Wins (Best Config per Video vs Baseline?)
    # Or just global wins
    wins = merged[merged['mrr_delta'] > 0]
    print(f"\nTotal instances where Phi-3 > Vec-Vid (MRR): {len(wins)} / {len(merged)}")
    
    # E. Best specific cases (Video + Config)
    print("\n--- Top 5 Phi-3 Wins (MRR Delta) ---")
    print(merged.sort_values('mrr_delta', ascending=False).head(5)[
        ['video_id', 'width', 'index', 'category', 'mrr_delta', 'phi_mrr', 'vec_mrr']
    ])
    
    # F. Analyze Losses
    print("\n--- Top 5 Phi-3 Losses (MRR Delta) ---")
    # Low delta (negative)
    print(merged.sort_values('mrr_delta', ascending=True).head(5)[
         ['video_id', 'width', 'index', 'category', 'mrr_delta', 'phi_mrr', 'vec_mrr']
    ])

    merged.to_csv("results/deep_analysis_final.csv", index=False)

if __name__ == "__main__":
    main()
