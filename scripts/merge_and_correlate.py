
import pandas as pd
import json
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Paths
TEMPORAL_METRICS_PATH = "results/temporal_metrics_final.csv"
EUCLIDEAN_METRICS_PATH = "results/euclidean_metrics.csv"
VIDEO_SURPRISAL_PATH = "results/video_surprisal_scores.csv"
PRIOR_SCORES_PATH = "results/scores/prior_scoring_7109bc27281d.json"
FULL_BASELINE_PATH = "results/baseline_full_metrics.csv"
OUTPUT_DIR = "results/plots/correlations"
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

def load_prior_scores():
    if not Path(PRIOR_SCORES_PATH).exists():
        print(f"Warning: {PRIOR_SCORES_PATH} not found.")
        return pd.DataFrame()
        
    with open(PRIOR_SCORES_PATH, 'r') as f:
        data = json.load(f)
        
    scores = data.get('scores', {})
    rows = []
    for vid_id, res in scores.items():
        if 'whole_video_surprisal' in res and res['whole_video_surprisal']:
            ws = res['whole_video_surprisal']
            rows.append({
                "video_id": vid_id,
                "text_surprisal_nll": ws.get('avg_surprisal_nll'),
                "text_perplexity": ws.get('avg_perplexity')
            })
    return pd.DataFrame(rows)

def main():
    print("Loading datasets...")
    
    # 1. Main Experiment Data (Phi-3)
    df_phi = pd.read_csv(TEMPORAL_METRICS_PATH)
    # We need to construct a unique key for merging if granular, or just aggregate by video_id?
    # Correlations are best done at the VIDEO level (aggregating over all masks for that video)
    # OR at the instance level if we can map it.
    # Let's aggregate by video first to see general trends.
    
    # Aggregating Phi Metrics by Video
    phi_vid_agg = df_phi.groupby('video_id')[['phi_mrr', 'temporal_ndcg']].mean().reset_index()
    
    # 2. Baseline Data (Vec)
    # We want the delta: Phi - Vec
    if Path(FULL_BASELINE_PATH).exists():
        df_base = pd.read_csv(FULL_BASELINE_PATH)
        # Baseline also has many points per video (width/index).
        # Aggregating by video to match Phi aggregation logic (avg performance on this video)
        base_vid_agg = df_base.groupby('video_id')[['mrr', 'temporal_ndcg']].mean().reset_index()
        base_vid_agg = base_vid_agg.rename(columns={'mrr': 'base_mrr', 'temporal_ndcg': 'base_temporal_ndcg'})
        
        # Merge Baseline
        merged = pd.merge(phi_vid_agg, base_vid_agg, on='video_id', how='left')
        merged['mrr_delta'] = merged['phi_mrr'] - merged['base_mrr']
        merged['t_ndcg_delta'] = merged['temporal_ndcg'] - merged['base_temporal_ndcg']
    else:
        merged = phi_vid_agg
        print("Warning: Full baseline not found, skipping Delta calculation.")
        
    # 3. Euclidean Metrics
    if Path(EUCLIDEAN_METRICS_PATH).exists():
        df_euc = pd.read_csv(EUCLIDEAN_METRICS_PATH)
        euc_agg = df_euc.groupby('video_id')[['euclidean_dist', 'cosine_dist_check']].mean().reset_index()
        merged = pd.merge(merged, euc_agg, on='video_id', how='left')
        
    # 4. Video Surprisal (Variance)
    if Path(VIDEO_SURPRISAL_PATH).exists():
        df_vid_surp = pd.read_csv(VIDEO_SURPRISAL_PATH)
        merged = pd.merge(merged, df_vid_surp, on='video_id', how='left')
        
    # 5. Text Surprisal (Prior)
    df_text_surp = load_prior_scores()
    if not df_text_surp.empty:
        merged = pd.merge(merged, df_text_surp, on='video_id', how='left')
        
    print(f"Merged Data Shape: {merged.shape}")
    merged.to_csv("results/final_correlations_master.csv", index=False)
    
    # --- PLOTTING CORRELATIONS ---
    
    # A. Text Surprisal vs MRR Delta
    # Hypothesis: Higher Surprisal -> Higher Phi Advantage (Positive Delta)
    if 'text_surprisal_nll' in merged.columns and 'mrr_delta' in merged.columns:
        plt.figure(figsize=(8, 6))
        sns.regplot(data=merged, x='text_surprisal_nll', y='mrr_delta')
        plt.title("Correlation: Text Surprisal (Difficulty) vs Phi-3 Advantage")
        plt.xlabel("Text Surprisal (NLL)")
        plt.ylabel("MRR Delta (Phi - Vec)")
        plt.savefig(f"{OUTPUT_DIR}/corr_text_surp_vs_delta.png")
        plt.close()
        
    # B. Video Variance vs MRR Delta
    # Hypothesis: Higher Variance (Dynamic Video) -> Higher Phi Advantage?
    if 'video_var_dist' in merged.columns and 'mrr_delta' in merged.columns:
        plt.figure(figsize=(8, 6))
        sns.regplot(data=merged, x='video_var_dist', y='mrr_delta')
        plt.title("Correlation: Video Dynamism (Variance) vs Phi-3 Advantage")
        plt.xlabel("Video Embedding Cosine Variance")
        plt.ylabel("MRR Delta (Phi - Vec)")
        plt.savefig(f"{OUTPUT_DIR}/corr_vid_var_vs_delta.png")
        plt.close()
        
    # C. Euclidean Distance vs MRR
    # Hypothesis: Does "Euclidean Alignment" correlate with MRR?
    # Note: Using unaligned spaces (Text vs DINO), but checking anyway.
    if 'euclidean_dist' in merged.columns:
        plt.figure(figsize=(8, 6))
        sns.regplot(data=merged, x='euclidean_dist', y='phi_mrr')
        plt.title("Correlation: Text-Video Euclidean Dist vs MRR")
        plt.xlabel("Euclidean Distance (Text vs DINO)")
        plt.ylabel("Phi-3 MRR")
        plt.savefig(f"{OUTPUT_DIR}/corr_euclidean_vs_mrr.png")
        plt.close()

    # D. Correlation Matrix Heatmap
    # Filter for numeric columns of interest
    cols = ['phi_mrr', 'temporal_ndcg', 'mrr_delta', 't_ndcg_delta', 
            'euclidean_dist', 'video_var_dist', 'text_surprisal_nll', 'video_length']
    cols = [c for c in cols if c in merged.columns]
    
    corr = merged[cols].corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title("Correlation Matrix of All Metrics")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/correlation_matrix.png")
    plt.close()
    
    print(f"Correlation plots saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
