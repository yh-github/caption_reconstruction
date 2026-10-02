import pandas as pd
import numpy as np
import os
from scipy.stats import pearsonr, spearmanr

def main():
    apriori_df = pd.read_csv("results/apriori_dynamism_scores.csv")
    eval_df = pd.read_csv("results/phi_vs_video_integration_summary.csv")
    
    # Merge datasets
    df = pd.merge(apriori_df, eval_df, on="video_id", how="inner")
    
    apriori_cols = [
        "average_dynamism", "peak_dynamism", "combined_dynamism", "APCS_V",
        "caption_surprisal_nll", "caption_perplexity"
    ]
    
    # Existing text vs video metrics
    base_metrics = ["mrr", "temporal_ndcg", "cos_sim_mean", "cos_sim_min", "cos_sim_residual_mean", "cos_sim_residual_min"]
    
    # Compute rankings and differences
    posteriori_cols = []
    
    for metric in base_metrics:
        t_col = f"text_{metric}"
        v_col = f"video_{metric}"
        if t_col in df.columns and v_col in df.columns:
            # Rank across all videos (lower rank number = better score, so ascending=False)
            df[f"{t_col}_rank"] = df[t_col].rank(ascending=False)
            df[f"{v_col}_rank"] = df[v_col].rank(ascending=False)
            
            # Rank difference (Text rank - Video rank). Negative means Text was better (lower rank).
            df[f"{metric}_rank_diff"] = df[f"{t_col}_rank"] - df[f"{v_col}_rank"]
            
            # Value difference
            df[f"{metric}_diff"] = df[t_col] - df[v_col]
            
            posteriori_cols.extend([t_col, v_col, f"{metric}_diff", f"{metric}_rank_diff"])
    
    # Optional: Rank per movie_id
    for metric in base_metrics:
        t_col = f"text_{metric}"
        v_col = f"video_{metric}"
        if t_col in df.columns and v_col in df.columns:
            df[f"{t_col}_movie_rank"] = df.groupby("movie_id")[t_col].rank(ascending=False)
            df[f"{v_col}_movie_rank"] = df.groupby("movie_id")[v_col].rank(ascending=False)
            df[f"{metric}_movie_rank_diff"] = df[f"{t_col}_movie_rank"] - df[f"{v_col}_movie_rank"]
            posteriori_cols.extend([f"{metric}_movie_rank_diff"])

    # Output full merged dataset
    os.makedirs("results/analysis", exist_ok=True)
    df.to_csv("results/analysis/full_merged_evaluations_and_ranks.csv", index=False)
    
    # Correlation analysis
    # We want to see how Apriori scores correlate with Posteriori diffs / rank diffs
    correlations = []
    
    # Clean apriori cols (drop missing)
    valid_apriori = [c for c in apriori_cols if c in df.columns]
    
    for ap_col in valid_apriori:
        for p_col in posteriori_cols:
            # Drop NaNs
            sub_df = df[[ap_col, p_col]].dropna()
            if len(sub_df) > 2:
                pear_r, pear_p = pearsonr(sub_df[ap_col], sub_df[p_col])
                spear_r, spear_p = spearmanr(sub_df[ap_col], sub_df[p_col])
                
                correlations.append({
                    "apriori_metric": ap_col,
                    "evaluation_metric": p_col,
                    "pearson_r": pear_r,
                    "pearson_p": pear_p,
                    "spearman_r": spear_r,
                    "spearman_p": spear_p
                })
                
    corr_df = pd.DataFrame(correlations)
    corr_df = corr_df.sort_values(by="spearman_r", key=abs, ascending=False)
    corr_df.to_csv("results/analysis/apriori_posteriori_correlations.csv", index=False)
    
    print("Top 15 Spearman correlations (absolute value):")
    print(corr_df.head(15).to_string(index=False))
    print(f"\nMerged data saved to results/analysis/full_merged_evaluations_and_ranks.csv")
    print(f"Correlations saved to results/analysis/apriori_posteriori_correlations.csv")

if __name__ == "__main__":
    main()
