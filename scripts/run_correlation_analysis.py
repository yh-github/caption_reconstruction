import pandas as pd
import scipy.stats

def compute_corrs(df, x_cols, y_cols):
    res = []
    for y in y_cols:
        for x in x_cols:
            df_valid = df.dropna(subset=[x, y])
            if len(df_valid) < 10:
                continue
            pearson, p_val = scipy.stats.pearsonr(df_valid[x], df_valid[y])
            spearman, s_pval = scipy.stats.spearmanr(df_valid[x], df_valid[y])
            res.append({
                'metric_y': y,
                'apriori_x': x,
                'pearson': pearson,
                'pearson_p': p_val,
                'spearman': spearman,
                'spearman_p': s_pval,
                'n': len(df_valid)
            })
    return pd.DataFrame(res)

def main():
    apriori = pd.read_csv('results/apriori_full_scores.csv')
    llama = pd.read_csv('results/llama_all_experiments_aggregated.csv')
    
    # Let's filter llama to the most standard config: w=6, i=29 (and w=3, i=29)
    # Actually let's just use w=6, i=29 for now as the main evaluation target
    df_eval = llama[(llama['w'] == 6) & (llama['i'] == 29)]
    
    merged = pd.merge(apriori, df_eval, on='video_id', how='inner')
    print(f"Merged {len(merged)} videos for correlation (w=6, i=29).")
    
    x_cols = ['combined_dynamism', 'APCS_V', 'text_combined_dynamism', 'APCS_T', 'apcs_nll', 'caption_perplexity']
    y_cols = ['mrr', 'cos_sim_mean']
    
    corrs = compute_corrs(merged, x_cols, y_cols)
    print("\n--- Correlations for w=6, i=29 ---")
    print(corrs.to_string(index=False))
    
    corrs.to_csv("results/analysis_correlations_w6_i29.csv", index=False)
    
if __name__ == "__main__":
    main()
