import pandas as pd
from scipy.stats import pearsonr, spearmanr
from analysis.llm_based import load_dfs

def check_correlation_trend():
    llm_path = "results/for_analysis/wild_dev_sim_one_shot_t=1.csv"
    vid_path = "results/for_analysis/wild_dev_sim_vec_vid.csv"
    
    print("Loading data...")
    llm_df = pd.read_csv(llm_path)
    vid_df = pd.read_csv(vid_path)
    
    llm_df['method'] = 'LLM'
    vid_df['method'] = 'Video'
    combined = pd.concat([llm_df, vid_df], ignore_index=True)
    combined['num_masked'] = combined['masked'].apply(lambda x: len(eval(x)))
    
    mask_levels = [6, 9, 12, 15]
    
    print(f"{'Mask':<6} {'N':<5} {'Pearson (Ranks)':<15} {'Spearman':<15}")
    print("-" * 45)
    
    for m in mask_levels:
        subset = combined[combined['num_masked'] == m].copy()
        
        # Intersection
        counts = subset.groupby('video_id')['method'].nunique()
        valid_vids = counts[counts == 2].index
        subset = subset[subset['video_id'].isin(valid_vids)]
        
        if subset.empty: continue
        
        agg = subset.groupby(['method', 'video_id'])['cos_sim_mean'].mean().reset_index()
        agg['rank'] = agg.groupby('method')['cos_sim_mean'].rank(method='first', ascending=False)
        pivoted = agg.pivot(index='video_id', columns='method', values='rank')
        
        # Correlation between LLM Rank and Video Rank
        # High Correlation = They agree on what is easy/hard
        # Low Correlation = They disagree (Spectrum) OR distinct failure modes
        
        p_r, _ = pearsonr(pivoted['LLM'], pivoted['Video'])
        s_r, _ = spearmanr(pivoted['LLM'], pivoted['Video'])
        
        print(f"{m:<6} {len(pivoted):<5} {p_r:.4f}           {s_r:.4f}")

if __name__ == "__main__":
    check_correlation_trend()
