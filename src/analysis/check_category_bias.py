import pandas as pd
import json
from scipy import stats
from pathlib import Path

def check_category_bias():
    llm_path = "results/for_analysis/wild_dev_sim_one_shot_t=1.csv"
    vid_path = "results/for_analysis/wild_dev_sim_vec_vid.csv"
    cats_path = "results/video_categories.json"
    
    # Load Data
    llm_df = pd.read_csv(llm_path)
    vid_df = pd.read_csv(vid_path)
    
    with open(cats_path, 'r') as f:
        cats = json.load(f)
    
    def get_cat(vid_name):
        return cats.get(vid_name, {}).get('category', 'Unknown')
        
    # Prepare combined data
    llm_df['method'] = 'LLM'
    vid_df['method'] = 'Video'
    combined = pd.concat([llm_df, vid_df], ignore_index=True)
    combined['num_masked'] = combined['masked'].apply(lambda x: len(eval(x)))
    
    results_summary = []
    
    # Analyze per masking level
    for mask_level in [6, 9, 12, 15]:
        print(f"\n{'='*40}")
        print(f"Bias Test for Num Masked = {mask_level}")
        print(f"H0: Median/Mean Rank Delta is 0 (No Bias)")
        print(f"{'='*40}")
        
        subset = combined[combined['num_masked'] == mask_level].copy()
        if subset.empty: continue
        
        counts = subset.groupby('video_id')['method'].nunique()
        valid_vids = counts[counts == 2].index
        subset = subset[subset['video_id'].isin(valid_vids)]
        
        agg = subset.groupby(['method', 'video_id'])['cos_sim_mean'].mean().reset_index()
        agg['rank'] = agg.groupby('method')['cos_sim_mean'].rank(method='min', ascending=False)
        
        pivoted = agg.pivot(index='video_id', columns='method', values='rank')
        pivoted['delta'] = pivoted['LLM'] - pivoted['Video'] # Neg = LLM Better
        pivoted['category'] = pivoted.index.map(get_cat)
        
        df_stats = pivoted.dropna()
        
        # Test per category
        for cat in df_stats['category'].unique():
            cat_data = df_stats[df_stats['category'] == cat]['delta']
            n = len(cat_data)
            mean_delta = cat_data.mean()
            median_delta = cat_data.median()
            
            # Wilcoxon Signed-Rank Test (Non-parametric test for mean/median != 0)
            # Only valid if n > ~6 usually, but we'll run it.
            try:
                # 'wilcoxon' tests null hypothesis that distribution is symmetric around zero
                # If all differences are zero, it throws an error or warning, so we handle expected cases.
                res = stats.wilcoxon(cat_data, alternative='two-sided')
                p_val = res.pvalue
            except ValueError:
                # Happens if all diffs are exactly zero or too small sample size issues
                p_val = 1.0
                
            sig_marker = "**" if p_val < 0.05 else ""
            
            bias_dir = "None"
            if mean_delta < 0: bias_dir = "LLM"
            elif mean_delta > 0: bias_dir = "Video"
            
            print(f"{cat:<15} (n={n:<2}) | Mean: {mean_delta:>6.1f} | Med: {median_delta:>4.0f} | p-val: {p_val:>6.4f} {sig_marker} -> Bias: {bias_dir}")

if __name__ == "__main__":
    check_category_bias()
