import pandas as pd
import json
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd

def check_significance():
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
    
    results = []
    
    # Analyze per masking level
    for mask_level in [6, 9, 12, 15]:
        print(f"\n{'='*40}")
        print(f"Significance Test for Num Masked = {mask_level}")
        print(f"{'='*40}")
        
        subset = combined[combined['num_masked'] == mask_level].copy()
        if subset.empty: continue
        
        # Intersection only
        counts = subset.groupby('video_id')['method'].nunique()
        valid_vids = counts[counts == 2].index
        subset = subset[subset['video_id'].isin(valid_vids)]
        
        # Rank Deltas
        agg = subset.groupby(['method', 'video_id'])['cos_sim_mean'].mean().reset_index()
        agg['rank'] = agg.groupby('method')['cos_sim_mean'].rank(method='min', ascending=False)
        
        pivoted = agg.pivot(index='video_id', columns='method', values='rank')
        pivoted['delta'] = pivoted['LLM'] - pivoted['Video'] # Neg = LLM Better
        pivoted['category'] = pivoted.index.map(get_cat)
        
        df_stats = pivoted.dropna()
        
        # 1. Kruskal-Wallis Test (Non-parametric ANOVA)
        # H0: All categories have the same median rank delta
        groups = [group['delta'].values for name, group in df_stats.groupby('category')]
        h_stat, p_val = stats.kruskal(*groups)
        
        print(f"Kruskal-Wallis H-test: H={h_stat:.2f}, p-value={p_val:.4g}")
        
        if p_val < 0.05:
            print("-> SIGNIFICANT difference between categories.")
            
            # 2. Pairwise Tukey HSD (Parametric, but robust enough for ranking roughly)
            print("\nPairwise Tukey HSD (Mean Diff):")
            tukey = pairwise_tukeyhsd(endog=df_stats['delta'], groups=df_stats['category'], alpha=0.05)
            print(tukey.summary())
        else:
            print("-> NO significant difference detected between categories.")

if __name__ == "__main__":
    check_significance()
