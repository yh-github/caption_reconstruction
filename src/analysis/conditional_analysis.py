import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def run_conditional_analysis(
    llm_results_path,
    video_emb_results_path,
    categories_path,
    output_dir,
    metric='sim_mean'
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load Data
    llm_df = pd.read_csv(llm_results_path)
    vid_df = pd.read_csv(video_emb_results_path)
    
    with open(categories_path, 'r') as f:
        cats = json.load(f)
        
    # Map video names to categories
    # The csvs usually have a 'video_name' column or similar. 
    # Let's inspect the first few rows if we were debugging, but here we assume standard format.
    # We need to ensure we can join on video name.
    
    def get_cat(vid_name):
        return cats.get(vid_name, {}).get('category', 'Unknown')
        
    # Merge
    # The CSVs are likely: video_id, masked, cos_sim_mean...
    
    # Combined DF
    llm_df['method'] = 'LLM'
    vid_df['method'] = 'Video'
    combined_all = pd.concat([llm_df, vid_df], ignore_index=True)
    combined_all['num_masked'] = combined_all['masked'].apply(lambda x: len(eval(x)))
    
    print("\nAvailable num_masked counts:")
    print(combined_all['num_masked'].value_counts().sort_index())

    targets = [6, 9, 12]
    
    for target_masked in targets:
        print(f"\n{'='*40}")
        print(f"ANALYZING num_masked == {target_masked}")
        print(f"{'='*40}")
        
        combined = combined_all[combined_all['num_masked'] == target_masked].copy()
        
        if combined.empty:
            print(f"No data for num_masked={target_masked}. Skipping.")
            continue
    
        # FILTER: Keep only (video_id, num_masked) present in BOTH methods to ensure fair ranking
        # Group by video_id and num_masked, count distinct methods
        counts = combined.groupby(['video_id', 'num_masked'])['method'].nunique()
        valid_groups = counts[counts == 2].index # (video_id, num_masked) tuples
        
        # Filter using join/merge is faster than apply
        valid_df = pd.DataFrame(valid_groups.tolist(), columns=['video_id', 'num_masked'])
        combined = pd.merge(combined, valid_df, on=['video_id', 'num_masked'])
        print(f"Filtered to {len(combined)} rows (intersection of videos per masking config).")
        
        # AGGREGATE: collapse multiple masking variations per video into a single score per (video, num_masked)
        # This ensures we are ranking "Videos" (1..100), not "Instances" (1..1000).
        agg_df = combined.groupby(['method', 'video_id', 'num_masked'])[metric].mean().reset_index()
        
        # Compute Rank on the aggregated data
        # Group by method and num_masked (e.g. rank all videos for LLM at num_masked=1)
        agg_df['rank'] = agg_df.groupby(['method', 'num_masked'])[metric].rank(method='min', ascending=False)
        
        # Now pivot to compare ranks for the same video & num_masked
        # (Since we already aggregated, video_id is unique per method/num_masked)
        llm_ranks = agg_df[agg_df['method'] == 'LLM'].groupby('video_id')['rank'].mean().rename('rank_llm')
        vid_ranks = agg_df[agg_df['method'] == 'Video'].groupby('video_id')['rank'].mean().rename('rank_vid')
        
        comparison = pd.concat([llm_ranks, vid_ranks], axis=1)
        comparison = comparison.dropna()
        comparison['category'] = comparison.index.map(get_cat)
        
        # Delta: Rank_Video - Rank_LLM
        comparison['delta'] = comparison['rank_vid'] - comparison['rank_llm']
        
        print("\nMean Rank Delta (Video - LLM) by Category (Positive = LLM Better):")
        delta_summary = comparison.groupby('category')['delta'].mean().sort_values(ascending=False)
        print(delta_summary)
        
        print("\nDetailed Statistics by Category:")
        stats = comparison.groupby('category')['delta'].agg(['count', 'mean', 'median', 'std', 'min', 'max']).sort_values('mean', ascending=False)
        print(stats)
        
        # Calculate counts
        counts = comparison['category'].value_counts()
        new_labels = [f"{cat}\n(n={counts[cat]})\nmean={delta_summary[cat]:.1f}" for cat in delta_summary.index]
        
        # Plotting
        plt.figure(figsize=(12, 7))
        # Boxplot with Mean Marker
        sns.boxplot(
            data=comparison, 
            x='category', 
            y='delta', 
            order=delta_summary.index, 
            palette='RdBu', 
            showfliers=False,
            showmeans=True,
            meanprops={"marker":"D","markerfacecolor":"yellow", "markeredgecolor":"black", "markersize": "14", "markeredgewidth": 2}
        )
        sns.stripplot(data=comparison, x='category', y='delta', order=delta_summary.index, color='black', alpha=0.6, jitter=0.2)
        
        plt.axhline(0, color='red', linewidth=1.5, linestyle='--', label='Video Better (Neg)')
        plt.title(f'Performance Rank Delta (Video - LLM) by Category\nMetric: {metric} (Higher = LLM Relatively Better)\nRanks computed per video (1 to N) at {target_masked} Masked Captions', fontsize=14)
        plt.ylabel('Rank Difference (Rank_Vid - Rank_LLM)', fontsize=12)
        plt.xlabel('Category', fontsize=12)
        plt.gca().set_xticklabels(new_labels)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        output_plot_path = output_dir / f'rank_delta_combined_masked_{target_masked}_{metric}.png'
        plt.savefig(output_plot_path)
        print(f"Saved rank delta plot to {output_plot_path}")
        
        # Negatives
        negatives = comparison[comparison['delta'] < 0].sort_values('delta')
        if not negatives.empty:
             negatives.to_csv(output_dir / f"video_wins_ranks_masked_{target_masked}.csv")

    
if __name__ == "__main__":
    # Define inputs (assuming cached results exist from previous steps)
    llm_path = "results/for_analysis/wild_dev_sim_one_shot_t=1.csv"
    vid_path = "results/for_analysis/wild_dev_sim_vec_vid.csv"
    cats_path = "results/video_categories.json"
    out_dir = "results/plots/conditional_analysis"
    
    # Check if files exist
    if not Path(llm_path).exists() or not Path(vid_path).exists():
        print("Results files not found. Please run valid experiments first.")
    else:
        run_conditional_analysis(llm_path, vid_path, cats_path, out_dir, metric='cos_sim_mean')
