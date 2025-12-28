import pandas as pd
import json
import matplotlib.pyplot as plt
import numpy as np
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
        
        # Delta: Rank_LLM - Rank_Vid
        # Consistent with Rank Scatter Plot:
        # Negative Delta (Rank_LLM < Rank_Vid) = LLM Better (Box)
        # Positive Delta (Rank_LLM > Rank_Vid) = Video Better (V)
        comparison['delta'] = comparison['rank_llm'] - comparison['rank_vid']
        
        print("\nMean Rank Delta (LLM - Video) by Category (Negative = LLM Better):")
        # Sort ascending (most negative / LLM best first)
        delta_summary = comparison.groupby('category')['delta'].mean().sort_values(ascending=True)
        print(delta_summary)
        
        print("\nDetailed Statistics by Category:")
        stats = comparison.groupby('category')['delta'].agg(['count', 'mean', 'median', 'std', 'min', 'max']).sort_values('mean', ascending=True)
        print(stats)
        
        # Calculate counts
        counts = comparison['category'].value_counts()
        new_labels = [f"{cat}\n(n={counts[cat]})\nmean={delta_summary[cat]:.1f}" for cat in delta_summary.index]
        
        # Plotting
        plt.figure(figsize=(14, 8))
        # Boxplot with Mean Marker
        sns.boxplot(
            data=comparison, 
            x='category', 
            y='delta', 
            order=delta_summary.index, 
            palette='RdBu_r', # Reversed so Blue (cold) is negative (LLM better), Red (hot) is positive? Or just aesthetic.
            showfliers=False,
            showmeans=True,
            meanprops={"marker":"D","markerfacecolor":"white", "markeredgecolor":"black", "markersize": "10", "markeredgewidth": 2},
            boxprops=dict(alpha=.3) # Lighter boxplot to make points visible
        )
        
        # Custom Stripplot Logic for Markers
        # Threshold for markers
        threshold = 10
        cat_map = {cat: i for i, cat in enumerate(delta_summary.index)}
        
        for cat in delta_summary.index:
            x_center = cat_map[cat]
            subset = comparison[comparison['category'] == cat]
            
            # Helper for jitter
            def get_jittered_x(n): 
                return np.random.normal(x_center, 0.08, n)

            # 1. Agreement (Green Circle)
            agree = subset[subset['delta'].abs() <= threshold]
            if not agree.empty:
                plt.scatter(
                    get_jittered_x(len(agree)), 
                    agree['delta'], 
                    marker='o', 
                    color='green', 
                    edgecolor='black',
                    linewidth=0.5,
                    alpha=0.7, 
                    s=40,
                    label='Agreement' if x_center == 0 else ""
                )
            
            # 2. LLM Better (Red Square "Box") -> Delta < -10
            llm_better = subset[subset['delta'] < -threshold]
            if not llm_better.empty:
                plt.scatter(
                    get_jittered_x(len(llm_better)), 
                    llm_better['delta'], 
                    marker='s', 
                    color='red', 
                    edgecolor='black',
                    linewidth=0.5,
                    alpha=0.7, 
                    s=40,
                     label='LLM ranked higher' if x_center == 0 else ""
                )
            
            # 3. Video Better (Red V) -> Delta > 10
            vid_better = subset[subset['delta'] > threshold]
            if not vid_better.empty:
                plt.scatter(
                    get_jittered_x(len(vid_better)), 
                    vid_better['delta'], 
                    marker='v', 
                    color='red', 
                    edgecolor='black',
                    linewidth=0.5,
                    alpha=0.7, 
                    s=40,
                    label='Video ranked higher' if x_center == 0 else ""
                )
        
        plt.axhline(0, color='gray', linewidth=1.5, linestyle='--', label='Zero Diff')
        
        # Adjust Title and Labels
        plt.title(
            f'Rank Difference by Category (LLM Rank - Video Rank)\n'
            f'Metric: {metric} | Negative values = %s Ranked Higher (Better)' % "LLM", 
            fontsize=14
        )
        plt.ylabel(f'Rank Difference ({metric})\n(Negative = LLM Better, Positive = Video Better)', fontsize=12)
        plt.xlabel('Category', fontsize=12)
        plt.gca().set_xticklabels(new_labels)
        plt.xticks(rotation=45)
        
        # Legend construction - tricky with manually added scatters. 
        # But we added labels to the first iteration. However, duplicates might appear or not.
        # Let's create custom legend elements to be safe.
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', label='Agreement (diff ≤ 10)',
                   markerfacecolor='green', markersize=8),
            Line2D([0], [0], marker='s', color='w', label='LLM Ranked Higher',
                   markerfacecolor='red', markersize=8),
            Line2D([0], [0], marker='v', color='w', label='Video Ranked Higher',
                   markerfacecolor='red', markersize=8)
        ]
        # plt.legend(handles=legend_elements, loc='best')

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
