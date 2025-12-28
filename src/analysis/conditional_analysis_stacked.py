import pandas as pd
import json
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pathlib import Path

def run_conditional_stacked_bar(
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
        
    def get_cat(vid_name):
        return cats.get(vid_name, {}).get('category', 'Unknown')
        
    # Combined DF
    llm_df['method'] = 'LLM'
    vid_df['method'] = 'Video'
    combined_all = pd.concat([llm_df, vid_df], ignore_index=True)
    combined_all['num_masked'] = combined_all['masked'].apply(lambda x: len(eval(x)))
    
    targets = [6, 9, 12]
    
    for target_masked in targets:
        print(f"\n{'='*40}")
        print(f"ANALYZING num_masked == {target_masked} (STACKED BAR)")
        print(f"{'='*40}")
        
        combined = combined_all[combined_all['num_masked'] == target_masked].copy()
        
        if combined.empty:
            continue
    
        counts = combined.groupby(['video_id', 'num_masked'])['method'].nunique()
        valid_groups = counts[counts == 2].index
        valid_df = pd.DataFrame(valid_groups.tolist(), columns=['video_id', 'num_masked'])
        combined = pd.merge(combined, valid_df, on=['video_id', 'num_masked'])
        
        agg_df = combined.groupby(['method', 'video_id', 'num_masked'])[metric].mean().reset_index()
        
        agg_df['rank'] = agg_df.groupby(['method', 'num_masked'])[metric].rank(method='min', ascending=False)
        
        llm_ranks = agg_df[agg_df['method'] == 'LLM'].groupby('video_id')['rank'].mean().rename('rank_llm')
        vid_ranks = agg_df[agg_df['method'] == 'Video'].groupby('video_id')['rank'].mean().rename('rank_vid')
        
        comparison = pd.concat([llm_ranks, vid_ranks], axis=1)
        comparison = comparison.dropna()
        comparison['category'] = comparison.index.map(get_cat)
        
        # Delta: Rank_LLM - Rank_Vid
        # Negative Delta = LLM Better (Rank_LLM < Rank_Vid)
        comparison['delta'] = comparison['rank_llm'] - comparison['rank_vid']
        
        # --- STACKED BAR CHART LOGIC ---
        
        # categorize rankings
        threshold = 10
        def categorize(d):
            if abs(d) <= threshold: return 'Agreement'
            if d < -threshold: return 'LLM Win' # Negative delta = LLM better
            return 'Video Win' # Positive delta = Video better
        
        comparison['Outcome'] = comparison['delta'].apply(categorize)
        
        # Count outcomes per category
        counts_df = comparison.pivot_table(index='category', columns='Outcome', values='delta', aggfunc='count', fill_value=0)
        
        # Ensure all columns exist
        for col in ['LLM Win', 'Agreement', 'Video Win']:
            if col not in counts_df.columns:
                counts_df[col] = 0
                
        # Reorder columns
        counts_df = counts_df[['LLM Win', 'Agreement', 'Video Win']]
        
        # Calculate percentages
        pct_df = counts_df.div(counts_df.sum(axis=1), axis=0) * 100
        
        # Sort by 'LLM Win' percentage (Descending)
        pct_df = pct_df.sort_values('LLM Win', ascending=False)
        
        # Update labels with N
        cat_counts = comparison['category'].value_counts()
        new_index = [f"{cat}\n(n={cat_counts[cat]})" for cat in pct_df.index]
        pct_df.index = new_index
        
        # Plotting
        ax = pct_df.plot(
            kind='bar', 
            stacked=True, 
            figsize=(12, 7), 
            color=['#D32F2F', '#4CAF50', '#1976D2'], # Red (LLM), Green (Agree), Blue (Video)
            width=0.75,
            edgecolor='black',
            linewidth=0.5
        )
        
        plt.title(
            f'Method Dominance by Category (num_masked={target_masked})\n'
            f'Metric: {metric} | Win Threshold: Rank Diff > {threshold}', 
            fontsize=14
        )
        plt.ylabel('Percentage of Videos', fontsize=12)
        plt.xlabel('Category', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.legend(title='Outcome', bbox_to_anchor=(1.02, 1), loc='upper left')
        plt.axhline(50, color='gray', linestyle=':', alpha=0.5)
        
        # Add value labels on the bars
        for c in ax.containers:
            labels = [f'{v.get_height():.0f}%' if v.get_height() > 5 else '' for v in c]
            ax.bar_label(c, labels=labels, label_type='center', color='white', fontweight='bold', fontsize=9)

        plt.tight_layout()
        
        output_plot_path = output_dir / f'method_dominance_stacked_masked_{target_masked}_{metric}.png'
        plt.savefig(output_plot_path)
        print(f"Saved stacked bar chart to {output_plot_path}")

    # --- Combined Plot Logic (Faceted) ---
    print(f"\n{'='*40}")
    print(f"GENERATING COMBINED PLOT FOR {targets} (FACETED)")
    print(f"{'='*40}")
    
    combined_targets_df = []
    
    for target_masked in targets:
        subset = combined_all[combined_all['num_masked'] == target_masked].copy()
        if subset.empty: continue
        
        counts = subset.groupby(['video_id', 'num_masked'])['method'].nunique()
        valid_groups = counts[counts == 2].index
        valid_df = pd.DataFrame(valid_groups.tolist(), columns=['video_id', 'num_masked'])
        subset = pd.merge(subset, valid_df, on=['video_id', 'num_masked'])
        
        agg_df = subset.groupby(['method', 'video_id', 'num_masked'])[metric].mean().reset_index()
        agg_df['rank'] = agg_df.groupby(['method', 'num_masked'])[metric].rank(method='min', ascending=False)
        
        llm = agg_df[agg_df['method'] == 'LLM'].groupby('video_id')['rank'].mean().rename('rank_llm')
        vid = agg_df[agg_df['method'] == 'Video'].groupby('video_id')['rank'].mean().rename('rank_vid')
        
        comp = pd.concat([llm, vid], axis=1).dropna()
        comp['category'] = comp.index.map(get_cat)
        # Ensure video_id column exists
        comp['video_id'] = comp.index
        comp['delta'] = comp['rank_llm'] - comp['rank_vid']
        comp['num_masked'] = target_masked
        
        comp['Outcome'] = comp['delta'].apply(lambda d: 'Agreement' if abs(d)<=10 else ('LLM Win' if d < -10 else 'Video Win'))
        combined_targets_df.append(comp)
        
    if not combined_targets_df:
        print("No data for combined plot.")
        return

    full_comparison = pd.concat(combined_targets_df, ignore_index=True)
    
    # Global Font Size Increase
    plt.rcParams.update({'font.size': 12, 'axes.titlesize': 14, 'axes.labelsize': 12})
    
    # Get unique categories and sort by dominance
    temp_counts = full_comparison.pivot_table(index=['category', 'num_masked'], columns='Outcome', values='delta', aggfunc='count', fill_value=0)
    for col in ['LLM Win', 'Agreement', 'Video Win']: 
        if col not in temp_counts.columns: temp_counts[col] = 0
    temp_counts = temp_counts[['LLM Win', 'Agreement', 'Video Win']]
    temp_pct = temp_counts.div(temp_counts.sum(axis=1), axis=0) * 100
    
    cat_order = temp_pct['LLM Win'].groupby('category').mean().sort_values(ascending=False).index.tolist()
    
    # Setup Subplots
    num_cats = len(cat_order)
    cols = 3
    rows = (num_cats + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(18, 6 * rows), sharey=True)
    axes = axes.flatten()
    
    colors = ['#D32F2F', '#4CAF50', '#1976D2'] # LLM, Agree, Video
    
    for i, cat in enumerate(cat_order):
        ax = axes[i]
        
        # Prepare data for this category
        cat_data = full_comparison[full_comparison['category'] == cat]
        
        # Calculate N (unique videos in this category)
        # Note: Since we have multiple num_masked, n refers to unique video_ids
        n_videos = cat_data['video_id'].nunique()
        
        counts = cat_data.pivot_table(index='num_masked', columns='Outcome', values='delta', aggfunc='count', fill_value=0)
        
        # Ensure cols exist
        for col in ['LLM Win', 'Agreement', 'Video Win']: 
            if col not in counts.columns: counts[col] = 0
        counts = counts[['LLM Win', 'Agreement', 'Video Win']]
        
        # Pct
        pcts = counts.div(counts.sum(axis=1), axis=0) * 100
        
        # Plot stacked bar on subplot
        pcts.plot(kind='bar', stacked=True, ax=ax, color=colors, width=0.6, edgecolor='black', linewidth=0.5, legend=False)
        
        ax.set_title(f"{cat} (n={n_videos})", fontsize=16, fontweight='bold')
        ax.set_xlabel("Num Masked", fontsize=12)
        ax.set_xticklabels(pcts.index, rotation=0, fontsize=11)
        ax.tick_params(axis='y', labelsize=11)
        ax.axhline(50, color='gray', linestyle=':', alpha=0.5)
        
        # Annotate
        for c in ax.containers:
            labels = [f'{v.get_height():.0f}%' if v.get_height() > 10 else '' for v in c]
            ax.bar_label(c, labels=labels, label_type='center', color='white', fontweight='bold', fontsize=10)

    # Hide unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')
        
    # Global Legend at Bottom
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, title='Outcome', loc='lower center', bbox_to_anchor=(0.5, 0.0), ncol=3, fontsize=14, title_fontsize=14, frameon=False)
    
    fig.suptitle(f"Method Dominance Evolution by Category (Faceted)\nMetric: {metric}", fontsize=20, y=0.98)
    
    # Adjust layout to make room for legend at bottom
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.12, top=0.90) 
    
    out_path = output_dir / f'method_dominance_combined_faceted_{metric}.png'
    plt.savefig(out_path, bbox_inches='tight')
    print(f"Saved combined faceted chart to {out_path}")

    
if __name__ == "__main__":
    llm_path = "results/for_analysis/wild_dev_sim_one_shot_t=1.csv"
    vid_path = "results/for_analysis/wild_dev_sim_vec_vid.csv"
    cats_path = "results/video_categories.json"
    out_dir = "results/plots/conditional_analysis"
    
    if not Path(llm_path).exists() or not Path(vid_path).exists():
        print("Results files not found.")
    else:
        run_conditional_stacked_bar(llm_path, vid_path, cats_path, out_dir, metric='cos_sim_mean')
