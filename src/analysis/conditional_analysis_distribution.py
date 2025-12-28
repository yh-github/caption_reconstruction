import pandas as pd
import json
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pathlib import Path

def run_conditional_distribution_plot(
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
    
    targets = [6, 9, 12, 15] 
    
    print(f"\n{'='*40}")
    print(f"GENERATING FACETED DISTRIBUTION PLOT FOR {targets}")
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
        comp['delta'] = comp['rank_llm'] - comp['rank_vid'] # Neg=LLM Better
        comp['num_masked'] = target_masked
        comp['video_id'] = comp.index
        
        combined_targets_df.append(comp)
        
    if not combined_targets_df:
        print("No data for combined plot.")
        return

    full_comparison = pd.concat(combined_targets_df, ignore_index=True)
    
    # Global Parameters
    plt.rcParams.update({'font.size': 12, 'axes.titlesize': 14, 'axes.labelsize': 11})
    
    # Determine Category Order (e.g. by mean delta across all masks, most negative/LLM-better first)
    cat_order = full_comparison.groupby('category')['delta'].mean().sort_values(ascending=True).index.tolist()
    
    # Setup Subplots
    num_cats = len(cat_order)
    cols = 3
    rows = (num_cats + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(18, 5 * rows), sharey=True)
    axes = axes.flatten()
    
    # Define colors/params
    violin_color = '#E0F2F1' # Very light teal
    point_color = 'black'
    
    for i, cat in enumerate(cat_order):
        ax = axes[i]
        cat_data = full_comparison[full_comparison['category'] == cat]
        n_videos = cat_data['video_id'].nunique()
        
        
        # Add Shaded Agreement Zone (darker) with boundary lines
        threshold = 10 
        ax.axhspan(-threshold, threshold, color='green', alpha=0.15, zorder=0, label=f'Agreement Zone (±{threshold})' if i==0 else "")
        ax.axhline(threshold, color='green', linestyle='--', linewidth=0.8, alpha=0.6, zorder=0)
        ax.axhline(-threshold, color='green', linestyle='--', linewidth=0.8, alpha=0.6, zorder=0)
        ax.axhline(0, color='gray', linestyle=':', alpha=0.5, zorder=0)

        # Violin Plot for Distribution
        sns.violinplot(
            data=cat_data,
            x='num_masked',
            y='delta',
            ax=ax,
            inner=None, # Clean violin
            color=violin_color,
            linewidth=0,
            alpha=0.4,
            width=0.7
        )
        
        # Categorical Scatter (Stripplot) on top
        def get_color(d):
            if d < -threshold: return '#D32F2F' # Red (LLM Win)
            if d > threshold: return '#1976D2' # Blue (Vid Win)
            return '#4CAF50' # Green (Agree)
            
        point_colors = cat_data['delta'].apply(get_color).tolist()
        
        unique_masks = sorted(cat_data['num_masked'].unique())
        mask_map = {m: i for i, m in enumerate(unique_masks)}
        
        # Annotate density (counts) manually since points overlap
        for m_idx, mask_val in enumerate(unique_masks):
            mask_subset = cat_data[cat_data['num_masked'] == mask_val]
            n_llm = len(mask_subset[mask_subset['delta'] < -threshold])
            n_vid = len(mask_subset[mask_subset['delta'] > threshold])
            
            # Text annotation at top/bottom of column
            # LLM Count (Bottom, Red)
            ax.text(m_idx, -95, f"{n_llm}", ha='center', va='bottom', color='#D32F2F', fontsize=10, fontweight='bold')
            # Video Count (Top, Blue)
            ax.text(m_idx, 95, f"{n_vid}", ha='center', va='top', color='#1976D2', fontsize=10, fontweight='bold')

        
        for idx, row in cat_data.iterrows():
            x_base = mask_map[row['num_masked']]
            x_jit = x_base + np.random.normal(0, 0.05)
            ax.scatter(x_jit, row['delta'], color=get_color(row['delta']), s=25, alpha=0.7, edgecolors='white', linewidth=0.5)

        # Add Trend Line (Overall Mean)
        means = cat_data.groupby('num_masked')['delta'].mean()
        ax.plot(range(len(means)), means.values, color='black', marker='D', markersize=6, linewidth=2, label='Overall Mean' if i==0 else "", alpha=0.9, zorder=10)

        # Add Sub-trend Lines (Mean of Positives/Negatives)
        # Reindex to ensure we have values for all x-ticks even if some are empty
        all_masks = sorted(cat_data['num_masked'].unique())
        
        # Mean of Video Wins (> Threshold)
        pos_means = cat_data[cat_data['delta'] > threshold].groupby('num_masked')['delta'].mean().reindex(all_masks)
        ax.plot(range(len(pos_means)), pos_means.values, color='#1976D2', linestyle='--', linewidth=1.5, marker='v', markersize=4, label='Mean Video Win' if i==0 else "", alpha=0.8)
        
        # Mean of LLM Wins (< -Threshold)
        neg_means = cat_data[cat_data['delta'] < -threshold].groupby('num_masked')['delta'].mean().reindex(all_masks)
        ax.plot(range(len(neg_means)), neg_means.values, color='#D32F2F', linestyle='--', linewidth=1.5, marker='s', markersize=4, label='Mean LLM Win' if i==0 else "", alpha=0.8)


        # Formatting
        ax.set_title(f"{cat} (n={n_videos})", fontsize=16, fontweight='bold')
        ax.set_xticks(range(len(unique_masks)))
        ax.set_xticklabels(unique_masks)
        ax.set_xlabel("Num Masked" if i >= num_cats-cols else "", fontsize=10)
        
        if i % cols == 0:
            ax.set_ylabel("Rank Delta\n(Neg=LLM Better)", fontsize=11)
            
        # Strict Y-axis limit
        ax.set_ylim(-105, 105)

    # Hide unused
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')
        
    # Custom Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='LLM Win (<-10)', markerfacecolor='#D32F2F', markersize=8),
        Line2D([0], [0], marker='o', color='w', label='Agreement (±10)', markerfacecolor='#4CAF50', markersize=8),
        Line2D([0], [0], marker='o', color='w', label='Video Win (>10)', markerfacecolor='#1976D2', markersize=8),
        Line2D([0], [0], color='black', marker='D', label='Overall Mean', markersize=6),
        Line2D([0], [0], color='#D32F2F', linestyle='--', marker='s', label='Mean LLM Win', markersize=5),
        Line2D([0], [0], color='#1976D2', linestyle='--', marker='v', label='Mean Video Win', markersize=5),
        Line2D([0], [0], color='green', alpha=0.15, linewidth=4, label='Agreement Zone')
    ]
    
    fig.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, 0.0), ncol=4, fontsize=11, frameon=False)
    
    fig.suptitle(f"Rank Difference Distribution by Category & Masking\n(Violin + Strip Plot)", fontsize=18, y=0.98)
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.10, top=0.90)
    
    out_path = output_dir / f'rank_delta_distribution_faceted_{metric}.png'
    plt.savefig(out_path, bbox_inches='tight')
    print(f"Saved distribution plot to {out_path}")

    
if __name__ == "__main__":
    llm_path = "results/for_analysis/wild_dev_sim_one_shot_t=1.csv"
    vid_path = "results/for_analysis/wild_dev_sim_vec_vid.csv"
    cats_path = "results/video_categories.json"
    out_dir = "results/plots/conditional_analysis"
    
    if not Path(llm_path).exists() or not Path(vid_path).exists():
        print("Results files not found.")
    else:
        run_conditional_distribution_plot(llm_path, vid_path, cats_path, out_dir, metric='cos_sim_mean')
