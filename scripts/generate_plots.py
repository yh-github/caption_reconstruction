
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Set style
sns.set_theme(style="whitegrid")

CSV_PATH = "results/deep_analysis_final.csv"
OUTPUT_DIR = "results/plots"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def main():
    if not os.path.exists(CSV_PATH):
        print(f"CSV not found: {CSV_PATH}")
        return

    df = pd.read_csv(CSV_PATH)
    # Filter for relevant widths for the paper
    df = df[df['width'] <= 12]
    print(f"Loaded {len(df)} rows (filtered width <= 12).")

    # 1. MRR vs Width (Grouped by Index)
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x="width", y="phi_mrr", hue="index", marker="o", palette="tab10")
    
    # Plot baseline for w=3 (since we only have that data point really)
    # We take the mean vec_mrr for w=3 cases if possible, or global mean
    baseline_val = df['vec_mrr'].mean()
    plt.axhline(y=baseline_val, color='r', linestyle='--', label=f"Vec Baseline (w=3 mostly, avg={baseline_val:.2f})")
    
    plt.title("Phi-3 MRR vs Mask Width (per Start Index)")
    plt.ylabel("MRR")
    plt.xlabel("Mask Width (Frames)")
    plt.xticks([3, 6, 9, 12])
    plt.legend(title="Start Index")
    plt.savefig(f"{OUTPUT_DIR}/mrr_vs_width_per_index.png")
    plt.close()
    
    # 2. MRR vs Index (Grouped by Width - for selected widths)
    plt.figure(figsize=(10, 6))
    # Select a few key widths to avoid clutter
    selected_widths = [3, 6, 9, 12]
    subset = df[df['width'].isin(selected_widths)]
    
    sns.barplot(data=subset, x="index", y="phi_mrr", hue="width", palette="viridis")
    plt.axhline(y=baseline_val, color='r', linestyle='--', label="Vec Baseline (w=3)")
    plt.title("Phi-3 MRR vs Position (grouped by Width)")
    plt.ylabel("MRR")
    plt.xlabel("Start Index")
    plt.legend(title="Width")
    plt.savefig(f"{OUTPUT_DIR}/mrr_vs_index_per_width.png")
    plt.close()

    # 2.5 Mean Rank vs Median Rank vs Width (Phi-3)
    plt.figure(figsize=(10, 6))
    rank_df = df.groupby('width')[['phi_mean_rank', 'phi_median_rank']].mean().reset_index()
    melted_rank = rank_df.melt(id_vars='width', var_name='Metric', value_name='Rank')
    sns.lineplot(data=melted_rank, x="width", y="Rank", hue="Metric", marker="o")
    plt.title("Phi-3 Mean Rank vs Median Rank vs Width")
    plt.ylabel("Rank (Lower is Better)")
    plt.xlabel("Mask Width")
    plt.xticks([3, 6, 9, 12])
    plt.savefig(f"{OUTPUT_DIR}/mean_vs_median_rank.png")
    plt.close()
    
    # 2.6 Temporal Metrics Comparison
    # Load separate temporal CSV if available
    TEMP_CSV = "results/temporal_metrics_final.csv"
    BASELINE_CSV = "results/baseline_full_metrics.csv"
    
    if os.path.exists(TEMP_CSV):
        temp_df = pd.read_csv(TEMP_CSV)
        temp_df = temp_df[temp_df['width'] <= 12] # Filter
        
        # Prepare Phi-3 Data
        # We want to compare standard Phi MRR (phi_mrr) vs Temporal NDCG vs R@1_w1
        phi_stats = temp_df.groupby('width')[['phi_mrr', 'phi_recall_at_1', 'temporal_recall_at_1_w1', 'temporal_ndcg']].mean().reset_index()
        phi_stats['Method'] = 'Phi-3'
        
        baseline_stats = None
        if os.path.exists(BASELINE_CSV):
            base_df = pd.read_csv(BASELINE_CSV)
            base_df = base_df[base_df['width'] <= 12] # Filter
            
            # rename for alignment
            # base: mrr, recall_at_1, temporal_recall_at_1_w1, temporal_ndcg
            # normalize names to match phi_stats columns or rename both to generic
            # Let's rename generically
            
            base_agg = base_df.groupby('width')[['mrr', 'recall_at_1', 'temporal_recall_at_1_w1', 'temporal_ndcg']].mean().reset_index()
            base_agg = base_agg.rename(columns={
                'mrr': 'phi_mrr', 
                'recall_at_1': 'phi_recall_at_1'
            })
            base_agg['Method'] = 'Baseline (Vec)'
            
            # Combine
            combined = pd.concat([phi_stats, base_agg])
        else:
            combined = phi_stats

        # Plotting
        # We want to show: MRR, Temporal NDCG for both methods.
        # Maybe separate plots or faceted?
        # Let's do huge comparison: Score vs Width, Hue=Metric, Style=Method
        
        melted = combined.melt(id_vars=['width', 'Method'], var_name='Metric', value_name='Score')
        
        # Filter metrics to keep plot clean
        # Let's show: MRR, Temporal NDCG
        target_metrics = ['phi_mrr', 'temporal_ndcg']
        melted = melted[melted['Metric'].isin(target_metrics)]
        
        friendly_names = {
            'phi_mrr': 'MRR (Standard)',
            'temporal_ndcg': 'Temporal NDCG'
        }
        melted['Metric'] = melted['Metric'].map(friendly_names)
        
        plt.figure(figsize=(10, 6))
        sns.lineplot(data=melted, x="width", y="Score", hue="Metric", style="Method", markers=True, dashes=False)
        plt.title("Phi-3 vs Baseline: MRR vs Temporal NDCG")
        plt.ylabel("Score")
        plt.xlabel("Mask Width")
        plt.xticks([3, 6, 9, 12])
        plt.savefig(f"{OUTPUT_DIR}/temporal_metrics_phis_vs_base.png")
        plt.close()
        
        # Also plot R@1 Comparison
        # Phi R@1, Phi Temp R@1, Base R@1, Base Temp R@1
        melted_r1 = combined.melt(id_vars=['width', 'Method'], var_name='Metric', value_name='Score')
        r1_metrics = ['phi_recall_at_1', 'temporal_recall_at_1_w1']
        melted_r1 = melted_r1[melted_r1['Metric'].isin(r1_metrics)]
        
        friendly_names_r1 = {
            'phi_recall_at_1': 'R@1 (Exact)',
            'temporal_recall_at_1_w1': 'R@1 (Window=1)'
        }
        melted_r1['Metric'] = melted_r1['Metric'].map(friendly_names_r1)
        
        plt.figure(figsize=(10, 6))
        sns.lineplot(data=melted_r1, x="width", y="Score", hue="Metric", style="Method", markers=True, dashes=False)
        plt.title("Phi-3 vs Baseline: Exact vs Windowed Recall@1")
        plt.ylabel("Recall")
        plt.xlabel("Mask Width")
        plt.xticks([3, 6, 9, 12])
        plt.savefig(f"{OUTPUT_DIR}/temporal_metrics_r1_compare.png")
        plt.close()
        
        print(f"Temporal comparison plots saved to {OUTPUT_DIR}")

    # 3. Category Delta (Keep existing but update title)
    cat_df = df.groupby('category')[['mrr_delta']].mean().reset_index().sort_values('mrr_delta', ascending=False)
    plt.figure(figsize=(12, 6))
    params = {'data': cat_df, 'x': 'category', 'y': 'mrr_delta', 'palette': 'viridis'}
    try:
        params['hue'] = 'category'
        params['legend'] = False
        sns.barplot(**params)
    except:
        del params['hue']
        del params['legend']
        sns.barplot(**params)
        
    plt.axhline(0, color='black', linewidth=1)
    plt.xticks(rotation=45, ha='right')
    plt.title("Mean MRR Improvement (Phi-3 vs Vec Baseline w=3) by Category")
    plt.ylabel("Delta MRR")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/category_delta.png")
    plt.close()
    
    print(f"Plots saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
