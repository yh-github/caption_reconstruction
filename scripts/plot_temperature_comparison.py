
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

INPUT_CSV = "results/temporal_metrics_final.csv"
OUTPUT_DIR = "results/plots/temperature_analysis"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def main():
    if not os.path.exists(INPUT_CSV):
        print(f"Error: {INPUT_CSV} not found.")
        return

    df = pd.read_csv(INPUT_CSV)
    
    # Filter for repetition_penalty = 1.2
    target_rp = 1.2
    df_filtered = df[(df['repetition_penalty'] == target_rp) & (df['width'].between(3, 12))]
    
    if df_filtered.empty:
        print(f"No data found for repetition_penalty={target_rp} and width in [3, 12].")
        return

    print(f"Filtered data shape: {df_filtered.shape}")
    metrics_to_analyze = ['phi_mrr', 'temporal_ndcg', 'phi_recall_at_1', 'temporal_recall_at_1_w1']
    
    with open(f"{OUTPUT_DIR}/temperature_stats_all_metrics.txt", "w") as f:
        f.write("--- Temperature Analysis Stats ---\n")

    for metric in metrics_to_analyze:
        print(f"\nAnalyzing {metric}...")
        
        # --- Bar Chart ---
        plt.figure(figsize=(10, 6))
        sns.barplot(data=df_filtered, x='width', y=metric, hue='temperature', palette='viridis', errorbar=None)
        plt.title(f"{metric} vs Mask Width (RP={target_rp})")
        plt.xlabel("Mask Width")
        plt.ylabel(metric)
        plt.grid(axis='y')
        plt.tight_layout()
        plt.savefig(f"{OUTPUT_DIR}/{metric}_vs_width_bars_rp{target_rp}.png")
        plt.close()

        # --- Oracle Analysis (Per Video) ---
        vid_temp_scores = df_filtered.groupby(['video_id', 'temperature'])[metric].mean().reset_index()
        pivot_scores = vid_temp_scores.pivot(index='video_id', columns='temperature', values=metric)
        
        # Best-of-All Oracle per video
        pivot_scores['Oracle'] = pivot_scores.max(axis=1)
        
        # Average comparison
        avg_scores = pivot_scores.mean()
        
        # Save Oracle Comparison Plot
        plt.figure(figsize=(8, 5))
        ax = sns.barplot(x=avg_scores.index, y=avg_scores.values, palette='magma')
        ax.bar_label(ax.containers[0], fmt='%.4f')
        plt.title(f"Average {metric}: Individual Temperatures vs Oracle")
        plt.ylabel(f"Mean {metric}")
        plt.tight_layout()
        plt.savefig(f"{OUTPUT_DIR}/oracle_comparison_{metric}_rp{target_rp}.png")
        plt.close()

        # --- Win/Tie Stats ---
        def get_winners(row):
            temps = [c for c in row.index if c != 'Oracle']
            best_val = row['Oracle']
            winners = [t for t in temps if abs(row[t] - best_val) < 1e-6]
            return winners

        pivot_scores['winners'] = pivot_scores.apply(get_winners, axis=1)
        # Check specifically for T=1.5 wins
        t15_wins = pivot_scores['winners'].apply(lambda x: 1.5 in x).sum()
        
        all_winners = [t for sublist in pivot_scores['winners'] for t in sublist]
        win_counts = pd.Series(all_winners).value_counts()
        total_ties = pivot_scores['winners'].apply(lambda x: len(x) > 1).sum()
        
        print(f"Win Counts for {metric}:")
        print(win_counts)
        print(f"T=1.5 Wins: {t15_wins}")

        # Append to stats file
        with open(f"{OUTPUT_DIR}/temperature_stats_all_metrics.txt", "a") as f:
            f.write(f"\n\n=== Metric: {metric} ===\n")
            f.write("Average Scores:\n")
            f.write(avg_scores.to_string())
            f.write("\n\nWin Counts:\n")
            f.write(win_counts.to_string())
            f.write(f"\n\nT=1.5 Specific Wins: {t15_wins}/{len(pivot_scores)}")
            f.write(f"\nTotal Ties: {total_ties}")

if __name__ == "__main__":
    main()
