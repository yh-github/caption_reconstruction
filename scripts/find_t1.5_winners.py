
import pandas as pd

INPUT_CSV = "results/temporal_metrics_final.csv"

def main():
    df = pd.read_csv(INPUT_CSV)
    
    # Filter for repetition_penalty = 1.2
    df = df[(df['repetition_penalty'] == 1.2) & (df['width'].between(3, 12))]
    
    metric = 'phi_recall_at_1'
    
    # Group by video and temperature
    vid_temp_scores = df.groupby(['video_id', 'temperature'])[metric].mean().reset_index()
    pivot_scores = vid_temp_scores.pivot(index='video_id', columns='temperature', values=metric)
    
    pivot_scores['Oracle'] = pivot_scores.max(axis=1)
    
    def get_winners(row):
        temps = [c for c in row.index if c != 'Oracle']
        best_val = row['Oracle']
        winners = [t for t in temps if abs(row[t] - best_val) < 1e-6]
        return winners

    pivot_scores['winners'] = pivot_scores.apply(get_winners, axis=1)
    
    # Find rows where 1.5 is a winner
    t15_wins_df = pivot_scores[pivot_scores['winners'].apply(lambda x: 1.5 in x)]
    
    print("Videos where T=1.5 is a winner (Recall@1):")
    print(t15_wins_df)
    
    # Save IDs to file for next step
    with open("results/t1.5_winners.txt", "w") as f:
        for vid in t15_wins_df.index:
            f.write(f"{vid}\n")

if __name__ == "__main__":
    main()
