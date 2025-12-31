import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def run_consistency_null_test():
    llm_path = "results/for_analysis/wild_dev_sim_one_shot_t=1.csv"
    vid_path = "results/for_analysis/wild_dev_sim_vec_vid.csv"
    
    print("Loading data...")
    llm_df = pd.read_csv(llm_path)
    vid_df = pd.read_csv(vid_path)
    
    # Standardize
    llm_df['method'] = 'LLM'
    vid_df['method'] = 'Video'
    combined = pd.concat([llm_df, vid_df], ignore_index=True)
    combined['num_masked'] = combined['masked'].apply(lambda x: len(eval(x)))
    
    # Calculate Ranks per mask level
    # We want a DataFrame: index=video_id, cols=mask_levels (6,9,12,15), values=Delta
    
    mask_levels = [6, 9, 12, 15]
    all_deltas = []
    
    for m in mask_levels:
        subset = combined[combined['num_masked'] == m].copy()
        counts = subset.groupby('video_id')['method'].nunique()
        valid_vids = counts[counts == 2].index
        subset = subset[subset['video_id'].isin(valid_vids)]
        
        if subset.empty: continue
        
        agg = subset.groupby(['method', 'video_id'])['cos_sim_mean'].mean().reset_index()
        agg['rank'] = agg.groupby('method')['cos_sim_mean'].rank(method='first', ascending=False)
        pivoted = agg.pivot(index='video_id', columns='method', values='rank')
        pivoted['delta'] = pivoted['LLM'] - pivoted['Video']
        
        # Rename column to include mask level
        pivoted = pivoted[['delta']].rename(columns={'delta': f'delta_{m}'})
        all_deltas.append(pivoted)
        
    # Join all mask levels
    full_df = pd.concat(all_deltas, axis=1)
    
    # Keep only videos present in ALL mask levels for fair "Consistency" check
    full_df = full_df.dropna() 
    n_videos = len(full_df)
    n_levels = len(mask_levels)
    
    print(f"Analysis on {n_videos} videos present in all {n_levels} conditions.")
    
    # 1. Real Average Delta
    real_avg_delta = full_df.mean(axis=1)
    
    # 2. Simulated Average Delta (Random Noise)
    # We simulate N_videos ranking randomly across N_levels trials, and average them.
    n_sims = 5000
    sim_avgs = []
    
    for _ in range(n_sims):
        # For one "Simulated Dataset":
        # Create a (n_videos, n_levels) matrix of random deltas
        # A random delta is (RandPerm - RandPerm)
        deltas = np.zeros((n_videos, n_levels))
        for j in range(n_levels):
            r1 = np.random.permutation(n_videos) + 1
            r2 = np.random.permutation(n_videos) + 1
            deltas[:, j] = r1 - r2
        
        # Average across the "levels"
        avg_deltas = deltas.mean(axis=1)
        sim_avgs.extend(avg_deltas) # Collect all 'video' outcomes
        
    sim_avgs = np.array(sim_avgs)
    
    # --- Plotting ---
    plt.figure(figsize=(10, 6))
    
    # Density
    sns.kdeplot(sim_avgs, color='black', linestyle='--', linewidth=2, label='Null Hypothesis\n(Random Noise averaged over 4 runs)', fill=True, alpha=0.1)
    sns.kdeplot(real_avg_delta, color='#D32F2F', linewidth=3, label='Real Data\n(Consistently Biased averaged over 4 runs)', fill=True, alpha=0.2)
    
    plt.xlabel(f"Average Rank Delta (across k={mask_levels})")
    plt.ylabel("Density")
    plt.title("Signal vs Noise: Do the Biases Persist?\n(Averaging across multiple experiments)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    out_path = "results/plots/null_hypothesis_consistency.png"
    plt.savefig(out_path, bbox_inches='tight')
    print(f"Saved plot to {out_path}")
    
    # Stats
    print(f"Real Std: {real_avg_delta.std():.2f}")
    print(f"Null Std: {np.std(sim_avgs):.2f}")

if __name__ == "__main__":
    run_consistency_null_test()
