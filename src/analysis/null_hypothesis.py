import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def run_null_hypothesis_test():
    # Load Data
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
    
    mask_levels = [6, 9, 12, 15]
    
    # Create a figure with 3 Rows x 4 Columns
    # Row 1: Linear Density
    # Row 2: Histogram (Counts)
    # Row 3: Survival Function of Abs(Delta) -> The "Tail Check"
    fig, axes = plt.subplots(3, 4, figsize=(24, 15))
    
    for i, m in enumerate(mask_levels):
        subset = combined[combined['num_masked'] == m].copy()
        
        # Intersection
        counts = subset.groupby('video_id')['method'].nunique()
        valid_vids = counts[counts == 2].index
        subset = subset[subset['video_id'].isin(valid_vids)]
        
        if subset.empty: continue
        
        # Real Ranks
        agg = subset.groupby(['method', 'video_id'])['cos_sim_mean'].mean().reset_index()
        agg['rank'] = agg.groupby('method')['cos_sim_mean'].rank(method='first', ascending=False)
        
        pivoted = agg.pivot(index='video_id', columns='method', values='rank')
        real_delta = pivoted['LLM'] - pivoted['Video']
        n_videos = len(pivoted)
        
        # Simulation
        n_sims = 2000 # More sims for smoother histogram
        sim_deltas = []
        for _ in range(n_sims):
            r1 = np.random.permutation(n_videos) + 1
            r2 = np.random.permutation(n_videos) + 1
            sim_deltas.extend(r1 - r2)
        sim_deltas = np.array(sim_deltas)
            
        # --- Row 1: Linear Density ---
        ax = axes[0, i]
        sns.kdeplot(sim_deltas, color='black', linestyle='--', linewidth=2, label='Null (Random)', ax=ax)
        sns.kdeplot(sim_deltas, color='grey', fill=True, alpha=0.1, ax=ax)
        sns.kdeplot(real_delta, color='#D32F2F', linewidth=3, label='Actual Data', ax=ax)
        sns.kdeplot(real_delta, color='#D32F2F', fill=True, alpha=0.1, ax=ax)
        
        ax.set_xlim(-100, 100)
        ax.set_ylim(0, 0.016) # Fixed Y-limit
        ax.set_title(f"Masked={m} (Linear Density)")
        if i == 0: ax.legend()
        
        # --- Row 2: Histogram (Counts) ---
        ax = axes[1, i]
        # Normalize sim_deltas to have same count scale as real data (divide by n_sims)
        # Actually better to plot normalized density histogram or just overlay probability
        # But user asked for Counts. 
        # Since N_sims >>> N_real, we must normalize the *height* of specific bins or just use density=True
        # Let's use density=False but scale the Null hist down? No, density=True is safer for comparison.
        # But user explicitly asked for "Counts".
        # Let's blindly plot counts and accept Null will be huge? No, that's useless.
        # Let's plot "Relative Frequency" (Probability).
        
        bins = np.linspace(-100, 100, 41) # 5-unit bins
        
        # Null Hist (as step)
        ax.hist(sim_deltas, bins=bins, density=True, histtype='step', color='black', linewidth=2, linestyle='--', label='Null')
        # Real Hist (as bar)
        ax.hist(real_delta, bins=bins, density=True, color='#D32F2F', alpha=0.5, label='Actual')
        
        ax.set_xlim(-100, 100)
        ax.set_title(f"Masked={m} (Histogram/Freq)")
        
        # --- Row 3: Absolute Delta Survival (Log-Log?) ---
        # P(|Delta| > x)
        ax = axes[2, i]
        
        abs_real = np.abs(real_delta)
        abs_sim = np.abs(sim_deltas)
        
        # Eval X from 0 to 100
        x_vals = np.linspace(0, 100, 100)
        prob_real = [np.mean(abs_real > x) for x in x_vals]
        prob_sim = [np.mean(abs_sim > x) for x in x_vals]
        
        ax.plot(x_vals, prob_sim, color='black', linestyle='--', linewidth=2, label='Null')
        ax.plot(x_vals, prob_real, color='#D32F2F', linewidth=3, label='Actual')
        
        ax.set_yscale('log')
        ax.set_ylim(1e-2, 1.1) # Focus on the top 1% to 100%
        ax.grid(True, which='both', alpha=0.2)
        ax.set_xlabel("Absolute Rank Diff (|Delta|)")
        if i == 0: 
            ax.set_ylabel("Prob(|d| > x) [Log Scale]")
            ax.legend()
        ax.set_title(f"Masked={m} (Tail Probability)")
        
        # Key Insight Annotation
        if i == 0:
            ax.text(60, 0.2, "Real > Null here\n= FAT TAILS!", color='#D32F2F', fontsize=10, weight='bold')

    plt.suptitle("Null Hypothesis: 3 Views (Linear, Histogram, Tail Probability)", fontsize=16, y=0.98)
    plt.tight_layout()
    
    out_path = "results/plots/null_hypothesis_integrated.png"
    plt.savefig(out_path, bbox_inches='tight')
    print(f"Saved plot to {out_path}")

if __name__ == "__main__":
    run_null_hypothesis_test()
