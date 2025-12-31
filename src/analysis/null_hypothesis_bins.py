import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def run_binned_null_test():
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
    
    # Define Thresholds
    # "Agreement": |Delta| <= threshold
    # "LLM Win": Delta < -threshold
    # "Video Win": Delta > threshold
    threshold = 20 # Ranks are 1-100. 20 is a significant difference (top quintile vs middle)
    
    results = []
    
    for m in mask_levels:
        subset = combined[combined['num_masked'] == m].copy()
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
        n_sims = 2000
        sim_deltas = []
        for _ in range(n_sims):
            r1 = np.random.permutation(n_videos) + 1
            r2 = np.random.permutation(n_videos) + 1
            sim_deltas.extend(r1 - r2)
        sim_deltas = np.array(sim_deltas)
        
        # Calculate Proportions
        def get_props(deltas):
            n = len(deltas)
            llm_win = np.sum(deltas < -threshold) / n
            agree = np.sum(np.abs(deltas) <= threshold) / n
            vid_win = np.sum(deltas > threshold) / n
            return llm_win, agree, vid_win
            
        real_props = get_props(real_delta)
        null_props = get_props(sim_deltas)
        
        results.append({
            'Mask': m,
            'Type': 'Real Data',
            'LLM Win (<-20)': real_props[0],
            'Agreement (+/-20)': real_props[1],
            'Video Win (>20)': real_props[2]
        })
        results.append({
            'Mask': m,
            'Type': 'Random Chance',
            'LLM Win (<-20)': null_props[0],
            'Agreement (+/-20)': null_props[1],
            'Video Win (>20)': null_props[2]
        })

    # Convert to DataFrame for Plotting
    res_df = pd.DataFrame(results)
    # Melt for Seaborn
    melted = res_df.melt(id_vars=['Mask', 'Type'], var_name='Category', value_name='Proportion')
    
    print(res_df)
    
    # Remove single barplot call
    # plt.figure(figsize=(12, 6))
    # sns.barplot(data=melted, x='Mask', y='Proportion', hue='Type', col='Category', errorbar=None)
    
    # Use Catplot for Faceted Bar Chart
    g = sns.catplot(
        data=melted, kind="bar",
        x="Mask", y="Proportion", hue="Type", col="Category",
        palette={'Real Data': '#D32F2F', 'Random Chance': 'grey'},
        height=5, aspect=0.8, alpha=0.9
    )
    
    g.fig.subplots_adjust(top=0.85)
    g.fig.suptitle(f'Distribution of Rank Deltas: Real vs Random (Threshold={threshold})', fontsize=16)
    
    # Annotate bars
    for ax in g.axes.flat:
        for container in ax.containers:
            ax.bar_label(container, fmt='%.2f', padding=3, fontsize=9)
            
    out_path = "results/plots/null_hypothesis_bins.png"
    plt.savefig(out_path, bbox_inches='tight')
    print(f"Saved plot to {out_path}")

if __name__ == "__main__":
    run_binned_null_test()
