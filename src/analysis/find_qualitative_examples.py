import pandas as pd
import json
from pathlib import Path

def find_examples():
    llm_path = "results/for_analysis/wild_dev_sim_one_shot_t=1.csv"
    vid_path = "results/for_analysis/wild_dev_sim_vec_vid.csv"
    cats_path = "results/video_categories.json"
    
    # Load Data
    llm_df = pd.read_csv(llm_path)
    vid_df = pd.read_csv(vid_path)
    
    with open(cats_path, 'r') as f:
        cats = json.load(f)
    
    def get_cat(vid_name):
        return cats.get(vid_name, {}).get('category', 'Unknown')
        
    # Prepare combined data
    llm_df['method'] = 'LLM'
    vid_df['method'] = 'Video'
    combined = pd.concat([llm_df, vid_df], ignore_index=True)
    combined['num_masked'] = combined['masked'].apply(lambda x: len(eval(x)))
    
    # Filter for intersection
    counts = combined.groupby(['video_id', 'num_masked'])['method'].nunique()
    valid_groups = counts[counts == 2].index
    valid_df = pd.DataFrame(valid_groups.tolist(), columns=['video_id', 'num_masked'])
    combined = pd.merge(combined, valid_df, on=['video_id', 'num_masked'])
    
    # Calculate Ranks
    agg = combined.groupby(['method', 'video_id', 'num_masked'])['cos_sim_mean'].mean().reset_index()
    agg['rank'] = agg.groupby(['method', 'num_masked'])['cos_sim_mean'].rank(method='min', ascending=False)
    
    pivoted = agg.pivot_table(index=['video_id', 'num_masked'], columns='method', values='rank').reset_index()
    pivoted['delta'] = pivoted['LLM'] - pivoted['Video'] # Neg = LLM Better
    pivoted['category'] = pivoted['video_id'].map(get_cat)
    
    # Sort by Delta
    # Strongest LLM Wins (Negative Delta)
    llm_wins = pivoted.sort_values('delta', ascending=True)
    
    # Strongest Video Wins (Positive Delta)
    vid_wins = pivoted.sort_values('delta', ascending=False)
    
    output_str = "# Qualitative Examples Candidate List\n\n"
    
    mask_levels = [6, 9, 12, 15]
    
    for m in mask_levels:
        output_str += f"## Num Masked: {m}\n"
        subset = pivoted[pivoted['num_masked'] == m]
        
        output_str += "### Top 5 LLM Wins (Predictable/Logical?)\n"
        top_llm = subset.sort_values('delta', ascending=True).head(5)
        for _, row in top_llm.iterrows():
            output_str += f"- **{row['video_id']}** ({row['category']}): LLM Rank {row['LLM']:.0f} vs Vid Rank {row['Video']:.0f} (Delta: {row['delta']:.0f})\n"
            
        output_str += "\n### Top 5 Video Wins (Stochastic/Visual?)\n"
        top_vid = subset.sort_values('delta', ascending=False).head(5)
        for _, row in top_vid.iterrows():
            output_str += f"- **{row['video_id']}** ({row['category']}): LLM Rank {row['LLM']:.0f} vs Vid Rank {row['Video']:.0f} (Delta: {row['delta']:.0f})\n"
        
        output_str += "\n" + "-"*30 + "\n"

    print(output_str)
    
    with open("results/qualitative_candidates.md", "w") as f:
        f.write(output_str)
    print("Saved candidates to results/qualitative_candidates.md")

if __name__ == "__main__":
    find_examples()
