import pandas as pd
import json
from functools import reduce

def check_consistency():
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
        
    llm_df['method'] = 'LLM'
    vid_df['method'] = 'Video'
    combined = pd.concat([llm_df, vid_df], ignore_index=True)
    combined['num_masked'] = combined['masked'].apply(lambda x: len(eval(x)))
    
    # Filter for intersection
    counts = combined.groupby(['video_id', 'num_masked'])['method'].nunique()
    combined = pd.merge(combined, counts[counts == 2].index.to_frame(index=False), on=['video_id', 'num_masked'])
    
    mask_levels = [6, 9, 12, 15]
    
    llm_winners_sets = []
    vid_winners_sets = []
    
    top_n = 20 # Look at top 20 videos (approx top 20%) in each direction
    
    print(f"Checking for consistency across mask levels: {mask_levels}")
    print(f"Defining 'Consistent' as appearing in the Top {top_n} diffs across ALL levels.\n")
    
    for m in mask_levels:
        subset = combined[combined['num_masked'] == m]
        agg = subset.groupby(['method', 'video_id'])['cos_sim_mean'].mean().reset_index()
        agg['rank'] = agg.groupby('method')['cos_sim_mean'].rank(method='min', ascending=False)
        pivoted = agg.pivot(index='video_id', columns='method', values='rank')
        pivoted['delta'] = pivoted['LLM'] - pivoted['Video']
        
        # LLM Winners (Lowest Delta, e.g. -90)
        llm_top = set(pivoted.sort_values('delta', ascending=True).head(top_n).index)
        llm_winners_sets.append(llm_top)
        
        # Video Winners (Highest Delta, e.g. +90)
        vid_top = set(pivoted.sort_values('delta', ascending=False).head(top_n).index)
        vid_winners_sets.append(vid_top)
        
    # Find Intersection
    consistent_llm = reduce(lambda a, b: a.intersection(b), llm_winners_sets)
    consistent_vid = reduce(lambda a, b: a.intersection(b), vid_winners_sets)
    
    output_str = "# Consistent 'Outliers' Across All Mask Levels\n\n"
    
    output_str += f"### Consistently Pro-LLM (Top {top_n} in all masks)\n"
    if consistent_llm:
        for vid in consistent_llm:
            output_str += f"- **{vid}** ({get_cat(vid)})\n"
    else:
        output_str += "(None found)\n"
        
    output_str += f"\n### Consistently Pro-Video (Top {top_n} in all masks)\n"
    if consistent_vid:
        for vid in consistent_vid:
            output_str += f"- **{vid}** ({get_cat(vid)})\n"
    else:
        output_str += "(None found)\n"

    print(output_str)        
    
    # Save for reference
    with open("results/consistent_outliers.md", "w") as f:
        f.write(output_str)

if __name__ == "__main__":
    check_consistency()
