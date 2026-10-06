"""Merge visual, textual and NLL a-priori scores into results/apriori_full_scores.csv."""
import pandas as pd
import json
from pathlib import Path

def main():
    # 1. Video Dynamism
    df_vid = pd.read_csv('results/apriori_dynamism_scores.csv')
    df_vid = df_vid[['video_id', 'movie_id', 'average_dynamism', 'peak_dynamism', 'combined_dynamism', 'APCS_V']]
    df_vid['video_id'] = df_vid['video_id'].str.replace("Olly_s-Farm", "Olly's-Farm")
    
    # 2. Text Dynamism
    df_text = pd.read_csv('results/apriori_textual_dynamism.csv')
    
    # 3. Linguistic Surprisal
    nll_rows = []
    for jf in Path('results').rglob('prior_scoring*.json'):
        with open(jf) as f:
            data = json.load(f)
        for vid, m in data.items():
            if isinstance(m, dict):
                nll_rows.append({
                    'video_id': vid,
                    'apcs_nll': m.get('apcs_surprisal_nll'),
                    'caption_perplexity': m.get('caption_perplexity')
                })
    df_nll = pd.DataFrame(nll_rows).drop_duplicates('video_id')
    if len(df_nll) > 0:
        df_nll['video_id'] = df_nll['video_id'].str.replace("Olly_s-Farm", "Olly's-Farm")
    
    # Merge all
    df_merged = pd.merge(df_vid, df_text, on='video_id', how='inner')
    if len(df_nll) > 0:
        df_merged = pd.merge(df_merged, df_nll, on='video_id', how='left')
        
    df_merged.to_csv('results/apriori_full_scores.csv', index=False)
    print(f"Saved merged a-priori scores for {len(df_merged)} videos.")

if __name__ == "__main__":
    main()
