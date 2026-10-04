import pandas as pd
import json
import numpy as np
import sys
sys.path.append('src')
from llm.local_embedder import SiglipTextEmbedder
from scipy.spatial.distance import cosine

def main():
    print("Loading data...")
    df = pd.read_csv("results/wild4_sweep_segment_level.csv")
    df['persistence_max'] = df[['T_RepeatClosest', 'T_MeanClosest']].max(axis=1)
    df['llm_win_margin'] = df['T_LLM'] - df['persistence_max']
    
    # Analyze all widths or just a representative one
    df = df[df['width'] == 4].copy()
    
    embedder = SiglipTextEmbedder()
    
    deltas = []
    margins = []
    
    for _, row in df.iterrows():
        vid = row['video_id']
        w = int(row['width'])
        idx = 29
        
        gt_file = f"datasets/wildQA/captions__wild4/{vid}.json"
        gt_text = []
        try:
            with open(gt_file) as f:
                data = json.load(f)
                if isinstance(data, list):
                    caps = data
                elif isinstance(data, dict):
                    caps = data.get('data', data.get('captions', data.get('clips', [])))
                if caps and 'caption' in caps[0]:
                    gt_text = [x['caption'] for x in caps]
        except Exception:
            continue
            
        if len(gt_text) > idx + w:
            before = gt_text[idx-1]
            after = gt_text[idx+w]
            
            embs = embedder.get_embeddings(f"siglip_context_{vid}", [before, after])
            delta = cosine(embs[0], embs[1])
            
            deltas.append(delta)
            margins.append(row['llm_win_margin'])
            
    if len(deltas) > 0:
        r = np.corrcoef(deltas, margins)[0, 1]
        print(f"Correlation between Context Delta (Semantic Distance Before->After) and LLM Win Margin: r = {r:.3f}")
        
        median_delta = np.median(deltas)
        high_delta_wins = np.mean([m > 0 for d, m in zip(deltas, margins) if d > median_delta])
        low_delta_wins = np.mean([m > 0 for d, m in zip(deltas, margins) if d <= median_delta])
        
        print(f"LLM Win Rate when Context Delta is HIGH: {high_delta_wins*100:.1f}%")
        print(f"LLM Win Rate when Context Delta is LOW:  {low_delta_wins*100:.1f}%")

if __name__ == "__main__":
    main()
