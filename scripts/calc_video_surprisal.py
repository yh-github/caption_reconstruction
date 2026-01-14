
import os
import glob
import numpy as np
import pandas as pd
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.data.video_surprisal import VideoSurprisalScorer, VideoSurprisalResult

EMB_DIR = Path("local/wild_videos_embs/")
OUTPUT_CSV = "results/video_surprisal_scores.csv"

def process_video(npy_path):
    try:
        vid_id = npy_path.stem
        embeddings = np.load(npy_path)
        
        scorer = VideoSurprisalScorer()
        result = scorer.calculate_surprisal(embeddings)
        
        return {
            "video_id": vid_id,
            "video_avg_dist": result.avg_cosine_distance,
            "video_max_dist": result.max_cosine_distance,
            "video_var_dist": result.variance_cosine_distance,
            "video_length": len(embeddings)
        }
    except Exception as e:
        print(f"Error {npy_path}: {e}")
        return None

def main():
    print("Listing embedding files...")
    files = list(EMB_DIR.glob("*.npy"))
    print(f"Found {len(files)} files.")
    
    results = []
    with ProcessPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(process_video, f) for f in files]
        for f in as_completed(futures):
            res = f.result()
            if res:
                results.append(res)
                
    df = pd.DataFrame(results)
    
    # Add Categories if possible
    try:
        import json
        with open("results/video_categories.json", 'r') as f:
            cats = json.load(f)
        df['category'] = df['video_id'].apply(lambda x: cats.get(x, {}).get("category", "Unknown") if isinstance(cats.get(x), dict) else "Unknown")
    except:
        pass
        
    print(f"Computed scores for {len(df)} videos.")
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved to {OUTPUT_CSV}")
    print(df.describe())

if __name__ == "__main__":
    main()
