"""Compute visual a-priori scores (dynamism, APCS_V) from SigLIP frame embeddings into results/apriori_dynamism_scores.csv."""
import os
import glob
import numpy as np
import pandas as pd
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import sys

# Ensure src can be imported
sys.path.append(str(Path(__file__).parent.parent))

from src.data.video_surprisal import VideoSurprisalScorer
from src.shared_target.pools import parse_video_id

EMB_DIR = Path("local/wild_videos_embs_siglip/")
OUTPUT_CSV = "results/apriori_dynamism_scores.csv"

def process_video(npy_path):
    try:
        vid_id = npy_path.stem
        embeddings = np.load(npy_path)
        
        scorer = VideoSurprisalScorer()
        result = scorer.calculate_surprisal(embeddings)
        
        channel, stem = parse_video_id(vid_id)
        
        return {
            "video_id": vid_id,
            "movie_id": channel,
            "stem": stem,
            "average_dynamism": result.avg_cosine_distance,
            "peak_dynamism": result.p95_cosine_distance,
            "combined_dynamism": result.combined_dynamism,
            "APCS_V": result.apcs_v,
            "video_length": len(embeddings)
        }
    except Exception as e:
        print(f"Error {npy_path}: {e}")
        return None

def main():
    print(f"Listing embedding files in {EMB_DIR}...")
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
    
    print(f"Computed scores for {len(df)} videos.")
    print(f"Found {df['movie_id'].nunique()} unique movie_ids (channels).")
    
    os.makedirs(Path(OUTPUT_CSV).parent, exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved to {OUTPUT_CSV}")
    print(df.describe())

if __name__ == "__main__":
    main()
