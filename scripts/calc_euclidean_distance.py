
import os
import sys
import json
import glob
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import euclidean

# Add source to path
sys.path.append(str(Path(__file__).parent.parent))

# Project Imports
MASK_FILL_DIR = Path("results/recon") # Where JSONs are
EMB_DIR = Path("local/wild_videos_embs/")
OUTPUT_CSV = "results/euclidean_metrics.csv"
MODEL_NAME = "all-MiniLM-L6-v2" # 384d to match DINOv2 (vit_small) dims, though spaces are not aligned!

def load_gt_embedding(video_id, timestamp):
    # This is tricky because we need the EXACT frame index or similar.
    # The JSON result contains 'timestamp'. 
    # The .npy file contains embeddings for frames.
    # We need to map timestamp -> frame index.
    # Assuming standard mapping: frame = timestamp * fps?
    # Or we can just trust that 'index' in the JSON refers to the index in the npy array?
    # Let's check `calc_baseline_full.py` logic or how `recalculate_metrics.py` works.
    pass

def main():
    print(f"Loading embedding model: {MODEL_NAME}...")
    model = SentenceTransformer(MODEL_NAME)
    
    # 1. Gather all result files
    # pattern matching existing logic
    files = glob.glob(str(MASK_FILL_DIR / "**" / "*.json"), recursive=True)
    # Filter for phi-3 t=0.1 if possible to save time, or do all?
    # Let's stick to the main analysis set: t=0.1
    files = [f for f in files if "phi-3__t=0.1" in f]
    print(f"Found {len(files)} result files.")
    
    rows = []
    
    for fpath in tqdm(files):
        try:
            with open(fpath, 'r') as f:
                data = json.load(f)
                
            vid_id = data['video_id']
            
            # Load GT Embeddings
            npy_path = EMB_DIR / f"{vid_id}.npy"
            if not npy_path.exists():
                continue
            gt_embs = np.load(npy_path)
            
            # Parse filename for config (width, index)
            # results/eval_phi-3/.../mask_fill_w={w}_i={i}_...
            parts = Path(fpath).name.split('__')
            # This parsing depends on filename format.
            # safe way: use the data inside if available?
            # data has 'config' usually.
            
            # Let's iterate over reconstructed captions
            recon = data.get('reconstructed_captions', {})
            
            # The indices in 'recon' should valid indices into gt_embs
            # BUT: recon keys are string indices.
            
            # Extract texts to batch embed
            indices = []
            texts = []
            for k, v in recon.items():
                indices.append(int(k))
                texts.append(v)
            
            if not texts:
                continue
                
            # Batch Embed
            pred_embs = model.encode(texts, show_progress_bar=False)
            
            # Compare
            for idx, pred_emb, text in zip(indices, pred_embs, texts):
                if idx >= len(gt_embs): 
                    continue
                    
                gt_emb = gt_embs[idx]
                
                # Metrics
                euc_dist = float(euclidean(pred_emb, gt_emb))
                # Cosine Distance (verification)
                cos_dist = float(1.0 - np.dot(pred_emb, gt_emb) / (np.linalg.norm(pred_emb)*np.linalg.norm(gt_emb)))
                
                # Parse additional info from filename if needed, or just assume aggregates later
                # We need width/index to group results.
                # Filename format: result_{vid}_{model}_{mask_conf}.json
                # Mask config: mask_fill_w=3_i=0
                
                # Extract width from path
                # Example: .../mask_fill_w=3_i=0_s=X/...
                # Let's simple regex for width
                import re
                width_match = re.search(r"w=(\d+)", fpath)
                width = int(width_match.group(1)) if width_match else -1
                
                index_match = re.search(r"i=(\d+)", fpath)
                start_index = int(index_match.group(1)) if index_match else -1
                
                rows.append({
                    "video_id": vid_id,
                    "frame_idx": idx,
                    "width": width,
                    "start_index": start_index,
                    "euclidean_dist": euc_dist,
                    "cosine_dist_check": cos_dist
                })
                
        except Exception as e:
            print(f"Error processing {fpath}: {e}")
            continue

    df = pd.DataFrame(rows)
    print(f"Computed {len(df)} records.")
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved to {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
