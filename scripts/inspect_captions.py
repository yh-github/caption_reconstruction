
import json
import sys
import glob
import os
from pathlib import Path

# Add src to path
sys.path.append("src")
from data.video_link_loader import load_wild_dataset

VIDEOS = ["Army-military-2018_8-clip-73", "Weathershot_7-clip-0"]
BASE_DIR = "results/recon/wild_dev_sim_text"
TEMPS = [0.1, 1.5]
DATASET_PATH = "datasets/wildQA/captions__wild2/"

def get_captions(video_id, temp):
    # Find a file for this video and temp. Prefer w=12 (highest context)
    pattern = f"**/*t={temp}*rp=1.2*w=12*/*{video_id}.json"
    files = glob.glob(os.path.join(BASE_DIR, pattern), recursive=True)
    
    if not files:
        # Try any width
        pattern = f"**/*t={temp}*rp=1.2*/*{video_id}.json"
        files = glob.glob(os.path.join(BASE_DIR, pattern), recursive=True)
        
    if not files:
        return None, None
        
    filepath = files[0]
    with open(filepath, 'r') as f:
        data = json.load(f)
        
    recon = data.get('reconstructed_captions', {})
    params = data.get('parameters', {})
    return recon, params

def get_original_captions(video_id):
    path = os.path.join(DATASET_PATH, f"{video_id}.json")
    if os.path.exists(path):
        with open(path, 'r') as f:
            data = json.load(f)
            
        # Case 1: List of caption objects
        if isinstance(data, list):
            if len(data) > 0 and 'caption' in data[0]:
                return [x['caption'] for x in data]
                
        # Case 2: Dict
        if isinstance(data, dict):
            # Check for known keys that might hold the list
            possible_keys = ['captions', 'evidences', 'data', 'clips']
            for k in possible_keys:
                if k in data and isinstance(data[k], list):
                    if len(data[k]) > 0 and 'caption' in data[k][0]:
                       return [x['caption'] for x in data[k]]
            
            # If not in known keys, maybe 'question' / 'answer' format (WildQA metadata)
            if 'question' in data:
                 return [f"Q: {data['question']}\nA: {data['answer']}"]
                 
            # Fallback: Check all values to see if any is a list of captions
            for v in data.values():
                if isinstance(v, list) and len(v) > 0 and isinstance(v[0], dict) and 'caption' in v[0]:
                    return [x['caption'] for x in v]

    return None

def main():
    for vid in VIDEOS:
        print(f"\n{'='*50}")
        print(f"VIDEO: {vid}")
        print(f"{'='*50}")
        
        # 1. Get Original
        captions = get_original_captions(vid)
        if captions:
            print(f"\n--- Original Captions (First 5) ---")
            for i, cap in enumerate(captions[:5]):
                print(f"Index {i}: {cap}")
        else:
            print("\n[Original Captions Not Found]")
            
        # 2. Reconstructions
        for t in TEMPS:
            recon, params = get_captions(vid, t)
            if recon:
                print(f"\n[Temperature {t}]")
                sorted_keys = sorted([int(k) for k in recon.keys() if k.isdigit()])
                
                # Print same amount
                count = 0
                for k in sorted_keys:
                    if count >= 5: break
                    print(f"Mask {k}: {recon[str(k)]}")
                    count += 1
            else:
                print(f"[Temperature {t}] No file found.")

if __name__ == "__main__":
    main()
