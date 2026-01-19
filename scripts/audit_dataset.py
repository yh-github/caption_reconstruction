import os
import json
import glob
from pathlib import Path

DATASET_PATH = "datasets/wildQA/captions__wild2/"

def get_caption_sample(fpath, num_lines=3):
    try:
        with open(fpath, 'r') as f:
            data = json.load(f)
            
        captions = []
        # Case 1: List of caption objects
        if isinstance(data, list):
            if len(data) > 0 and 'caption' in data[0]:
                captions = [x['caption'] for x in data]
                
        # Case 2: Dict
        elif isinstance(data, dict):
            # Known keys
            possible_keys = ['captions', 'evidences', 'data', 'clips']
            for k in possible_keys:
                if k in data and isinstance(data[k], list):
                    if len(data[k]) > 0 and 'caption' in data[k][0]:
                       captions = [x['caption'] for x in data[k]]
                       break
            
            # Metadata style
            if not captions and 'question' in data:
                 captions = [f"Q: {data['question']}", f"A: {data['answer']}"]
                 
            # Fallback values scan
            if not captions:
                for v in data.values():
                    if isinstance(v, list) and len(v) > 0 and isinstance(v[0], dict) and 'caption' in v[0]:
                        captions = [x['caption'] for x in v]
                        break
        
        if captions:
            return captions[:num_lines]
        else:
            return ["<No captions found or unknown format>"]
            
    except Exception as e:
        return [f"<Error reading file: {e}>"]

def main():
    files = sorted(glob.glob(os.path.join(DATASET_PATH, "*.json")))
    
    print(f"Found {len(files)} files.\n")
    print("| Filename | Sample Captions |")
    print("| :--- | :--- |")
    
    for fpath in files:
        filename = os.path.basename(fpath)
        samples = get_caption_sample(fpath)
        
        # Format samples for markdown table
        # Join with <br> and escape pipes
        formatted_samples = "<br>".join([s.replace("|", "\|") for s in samples])
        
        print(f"| `{filename}` | {formatted_samples} |")

if __name__ == "__main__":
    main()
