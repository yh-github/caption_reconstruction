import pandas as pd
import json
import os
import argparse
from pathlib import Path
from tqdm import tqdm

# Ensure src is in path for imports
import sys
sys.path.append('src')

# Import the specific model wrappers (assuming they exist in the repo)
from llm.local_llm import LocalLLM
from llm.prompting import DefaultPromptBuilder
from data.data_models import CaptionedVideo, Clip

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", type=str, default="llama-3.1-8b", help="ID of model to run")
    parser.add_argument("--widths", type=int, nargs="+", default=[4, 8])
    args = parser.parse_args()
    
    print("1. Loading Key Events dataset...")
    df = pd.read_csv("results/wild_key_events.csv")
    print(f"Loaded {len(df)} key events.")
    
    print(f"2. Initializing LocalLLM with ID: {args.model_id}...")
    llm = LocalLLM(args.model_id)
    prompter = DefaultPromptBuilder()
    
    out_dir = Path("results/key_events_reconstruction")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # We will loop through the dataset and reconstruct the climax gaps
    for w in args.widths:
        print(f"\n--- Running evaluation for gap width = {w}s ---")
        w_dir = out_dir / f"w={w}"
        w_dir.mkdir(exist_ok=True)
        
        for _, row in tqdm(df.iterrows(), total=len(df)):
            vid = row['video_id']
            climax_t = int(row['climax_start'])
            
            # Find GT file
            import glob
            gt_file = glob.glob(f"datasets/**/{vid}.json", recursive=True)
            if not gt_file:
                continue
                
            with open(gt_file[0]) as f:
                data = json.load(f)
                if isinstance(data, list):
                    caps = data
                elif isinstance(data, dict):
                    caps = data.get('data', data.get('captions', data.get('clips', [])))
                    
            if not caps or 'caption' not in caps[0]:
                continue
                
            # Create masked video object
            clips = []
            for i, c in enumerate(caps):
                is_masked = (climax_t <= i < climax_t + w)
                clips.append(Clip(index=i, caption=c['caption'], start=c['start'], end=c['end'], masked=is_masked))
                
            video = CaptionedVideo(video_id=vid, clips=clips)
            
            # Generate prompt
            context_dict = prompter.build_prompt(video)
            prompt_str = prompter.build_prompt_string(context_dict)
            
            # Call LLM
            out_file = w_dir / f"{vid}.json"
            if not out_file.exists():
                try:
                    response_text = llm.generate(prompt_str)
                    
                    # Try basic parsing
                    recon = {}
                    lines = response_text.split('\n')
                    for line in lines:
                        if ':' in line:
                            sec, text = line.split(':', 1)
                            sec = sec.replace('sec', '').replace('second', '').strip()
                            if sec.isdigit():
                                recon[int(sec)] = text.strip()
                                
                    with open(out_file, 'w') as f:
                        json.dump({
                            "video_id": vid,
                            "climax_start": climax_t,
                            "width": w,
                            "raw_response": response_text,
                            "reconstructed_captions": recon
                        }, f, indent=2)
                except Exception as e:
                    print(f"Failed on {vid}: {e}")

if __name__ == "__main__":
    main()
