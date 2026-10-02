import argparse
import pandas as pd
import json
import os
import sys
from pathlib import Path

# Add src to pythonpath
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from llm.local_llm import LocalLLM
from llm.prompting import DefaultPromptBuilder
from data.data_models import CaptionedVideo, Clip
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="llama-3.1-8b")
    parser.add_argument("--widths", type=int, nargs="+", default=[4, 8])
    args = parser.parse_args()
    
    print(f"Loading Key Events Dataset...")
    df = pd.read_csv("results/wild_key_events.csv")
    
    print(f"Initializing model {args.model}...")
    llm = LocalLLM(args.model)
    prompter = DefaultPromptBuilder()
    
    out_dir = Path(f"results/key_events_{args.model.replace('/', '_')}")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    for w in args.widths:
        w_dir = out_dir / f"w={w}"
        w_dir.mkdir(exist_ok=True)
        print(f"\n--- Generating for Gap Width {w}s ---")
        
        for _, row in tqdm(df.iterrows(), total=len(df)):
            vid = row['video_id']
            climax_t = int(row['climax_start'])
            out_file = w_dir / f"{vid}.json"
            
            if out_file.exists():
                continue
                
            import glob
            gt_file = glob.glob(f"datasets/**/{vid}.json", recursive=True)
            if not gt_file:
                continue
            
            with open(gt_file[0]) as f:
                data = json.load(f)
                if isinstance(data, list): caps = data
                else: caps = data.get('data', data.get('captions', data.get('clips', [])))
                
            clips = []
            for i, c in enumerate(caps):
                is_masked = (climax_t <= i < climax_t + w)
                clips.append(Clip(index=i, caption=c['caption'], start=c['start'], end=c['end'], masked=is_masked))
                
            video = CaptionedVideo(video_id=vid, clips=clips)
            ctx = prompter.build_prompt(video)
            prompt_str = prompter.build_prompt_string(ctx)
            
            try:
                response = llm.generate(prompt_str)
                # Quick parse
                recon = {}
                for line in response.split('\n'):
                    if ':' in line:
                        sec, txt = line.split(':', 1)
                        sec = sec.replace('sec', '').replace('second', '').replace('*', '').strip()
                        if sec.isdigit():
                            recon[int(sec)] = txt.strip()
                            
                with open(out_file, 'w') as f:
                    json.dump({
                        "video_id": vid,
                        "climax_start": climax_t,
                        "width": w,
                        "raw_response": response,
                        "reconstructed_captions": recon
                    }, f, indent=2)
            except Exception as e:
                print(f"Error on {vid}: {e}")

if __name__ == "__main__":
    main()
