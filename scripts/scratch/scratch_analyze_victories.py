import pandas as pd
import json
import glob
from pathlib import Path

def main():
    df = pd.read_csv("results/wild4_sweep_segment_level.csv")
    df['persistence_max'] = df[['T_RepeatClosest', 'T_MeanClosest']].max(axis=1)
    df['llm_win_margin'] = df['T_LLM'] - df['persistence_max']
    
    wins = df[df['llm_win_margin'] > 0.1].sort_values(by='llm_win_margin', ascending=False)
    
    md_content = "# Qualitative Analysis of SLM Reconstruction Victories\n\n"
    md_content += "This document analyzes instances where the SLM significantly outperforms persistence baselines.\n\n"
    
    count = 0
    for _, row in wins.iterrows():
        if count >= 15:
            break
        vid = row['video_id']
        w = int(row['width'])
        margin = row['llm_win_margin']
        cat = row['category']
        
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
        except Exception as e:
            continue
            
        base_cache = Path.home() / ".cache/huggingface/hub/datasets--Y3--dense_video_captions"
        pred_files = list(base_cache.glob(f"**/*{vid}*.json"))
        pred_text = "NOT FOUND"
        for pf in pred_files:
            if f"w={w}" in str(pf) and "i=29" in str(pf):
                try:
                    with open(pf) as f:
                        data = json.load(f)
                        recon = data.get("reconstructed_captions", {})
                        if recon:
                            pred_text = "\n".join([f"> *sec {k}: {v}*" for k, v in sorted(recon.items(), key=lambda x: int(x[0]))])
                            break
                except Exception as e:
                    pass
        
        if not gt_text or pred_text == "NOT FOUND":
            continue
            
        md_content += f"### {vid} (Gap: {w}s, Category: {cat})\n"
        md_content += f"**SLM Win Margin**: +{margin:.3f} (LLM: {row['T_LLM']:.2f}, Persistence: {row['persistence_max']:.2f})\n\n"
        
        idx = 29
        md_content += f"**Context Before (sec {idx-1}):** {gt_text[idx-1]}\n\n"
        md_content += f"**SLM Reconstruction:**\n{pred_text}\n\n"
        md_content += f"**Ground Truth (Masked):**\n"
        for i in range(idx, idx+w):
            if i < len(gt_text):
                md_content += f"> *sec {i}: {gt_text[i]}*\n"
        md_content += f"\n**Context After (sec {idx+w}):** {gt_text[idx+w] if idx+w < len(gt_text) else 'END'}\n\n"
        md_content += "---\n\n"
        count += 1

    # Save to artifacts directory
    artifact_path = "/home/yoavh/.gemini/antigravity/brain/fb95950a-187d-465f-ac10-b4b09aebc503/slm_qualitative_victories.md"
    with open(artifact_path, "w") as f:
        f.write(md_content)
        
    print(f"Generated slm_qualitative_victories.md with {count} examples.")

if __name__ == "__main__":
    main()
