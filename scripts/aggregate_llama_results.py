import json
from pathlib import Path
import pandas as pd
import numpy as np

def parse_llama_json(filepath):
    try:
        with open(filepath) as f:
            data = json.load(f)
    except:
        return None
        
    vid = data.get("video_id")
    if not vid:
        return None
        
    m = data.get("metrics", {})
    if not m:
        return None
        
    path_str = str(filepath)
    import re
    m_w = re.search(r'w=(\d+)', path_str)
    w_match = int(m_w.group(1)) if m_w else None
    
    m_i = re.search(r'i=(\d+)', path_str)
    i_match = int(m_i.group(1)) if m_i else None
    
    parts = filepath.parts
    try:
        llama_idx = next(i for i, p in enumerate(parts) if p.startswith('llama-3.1-8b'))
        exp_category = parts[llama_idx - 1]
    except:
        exp_category = "unknown"
        
    cos_sims = m.get("cos_sim", [])
    cos_res = m.get("cos_sim_residual", [])
    
    dataset = "wild4" if "wild4" in exp_category else "wild5"
    
    return {
        "video_id": vid,
        "dataset": dataset,
        "exp_category": exp_category,
        "w": w_match,
        "i": i_match,
        "mrr": m.get("mrr"),
        "mean_rank": m.get("mean_rank"),
        "cos_sim_mean": float(np.mean(cos_sims)) if len(cos_sims) else None,
    }

def main():
    dirs_to_search = [
        Path("results/reconstruction/wild4_llama_w6"),
        Path("results/recon/manual_download/reconstruction"),
    ]
    
    rows = []
    for d in dirs_to_search:
        if not d.exists():
            continue
        for jf in d.rglob("*.json"):
            # Only parse llama files
            if "llama-3.1-8b" in str(jf):
                res = parse_llama_json(jf)
                if res:
                    rows.append(res)
                
    df = pd.DataFrame(rows)
    # Deduplicate in case directories overlap
    df = df.drop_duplicates(subset=["video_id", "exp_category", "w", "i"])
    
    out_file = "results/llama_all_experiments_aggregated.csv"
    df.to_csv(out_file, index=False)
    
    # Print summary
    if len(df) > 0:
        summary = df.groupby(['dataset', 'exp_category', 'w', 'i'])[['mrr', 'cos_sim_mean']].mean().reset_index()
        print(summary.to_string())

if __name__ == "__main__":
    main()
