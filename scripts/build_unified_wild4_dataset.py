#!/usr/bin/env python
import json
import re
from pathlib import Path
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
V2_DIR = PROJECT_ROOT / "results" / "recon" / "wild4_sim_text_v2_direct" / "reconstruction" / "wild4_sim_text_v2"
VEC_CAP_CSV = PROJECT_ROOT / "results" / "recon" / "wild4_sim_vec" / "wild4_sim_vec.csv"
VEC_VID_CSV = PROJECT_ROOT / "results" / "recon" / "wild4_sim_vec_vid" / "wild4_sim_vec_vid.csv"
CATEGORIES_FILE = PROJECT_ROOT / "local" / "categories_wild2_vs_wild4.json"
OUTPUT_CSV = PROJECT_ROOT / "results" / "wild4_reconstruction_master.csv"

def load_categories():
    if not CATEGORIES_FILE.exists():
        return {}
    with open(CATEGORIES_FILE, 'r') as f:
        d = json.load(f)
    cats = {}
    for vid, val in d.items():
        if isinstance(val, dict):
            cats[vid] = val.get("ground_truth", "Unknown")
        else:
            cats[vid] = str(val)
    return cats

def parse_masked_indices(masked_str):
    # Parse e.g. "[0, 1, 2]" -> width=3, start_index=0
    # or "[28, 29, 30]" -> width=3, start_index=29 (approximate/center)
    try:
        indices = json.loads(masked_str)
        if not indices: return None, None
        w = len(indices)
        # Determine closest canonical start_ind from [0, 29, 59]
        # In fixed_fill, 0 starts at 0; 29 centers around 29; 59 ends around 59
        min_idx, max_idx = min(indices), max(indices)
        if min_idx == 0:
            i_pos = 0
        elif 59 in indices or max_idx >= 58:
            i_pos = 59
        else:
            i_pos = 29
        return w, i_pos
    except Exception:
        return None, None

def main():
    cats = load_categories()
    records = []

    # 1. Parse Phi-3 SLM Results
    print("Parsing Phi-3 SLM JSON results...")
    phi_dirs = [d for d in V2_DIR.iterdir() if d.is_dir()]
    folder_re = re.compile(r"(phi-3_v2__t=[\d\.]+_rp=[\d\.]+)__fixed_fill\(w=(\d+),\s*i=(\d+)\)")

    for s_dir in phi_dirs:
        m = folder_re.match(s_dir.name)
        if not m: continue
        strat_name, w_str, i_str = m.groups()
        w = int(w_str)
        i_pos = int(i_str)

        for jf in s_dir.glob("*.json"):
            vid = jf.stem
            try:
                with open(jf) as fp:
                    data = json.load(fp)
                metrics = data.get("metrics", {})
                cos_sim_list = metrics.get("cos_sim", [])
                mean_cos = float(np.mean(cos_sim_list)) if cos_sim_list else None
                mrr = metrics.get("mrr")
                r1 = metrics.get("recall_at_1")
                r5 = metrics.get("recall_at_5")
                mean_rank = metrics.get("mean_rank")

                records.append({
                    "video_id": vid,
                    "category": cats.get(vid, "Unknown"),
                    "method": strat_name,
                    "method_type": "SLM_Text",
                    "width": w,
                    "index": i_pos,
                    "cos_sim": mean_cos,
                    "mrr": mrr,
                    "recall_at_1": r1,
                    "recall_at_5": r5,
                    "mean_rank": mean_rank
                })
            except Exception as e:
                pass

    print(f"Collected {len(records)} Phi-3 records.")

    # 2. Parse Caption Vector Baselines (wild4_sim_vec.csv)
    if VEC_CAP_CSV.exists():
        print("Parsing Caption Vector Baselines (wild4_sim_vec.csv)...")
        df_vec_cap = pd.read_csv(VEC_CAP_CSV)
        for _, row in df_vec_cap.iterrows():
            vid = row["video_id"]
            strat = row["recon_strategy"]
            w, i_pos = parse_masked_indices(str(row["masked"]))
            if w is None: continue
            
            method_name = f"vec_cap_{'mean' if 'Mean' in strat else 'repeat'}"
            records.append({
                "video_id": vid,
                "category": cats.get(vid, "Unknown"),
                "method": method_name,
                "method_type": "Vector_Caption",
                "width": w,
                "index": i_pos,
                "cos_sim": row.get("cos_sim_mean"),
                "mrr": row.get("mrr_mean"),
                "recall_at_1": row.get("recall_at_1_mean"),
                "recall_at_5": row.get("recall_at_5_mean"),
                "mean_rank": row.get("mean_rank_mean")
            })

    # 3. Parse Video Vector Baselines (wild4_sim_vec_vid.csv)
    if VEC_VID_CSV.exists():
        print("Parsing Video Vector Baselines (wild4_sim_vec_vid.csv)...")
        df_vec_vid = pd.read_csv(VEC_VID_CSV)
        for _, row in df_vec_vid.iterrows():
            vid = row["video_id"]
            strat = row["recon_strategy"]
            w, i_pos = parse_masked_indices(str(row["masked"]))
            if w is None: continue
            
            method_name = f"vec_vid_{'mean' if 'Mean' in strat else 'repeat'}"
            records.append({
                "video_id": vid,
                "category": cats.get(vid, "Unknown"),
                "method": method_name,
                "method_type": "Vector_Video",
                "width": w,
                "index": i_pos,
                "cos_sim": row.get("cos_sim_mean"),
                "mrr": row.get("mrr_mean"),
                "recall_at_1": row.get("recall_at_1_mean"),
                "recall_at_5": row.get("recall_at_5_mean"),
                "mean_rank": row.get("mean_rank_mean")
            })

    master_df = pd.DataFrame(records)
    print(f"Total Master Records: {len(master_df)}")
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    master_df.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved master dataset to: {OUTPUT_CSV}")

    # Summary table
    print("\n--- Summary by Method & Width ---")
    summary = master_df.groupby(["method", "width"])[["mrr", "cos_sim", "recall_at_1", "recall_at_5"]].mean().reset_index()
    print(summary.to_string(index=False))

if __name__ == "__main__":
    main()
