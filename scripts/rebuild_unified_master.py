#!/usr/bin/env python3
"""
rebuild_unified_master.py

Rebuilds results/unified_benchmark_master.csv combining:
1. All baseline methods (Visual_SigLIP_MeanClosest, Visual_SigLIP_RepeatClosest,
   Caption_MeanClosest, Caption_RepeatClosest) across all widths [1, 2, 3, 4, 6, 8, 12, 16]
   and indices [0, 29, 59] for both Wild4 (100 vids) and Wild5 (235 vids).
2. Llama-3.1-8B across all available widths [1, 2, 3, 4, 6, 8, 12, 16] at i=29 (and w=3, 6 at i=0, 29, 59).
"""

import json
import re
from pathlib import Path
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).parent.parent
RESULTS_DIR = REPO_ROOT / "results"

def load_categories():
    known = {}
    for ds in ["wild4", "wild5"]:
        cp = REPO_ROOT / f"datasets/wildQA/captions__{ds}/categories.json"
        if cp.exists():
            with open(cp) as f:
                d = json.load(f)
                for k, v in d.items():
                    c = v.get("category", "Unknown")
                    if c != "Unknown": known[k] = c
    return known

KNOWN_CATS = load_categories()

def get_category(vid: str) -> str:
    if vid in KNOWN_CATS and KNOWN_CATS[vid] != "Unknown":
        c = KNOWN_CATS[vid]
        if c == "Geography": return "Nature & Scenery"
        if c == "Human Survival": return "Survival"
        return c
    v = vid.lower()
    if any(k in v for k in ["military", "army", "defense", "air-force", "navy", "war", "soldier", "gung-ho", "sandboxx", "aiirsource"]):
        return "Military"
    elif any(k in v for k in ["farm", "agriculture", "tractor", "harvest", "suscovich", "crop", "cattle"]):
        return "Farming"
    elif any(k in v for k in ["survival", "primitive", "bushcraft", "robinet", "instinct", "zuber", "gaillard", "wilderness", "craft"]):
        return "Survival"
    elif any(k in v for k in ["disaster", "tornado", "storm", "hurricane", "tsunami", "flood", "earthquake", "climate", "skip-talbot", "dan-robinson", "weathershot"]):
        return "Natural Disaster"
    elif any(k in v for k in ["relaxation", "nature", "earth", "treadmill", "scenery", "landscape", "amazon", "safari", "explorer", "flying-dutchman", "virtual-running"]):
        return "Nature & Scenery"
    elif any(k in v for k in ["vehicle", "aviation", "airplane", "chase", "train", "flight", "car", "speed", "hinshaw"]):
        return "Action & Vehicle"
    return "Other"

def parse_masked_w_i(masked_str):
    try:
        inds = json.loads(masked_str)
        if not inds: return None, None
        w = len(inds)
        min_idx, max_idx = min(inds), max(inds)
        if min_idx <= 2:
            i_pos = 0
        elif max_idx >= 57:
            i_pos = 59
        else:
            i_pos = 29
        return w, i_pos
    except:
        return None, None

def main():
    records = []

    # 1. Baseline runs from results/for_analysis/
    baseline_configs = [
        ("Wild4", "results/for_analysis/wild4_siglip_sim_vec_vid.csv", "Visual_SigLIP", "Vector_Video"),
        ("Wild5", "results/for_analysis/wild5_siglip_sim_vec_vid.csv", "Visual_SigLIP", "Vector_Video"),
        ("Wild4", "results/for_analysis/wild4_sim_vec.csv", "Caption", "Vector_Caption"),
        ("Wild5", "results/for_analysis/wild5_sim_vec.csv", "Caption", "Vector_Caption"),
    ]

    for dataset_name, rel_path, prefix, m_type in baseline_configs:
        f_path = REPO_ROOT / rel_path
        if not f_path.exists():
            print(f"Skipping missing: {rel_path}")
            continue
        print(f"Loading {dataset_name} {prefix} from {rel_path}...")
        df = pd.read_csv(f_path)
        for _, row in df.iterrows():
            vid = str(row["video_id"]).replace("Olly_s-Farm", "Olly's-Farm")
            strat = str(row["recon_strategy"])
            strat_label = f"{prefix}_MeanClosest" if "Mean" in strat else f"{prefix}_RepeatClosest"
            w, i_pos = parse_masked_w_i(row["masked"])
            if w is None: continue
            records.append({
                "dataset": dataset_name,
                "video_id": vid,
                "category": get_category(vid),
                "method": strat_label,
                "method_family": m_type,
                "width": int(w),
                "index": int(i_pos),
                "mrr": float(row["mrr_mean"]) if pd.notnull(row["mrr_mean"]) else None,
                "recall_at_1": float(row["recall_at_1_mean"]) if pd.notnull(row["recall_at_1_mean"]) else None,
                "recall_at_5": float(row["recall_at_5_mean"]) if pd.notnull(row["recall_at_5_mean"]) else None,
                "cos_sim": float(row["cos_sim_mean"]) if pd.notnull(row["cos_sim_mean"]) else None,
                "cos_sim_min": float(row["cos_sim_min"]) if ("cos_sim_min" in row and pd.notnull(row["cos_sim_min"])) else None,
                "cos_sim_residual": float(row["cos_sim_residual_mean"]) if pd.notnull(row["cos_sim_residual_mean"]) else None,
                "mean_rank": float(row["mean_rank_mean"]) if pd.notnull(row["mean_rank_mean"]) else None,
            })

    # 2. Llama-3.1-8B runs from all directories
    llama_dirs = [
        (REPO_ROOT / "results/reconstruction/wild4_llama_w6", "Wild4"),
        # v3 was scored with the legacy pool_scope "window"; use the video-pool re-score
        # produced by scripts/rescore_window_pool_run.py instead.
        (REPO_ROOT / "results/recon/manual_download/reconstruction/wild4_llama_w3_window_v3_videopool", "Wild4"),
        (REPO_ROOT / "results/recon/manual_download/reconstruction/wild4_llama_w6", "Wild4"),
        (REPO_ROOT / "results/recon/manual_download/reconstruction/wild4_llama_multi_width", "Wild4"),
        (REPO_ROOT / "results/recon/manual_download/reconstruction/wild5_llama_w3_w6", "Wild5"),
        (REPO_ROOT / "results/recon/manual_download/reconstruction/wild5_llama_multi_width", "Wild5"),
    ]

    print("Loading Llama-3.1-8B evaluations...")
    for root_p, ds in llama_dirs:
        if not root_p.exists(): continue
        for s_dir in root_p.iterdir():
            if not s_dir.is_dir(): continue
            m = re.search(r"w=(\d+),\s*i=(\d+)", s_dir.name)
            if not m: continue
            w, i_pos = int(m.group(1)), int(m.group(2))
            for jf in s_dir.glob("*.json"):
                if jf.name.startswith("skip__") or jf.name.endswith("metadata.json"): continue
                vid = jf.stem.replace("Olly_s-Farm", "Olly's-Farm")
                try:
                    with open(jf) as fp: d = json.load(fp)
                    metrics = d.get("metrics", {})
                    cos_list = metrics.get("cos_sim", [])
                    records.append({
                        "dataset": ds,
                        "video_id": vid,
                        "category": get_category(vid),
                        "method": "Llama-3.1-8B",
                        "method_family": "SLM_Text",
                        "width": int(w),
                        "index": int(i_pos),
                        "mrr": float(metrics["mrr"]) if "mrr" in metrics and metrics["mrr"] is not None else None,
                        "recall_at_1": float(metrics["recall_at_1"]) if "recall_at_1" in metrics and metrics["recall_at_1"] is not None else None,
                        "recall_at_5": float(metrics["recall_at_5"]) if "recall_at_5" in metrics and metrics["recall_at_5"] is not None else None,
                        "cos_sim": float(np.mean(cos_list)) if cos_list else None,
                        "cos_sim_min": float(np.min(cos_list)) if cos_list else None,
                        "cos_sim_residual": float(np.mean(metrics["cos_sim_residual"])) if "cos_sim_residual" in metrics and metrics["cos_sim_residual"] else None,
                        "mean_rank": float(metrics["mean_rank"]) if "mean_rank" in metrics and metrics["mean_rank"] is not None else None,
                    })
                except Exception:
                    pass

    df_master = pd.DataFrame(records)
    print(f"Total compiled records: {len(df_master)}")
    # Deduplicate keeping last
    df_master = df_master.drop_duplicates(subset=["dataset", "video_id", "method", "width", "index"], keep="last")
    print(f"Deduplicated unique records: {len(df_master)}")

    master_csv = RESULTS_DIR / "unified_benchmark_master.csv"
    df_master.to_csv(master_csv, index=False)
    print(f"Saved: {master_csv}")

    # Summary check
    print("\nCounts per method and width at i=29:")
    sub29 = df_master[df_master["index"] == 29]
    print(pd.crosstab(sub29["method"], sub29["width"]).to_string())

if __name__ == "__main__":
    main()
