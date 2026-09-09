#!/usr/bin/env python3
"""
Audits and analyzes Llama-3.1-8B Whole-Window reconstruction results on Wild4.
Pulls results from Hugging Face if not available locally, computes overall metrics,
and inspects the top, median, and bottom instances with side-by-side ground truth comparisons.
"""

import os
import sys
import json
import glob
from pathlib import Path
from collections import defaultdict
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
WILD4_GT_DIR = PROJECT_ROOT / "datasets" / "wildQA" / "captions__wild4"
LOCAL_RESULTS_DIR = PROJECT_ROOT / "results" / "recon" / "wild4_llama_w3_window"

def load_gt_captions():
    gt = {}
    for p in WILD4_GT_DIR.glob("*.json"):
        if p.name == "categories.json":
            continue
        try:
            with open(p, "r", encoding="utf-8") as f:
                d = json.load(f)
            gt[p.stem] = [c.get("caption", "") for c in d.get("captions", [])]
        except Exception:
            pass
    return gt

def fetch_hf_results():
    """Downloads results from HF if not present locally."""
    try:
        from huggingface_hub import HfApi, hf_hub_download
        token = os.environ.get("HF_TOKEN")
        api = HfApi(token=token)
        repo_id = "Y3/dense_video_captions"
        
        print("Checking Hugging Face for reconstruction/wild4_llama_w3_window...")
        tree = list(api.list_repo_tree(repo_id=repo_id, repo_type="dataset", path_in_repo="reconstruction/wild4_llama_w3_window", recursive=True))
        json_files = [item.path for item in tree if item.path.endswith(".json")]
        print(f"Found {len(json_files)} JSON files on Hugging Face.")
        
        for r_path in json_files:
            local_dest = PROJECT_ROOT / "results" / "recon" / Path(r_path)
            if not local_dest.exists():
                local_dest.parent.mkdir(parents=True, exist_ok=True)
                downloaded = hf_hub_download(repo_id=repo_id, repo_type="dataset", filename=r_path, token=token)
                import shutil
                shutil.copy(downloaded, local_dest)
        print("All remote files synchronized locally.")
    except Exception as e:
        print(f"Note: HF sync skipped/failed: {e}")

def main():
    fetch_hf_results()
    gt_map = load_gt_captions()
    print(f"Loaded ground truth for {len(gt_map)} videos.")

    # Search for result files locally
    search_pattern = str(PROJECT_ROOT / "results" / "recon" / "**" / "wild4_llama_w3_window" / "**" / "*.json")
    all_files = glob.glob(search_pattern, recursive=True)
    # Filter out metadata.yaml and skip__ files for primary analysis
    result_files = [f for f in all_files if not os.path.basename(f).startswith("skip__") and not f.endswith("metadata.json")]
    skip_files = [f for f in all_files if os.path.basename(f).startswith("skip__")]

    print(f"Found {len(result_files)} successful reconstruction files and {len(skip_files)} skip files.")

    if not result_files:
        print("No successful results found yet. Check if experiment is still running.")
        return

    records = []
    by_pos = defaultdict(list)

    for f_path in result_files:
        try:
            with open(f_path, "r", encoding="utf-8") as fp:
                d = json.load(fp)
            
            video_id = d.get("video_id")
            recon_caps = d.get("reconstructed_captions", {})
            metrics = d.get("metrics", {})
            
            # Subdir pattern: e.g. llama-3.1-8b__whole_window__t=0.6__fixed_fill(w=3, i=29)
            parent_name = Path(f_path).parent.name
            start_ind = None
            if "i=" in parent_name:
                try:
                    start_ind = int(parent_name.split("i=")[1].split(")")[0])
                except Exception:
                    pass

            mrr = metrics.get("mrr") or metrics.get("mean_reciprocal_rank") or 0.0
            r1 = metrics.get("recall@1") or metrics.get("r@1") or 0.0
            r5 = metrics.get("recall@5") or metrics.get("r@5") or 0.0

            # Compute length and linguistic stats
            caps_list = [c for _, c in sorted(recon_caps.items(), key=lambda x: int(x[0]))] if recon_caps else []
            word_counts = [len(c.split()) for c in caps_list]
            complete_sentences = [c.strip().endswith((".", "!", "?")) for c in caps_list]

            rec = {
                "file": f_path,
                "video_id": video_id,
                "start_ind": start_ind,
                "parent": parent_name,
                "mrr": float(mrr),
                "r1": float(r1),
                "r5": float(r5),
                "recon_caps": recon_caps,
                "caps_list": caps_list,
                "avg_word_count": np.mean(word_counts) if word_counts else 0.0,
                "complete_rate": np.mean(complete_sentences) if complete_sentences else 0.0
            }
            records.append(rec)
            if start_ind is not None:
                by_pos[start_ind].append(rec)
        except Exception as e:
            print(f"Error reading {f_path}: {e}")

    print("\n" + "="*80)
    print(f"=== LLAMA-3.1-8B WHOLE-WINDOW EVALUATION SUMMARY ({len(records)} INSTANCES) ===")
    print("="*80)

    all_mrrs = [r["mrr"] for r in records]
    all_r1s = [r["r1"] for r in records]
    all_r5s = [r["r5"] for r in records]
    all_lens = [r["avg_word_count"] for r in records]
    all_completes = [r["complete_rate"] for r in records]

    print(f"Overall Metrics (N={len(records)}):")
    print(f"  • Mean MRR:      {np.mean(all_mrrs):.4f} (median: {np.median(all_mrrs):.4f}, std: {np.std(all_mrrs):.4f})")
    print(f"  • Mean Recall@1: {np.mean(all_r1s):.4f}")
    print(f"  • Mean Recall@5: {np.mean(all_r5s):.4f}")
    print(f"  • Mean Word Count per Caption: {np.mean(all_lens):.1f} words")
    print(f"  • Sentence Punctuation Completeness: {np.mean(all_completes)*100:.1f}%")

    print("\nBreakdown by Window Position:")
    for pos, p_recs in sorted(by_pos.items()):
        p_mrrs = [r["mrr"] for r in p_recs]
        p_r1s = [r["r1"] for r in p_recs]
        p_r5s = [r["r5"] for r in p_recs]
        pos_label = "Prefix (t=0-2)" if pos == 0 else ("Middle (t=29-31)" if pos == 29 else f"Suffix (t={pos})")
        print(f"  • {pos_label} [N={len(p_recs)}]: MRR={np.mean(p_mrrs):.4f} | R@1={np.mean(p_r1s):.4f} | R@5={np.mean(p_r5s):.4f}")

    # Rank instances by MRR
    sorted_records = sorted(records, key=lambda x: x["mrr"], reverse=True)

    def print_instance(rec, rank_title):
        vid = rec["video_id"]
        gt_caps = gt_map.get(vid, [])
        start = rec["start_ind"] if rec["start_ind"] is not None else 0
        w = 3

        print(f"\n[{rank_title}] Video: {vid} (MRR: {rec['mrr']:.4f}, R@1: {rec['r1']:.1f}, Window: [{start}..{start+w-1}])")
        
        # Context Before
        if start > 0 and gt_caps:
            ctx_start = max(0, start - 2)
            for c_idx in range(ctx_start, start):
                print(f"  [Ctx t={c_idx:02d}] {gt_caps[c_idx] if c_idx < len(gt_caps) else ''}")

        # Missing Window: GT vs Recon
        print("  " + "-"*60)
        for offset in range(w):
            idx = start + offset
            gt_text = gt_caps[idx] if idx < len(gt_caps) else "(None)"
            recon_text = rec["recon_caps"].get(str(idx), rec["recon_caps"].get(idx, "(missing)"))
            print(f"  [Missing t={idx:02d}] GT:    {gt_text}")
            print(f"                 Recon: {recon_text}")
        print("  " + "-"*60)

        # Context After
        if gt_caps and (start + w) < len(gt_caps):
            ctx_end = min(len(gt_caps), start + w + 2)
            for c_idx in range(start + w, ctx_end):
                print(f"  [Ctx t={c_idx:02d}] {gt_caps[c_idx]}")

    print("\n" + "="*80)
    print("=== TOP 5 HIGHEST-SCORING RECONSTRUCTIONS (BEST RETRIEVAL) ===")
    print("="*80)
    for i, r in enumerate(sorted_records[:5], 1):
        print_instance(r, f"TOP #{i}")

    print("\n" + "="*80)
    print("=== 3 MEDIAN-SCORING RECONSTRUCTIONS ===")
    print("="*80)
    med_start = max(0, len(sorted_records)//2 - 1)
    for i, r in enumerate(sorted_records[med_start:med_start+3], med_start+1):
        print_instance(r, f"MEDIAN #{i}")

    print("\n" + "="*80)
    print("=== 3 LOWEST-SCORING RECONSTRUCTIONS ===")
    print("="*80)
    for i, r in enumerate(sorted_records[-3:], len(sorted_records)-2):
        print_instance(r, f"LOWEST #{i}")

if __name__ == "__main__":
    main()
