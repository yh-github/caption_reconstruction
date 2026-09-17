#!/usr/bin/env python3
"""
Audits and analyzes Llama-3.1-8B Whole-Window reconstruction results on Wild4 for W=6.
Computes retrieval metrics (MRR, Recall@1, Recall@5), semantic similarity (Cosine Sim, Residual Cosine Sim),
positional breakdowns (prefix, middle, suffix), and side-by-side ground truth comparisons.
"""

import os
import sys
import json
import glob
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
WILD4_GT_DIR = PROJECT_ROOT / "datasets" / "wildQA" / "captions__wild4"
W6_RESULTS_DIR = PROJECT_ROOT / "results" / "reconstruction" / "wild4_llama_w6"

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

def main():
    gt_map = load_gt_captions()
    print(f"Loaded ground truth captions for {len(gt_map)} videos.")

    search_pattern = str(W6_RESULTS_DIR / "**" / "*.json")
    all_files = sorted(glob.glob(search_pattern, recursive=True))
    result_files = [f for f in all_files if not os.path.basename(f).startswith("skip__") and not f.endswith("metadata.json")]
    skip_files = [f for f in all_files if os.path.basename(f).startswith("skip__")]

    print(f"Found {len(result_files)} successful reconstruction JSON files across W=6 (0 skip files).")

    records = []
    by_pos = defaultdict(list)

    for f_path in result_files:
        try:
            with open(f_path, "r", encoding="utf-8") as fp:
                d = json.load(fp)

            video_id = d.get("video_id")
            recon_caps = d.get("reconstructed_captions", {})
            metrics = d.get("metrics", {})

            parent_name = Path(f_path).parent.name
            start_ind = int(parent_name.split("i=")[1].split(")")[0])

            mrr = float(metrics.get("mrr", 0.0))
            r1 = float(metrics.get("recall_at_1", 0.0))
            r5 = float(metrics.get("recall_at_5", 0.0))
            cos_sim_list = metrics.get("cos_sim", [])
            mean_cos = float(np.mean(cos_sim_list)) if cos_sim_list else 0.0
            res_cos_list = metrics.get("cos_sim_residual", [])
            mean_res = float(np.mean(res_cos_list)) if res_cos_list else 0.0

            caps_list = [c for _, c in sorted(recon_caps.items(), key=lambda x: int(x[0]))] if recon_caps else []
            word_counts = [len(c.split()) for c in caps_list]
            complete_sentences = [c.strip().endswith((".", "!", "?")) for c in caps_list]

            rec = {
                "file": f_path,
                "video_id": video_id,
                "start_ind": start_ind,
                "parent": parent_name,
                "mrr": mrr,
                "recall_at_1": r1,
                "recall_at_5": r5,
                "mean_cos_sim": mean_cos,
                "mean_cos_sim_residual": mean_res,
                "avg_word_count": np.mean(word_counts) if word_counts else 0.0,
                "complete_rate": np.mean(complete_sentences) if complete_sentences else 0.0,
                "recon_caps": recon_caps
            }
            records.append(rec)
            by_pos[start_ind].append(rec)
        except Exception as e:
            print(f"Error reading {f_path}: {e}")

    df = pd.DataFrame(records)
    csv_out = PROJECT_ROOT / "results" / "wild4_llama_w6_analysis.csv"
    df.drop(columns=["recon_caps"]).to_csv(csv_out, index=False)
    print(f"Saved full per-video analysis to {csv_out}")

    print("\n" + "="*85)
    print("=== LLAMA-3.1-8B WHOLE-WINDOW EVALUATION: WILD4 (W=6) ===")
    print("="*85)

    all_mrrs = df["mrr"].values
    all_r1s = df["recall_at_1"].values
    all_r5s = df["recall_at_5"].values
    all_cos = df["mean_cos_sim"].values
    all_res = df["mean_cos_sim_residual"].values
    all_words = df["avg_word_count"].values
    all_completes = df["complete_rate"].values

    print(f"Overall Metrics (N={len(df)} evaluations across 100 unique videos):")
    print(f"  • Mean MRR:                     {np.mean(all_mrrs):.4f} (median: {np.median(all_mrrs):.4f}, std: {np.std(all_mrrs):.4f})")
    print(f"  • Mean Recall@1:                {np.mean(all_r1s)*100:.2f}%")
    print(f"  • Mean Recall@5:                {np.mean(all_r5s)*100:.2f}%")
    print(f"  • Mean Cosine Sim:              {np.mean(all_cos):.4f} (median: {np.median(all_cos):.4f})")
    print(f"  • Mean Residual Cosine Sim:     {np.mean(all_res):.4f}")
    print(f"  • Mean Words per Caption:       {np.mean(all_words):.1f}")
    print(f"  • Punctuation Completeness:     {np.mean(all_completes)*100:.1f}%")

    print("\nBreakdown by Window Position:")
    pos_labels = {
        0: "Prefix (t=0..5)",
        29: "Middle (t=27..32)",
        59: "Suffix (t=54..59)"
    }
    for pos, p_recs in sorted(by_pos.items()):
        p_df = pd.DataFrame(p_recs)
        label = pos_labels.get(pos, f"Position i={pos}")
        print(f"  • {label:18s} [N={len(p_df)}]: MRR={p_df['mrr'].mean():.4f} | R@1={p_df['recall_at_1'].mean()*100:4.1f}% | R@5={p_df['recall_at_5'].mean()*100:4.1f}% | CosSim={p_df['mean_cos_sim'].mean():.4f} | ResSim={p_df['mean_cos_sim_residual'].mean():+.4f}")

    def print_instance(rec, rank_title):
        vid = rec["video_id"]
        gt_caps = gt_map.get(vid, [])
        recon_caps = rec["recon_caps"]
        recon_indices = sorted([int(k) for k in recon_caps.keys()])
        start = recon_indices[0] if recon_indices else rec["start_ind"]
        end = recon_indices[-1] if recon_indices else start + 5

        print(f"\n[{rank_title}] Video: {vid} (MRR: {rec['mrr']:.4f}, R@1: {rec['recall_at_1']:.2f}, Window: [{start}..{end}])")
        
        # Context Before
        if start > 0 and gt_caps:
            ctx_start = max(0, start - 2)
            for c_idx in range(ctx_start, start):
                print(f"  [Ctx t={c_idx:02d}] {gt_caps[c_idx] if c_idx < len(gt_caps) else ''}")

        # Missing Window
        print("  " + "-"*75)
        for idx in recon_indices:
            gt_text = gt_caps[idx] if idx < len(gt_caps) else "(None)"
            recon_text = recon_caps.get(str(idx), recon_caps.get(idx, "(missing)"))
            print(f"  [Recon t={idx:02d}] GT:    {gt_text}")
            print(f"                  Recon: {recon_text}")
        print("  " + "-"*75)

        # Context After
        if gt_caps and (end + 1) < len(gt_caps):
            ctx_end = min(len(gt_caps), end + 3)
            for c_idx in range(end + 1, ctx_end):
                print(f"  [Ctx t={c_idx:02d}] {gt_caps[c_idx]}")

    sorted_records = sorted(records, key=lambda x: x["mrr"], reverse=True)

    print("\n" + "="*85)
    print("=== TOP 3 HIGHEST-SCORING RECONSTRUCTIONS (BEST RETRIEVAL) ===")
    print("="*85)
    for i, r in enumerate(sorted_records[:3], 1):
        print_instance(r, f"TOP #{i}")

    print("\n" + "="*85)
    print("=== 3 MEDIAN-SCORING RECONSTRUCTIONS ===")
    print("="*85)
    med_start = max(0, len(sorted_records)//2 - 1)
    for i, r in enumerate(sorted_records[med_start:med_start+3], med_start+1):
        print_instance(r, f"MEDIAN #{i}")

    print("\n" + "="*85)
    print("=== 3 LOWEST-SCORING RECONSTRUCTIONS ===")
    print("="*85)
    for i, r in enumerate(sorted_records[-3:], len(sorted_records)-2):
        print_instance(r, f"LOWEST #{i}")

if __name__ == "__main__":
    main()
