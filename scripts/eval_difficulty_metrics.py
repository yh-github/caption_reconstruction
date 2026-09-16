#!/usr/bin/env python3
"""
scripts/eval_difficulty_metrics.py

Evaluates caption reconstruction using the Difficulty-Conditioned framework:
- Nearest Boundary baseline (ties broken by left/earlier)
- Visual Baseline (mean-closest boundary interpolation)
- Generative Model (LLM / SLM)
- Narrative Gap Difficulty D(t) = 1 - cos(boundary, GT)
- Model Advantage Delta(t) = cos(model, GT) - cos(boundary, GT)
- Robust aggregations: Quartiles (Q1-Q4), Difficulty-Weighted Advantage, Scaling Correlation
"""

import os
import sys
import json
import glob
import argparse
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd
from scipy import stats
import diskcache

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Difficulty-Conditioned Metrics")
    parser.add_argument("--results_dir", type=str, default=None,
                        help="Path to local directory with reconstruction JSONs (e.g. results/reconstruction/wild4_llama_w6)")
    parser.add_argument("--hf_pattern", type=str, default=None,
                        help="Pattern to match in HF dataset Y3/dense_video_captions (e.g. wild4_llama_w3_window_v3)")
    parser.add_argument("--gt_dir", type=str, default="datasets/wildQA/captions__wild4",
                        help="Directory containing ground truth JSONs")
    parser.add_argument("--cache_dir", type=str, default="disk_cache/local_all-mpnet-base-v2",
                        help="Path to embedding diskcache")
    parser.add_argument("--output_csv", type=str, default=None,
                        help="Path to save detailed per-clip results CSV")
    parser.add_argument("--summary_csv", type=str, default=None,
                        help="Path to save quartile summary CSV")
    return parser.parse_args()

def load_gt_captions(gt_dir: str) -> dict[str, list[str]]:
    gt_map = {}
    for p in glob.glob(os.path.join(gt_dir, "*.json")):
        if os.path.basename(p) == "categories.json":
            continue
        try:
            with open(p, "r", encoding="utf-8") as f:
                d = json.load(f)
            vid = d.get("video_id") or Path(p).stem
            clips = d.get("captions") or d.get("clips") or []
            gt_map[vid] = [c["caption"] for c in clips if "caption" in c]
        except Exception as e:
            print(f"Warning: Failed loading {p}: {e}")
    return gt_map

def find_nearest_boundary(idx: int, unmasked: list[int]) -> int:
    """Finds nearest unmasked boundary, strictly breaking ties to the left/earlier."""
    left = [b for b in unmasked if b < idx]
    right = [b for b in unmasked if b > idx]
    b_left = max(left) if left else None
    b_right = min(right) if right else None
    
    if b_left is not None and b_right is not None:
        dist_l = idx - b_left
        dist_r = b_right - idx
        return b_left if dist_l <= dist_r else b_right
    elif b_left is not None:
        return b_left
    elif b_right is not None:
        return b_right
    raise ValueError(f"No boundary available for idx {idx}")

def get_boundary_neighbors(idx: int, unmasked: list[int]) -> tuple[int | None, int | None]:
    left = [b for b in unmasked if b < idx]
    right = [b for b in unmasked if b > idx]
    return (max(left) if left else None, min(right) if right else None)

def main():
    args = parse_args()
    cache = diskcache.Cache(args.cache_dir)
    print(f"Loaded diskcache from {args.cache_dir} ({len(cache)} entries).")

    gt_dict = load_gt_captions(args.gt_dir)
    print(f"Loaded ground truth for {len(gt_dict)} videos from {args.gt_dir}.")

    # Gather JSON files
    json_files = []
    if args.results_dir:
        pattern = os.path.join(args.results_dir, "**", "*.json")
        for f in sorted(glob.glob(pattern, recursive=True)):
            b = os.path.basename(f)
            if not b.startswith("skip__") and not b.endswith("metadata.json"):
                json_files.append(f)
        print(f"Found {len(json_files)} local result JSON files in {args.results_dir}")
    elif args.hf_pattern:
        from huggingface_hub import HfApi, hf_hub_download
        api = HfApi()
        repo_files = api.list_repo_files(repo_id="Y3/dense_video_captions", repo_type="dataset")
        matching = [f for f in repo_files if args.hf_pattern in f and f.endswith(".json")]
        print(f"Found {len(matching)} matching files on HuggingFace ({args.hf_pattern}). Downloading...")
        for f in matching:
            lp = hf_hub_download(repo_id="Y3/dense_video_captions", filename=f, repo_type="dataset")
            json_files.append(lp)
    else:
        print("Error: Must specify either --results_dir or --hf_pattern")
        sys.exit(1)

    if not json_files:
        print("No result files found. Exiting.")
        sys.exit(1)

    from llm.local_embedder import LocalEmbedder
    embedder = LocalEmbedder(model_name="all-mpnet-base-v2")

    records = []

    for f_path in json_files:
        try:
            with open(f_path, "r", encoding="utf-8") as fp:
                d = json.load(fp)
        except Exception as e:
            continue

        vid = d.get("video_id")
        if not vid or vid not in gt_dict:
            continue

        recon_caps = d.get("reconstructed_captions", {})
        if not recon_caps:
            continue

        gt_caps = gt_dict[vid]
        n_clips = len(gt_caps)
        all_indices = list(range(n_clips))

        sorted_keys = sorted(recon_caps.keys(), key=lambda x: int(x))
        gap_indices = [int(k) for k in sorted_keys]
        gap_set = set(gap_indices)
        unmasked = sorted(list(set(all_indices) - gap_set))

        # Position label
        start_idx = gap_indices[0]
        if start_idx <= 2:
            pos = "start"
        elif start_idx >= n_clips - len(gap_indices) - 2:
            pos = "end"
        else:
            pos = "mid"

        # Embed missing GT or Predictions
        for idx in gap_indices:
            if idx >= n_clips:
                continue
            gt_text = gt_caps[idx]
            pred_text = recon_caps[str(idx)]
            if gt_text not in cache:
                embedder.get_embeddings(f"{vid}_gt", [gt_text])
            if pred_text not in cache:
                embedder.get_embeddings(f"{vid}_pred", [pred_text])

        for idx in gap_indices:
            if idx >= n_clips:
                continue
            gt_text = gt_caps[idx]
            pred_text = recon_caps[str(idx)]
            if gt_text not in cache or pred_text not in cache:
                continue

            gt_vec = cache[gt_text]
            pred_vec = cache[pred_text]

            # 1. Nearest boundary (ties broken left)
            b_idx = find_nearest_boundary(idx, unmasked)
            b_text = gt_caps[b_idx]
            if b_text not in cache:
                embedder.get_embeddings(f"{vid}_gt", [b_text])
            b_vec = cache[b_text]

            # 2. Visual / Interpolation baseline (mean-closest)
            b_left, b_right = get_boundary_neighbors(idx, unmasked)
            interp_vecs = []
            if b_left is not None:
                bl_text = gt_caps[b_left]
                if bl_text not in cache: embedder.get_embeddings(f"{vid}_gt", [bl_text])
                interp_vecs.append(cache[bl_text])
            if b_right is not None:
                br_text = gt_caps[b_right]
                if br_text not in cache: embedder.get_embeddings(f"{vid}_gt", [br_text])
                interp_vecs.append(cache[br_text])

            if interp_vecs:
                interp_vec = np.mean(interp_vecs, axis=0)
                interp_vec = interp_vec / (np.linalg.norm(interp_vec) + 1e-9)
                interp_cos = float(np.dot(interp_vec, gt_vec))
            else:
                interp_cos = float(np.dot(b_vec, gt_vec))

            # Cosine similarities
            nearest_cos = float(np.dot(b_vec, gt_vec))
            model_cos = float(np.dot(pred_vec, gt_vec))

            difficulty = 1.0 - nearest_cos
            adv_model = model_cos - nearest_cos
            adv_interp = interp_cos - nearest_cos
            rgb = adv_model / difficulty if difficulty > 0.01 else 0.0

            records.append({
                "video_id": vid,
                "position": pos,
                "clip_idx": idx,
                "gap_width": len(gap_indices),
                "boundary_idx": b_idx,
                "difficulty": difficulty,
                "nearest_cos": nearest_cos,
                "interp_cos": interp_cos,
                "model_cos": model_cos,
                "adv_model": adv_model,
                "adv_interp": adv_interp,
                "rgb": rgb,
                "win_vs_nearest": adv_model > 0,
                "win_vs_interp": model_cos > interp_cos
            })

    df = pd.DataFrame(records)
    print(f"\nSuccessfully evaluated {len(df)} clips across {df['video_id'].nunique()} videos (Gap Width W={df['gap_width'].iloc[0]}).")

    if args.output_csv:
        Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.output_csv, index=False)
        print(f"Saved detailed per-clip metrics to {args.output_csv}")

    # =========================================================
    # SUMMARY REPORT
    # =========================================================
    print("\n" + "="*85)
    print("=== OVERALL MEAN PERFORMANCE ===")
    print("="*85)
    print(f"Nearest Boundary CosSim:    {df['nearest_cos'].mean():.4f} (std={df['nearest_cos'].std():.3f})")
    print(f"Visual (Mean-Closest) CosSim:{df['interp_cos'].mean():.4f} (std={df['interp_cos'].std():.3f})")
    print(f"Model CosSim:               {df['model_cos'].mean():.4f} (std={df['model_cos'].std():.3f})")
    print(f"Mean Narrative Difficulty:  {df['difficulty'].mean():.4f}")
    print(f"Overall Win Rate vs Nearest:{df['win_vs_nearest'].mean()*100:.1f}%")
    print(f"Overall Win Rate vs Interp: {df['win_vs_interp'].mean()*100:.1f}%")

    t_nr, p_nr = stats.ttest_rel(df['model_cos'], df['nearest_cos'])
    d_nr = (df['model_cos'] - df['nearest_cos']).mean() / (df['adv_model'].std() + 1e-9)
    print(f"Paired t-test vs Nearest:   t={t_nr:.2f}, p={p_nr:.2e}, Cohen's d={d_nr:.3f}")

    # =========================================================
    # DIFFICULTY-WEIGHTED MEAN ADVANTAGE
    # =========================================================
    w = df['difficulty'].values
    w_nr = np.average(df['nearest_cos'], weights=w)
    w_int = np.average(df['interp_cos'], weights=w)
    w_mod = np.average(df['model_cos'], weights=w)

    print("\n" + "="*85)
    print("=== DIFFICULTY-WEIGHTED AGGREGATE (Singularity-Free) ===")
    print("="*85)
    print(f"Weighted Nearest Boundary:  {w_nr:.4f}")
    print(f"Weighted Visual Baseline:   {w_int:.4f} (Advantage = {w_int - w_nr:+.4f})")
    print(f"Weighted Model:             {w_mod:.4f} (Advantage = {w_mod - w_nr:+.4f})")

    # =========================================================
    # DIFFICULTY QUARTILE BREAKDOWN
    # =========================================================
    print("\n" + "="*85)
    print("=== PERFORMANCE STRATIFIED BY DIFFICULTY QUARTILE ===")
    print("="*85)
    df['quartile'] = pd.qcut(df['difficulty'], q=4, labels=['Q1 (Easy/Static)', 'Q2 (Mild)', 'Q3 (Moderate)', 'Q4 (Hard/Dynamic)'])

    summary_rows = []
    for q_name, grp in df.groupby('quartile', observed=True):
        n = len(grp)
        d_mean = grp['difficulty'].mean()
        nr_mean = grp['nearest_cos'].mean()
        int_mean = grp['interp_cos'].mean()
        mod_mean = grp['model_cos'].mean()
        wins_nr = grp['win_vs_nearest'].sum()
        wins_int = grp['win_vs_interp'].sum()
        t_q, p_q = stats.ttest_rel(grp['model_cos'], grp['nearest_cos'])
        d_q = grp['adv_model'].mean() / (grp['adv_model'].std() + 1e-9)

        print(f"[{q_name:18s}] N={n:3d} | Difficulty={d_mean:.3f}")
        print(f"   Nearest Boundary: {nr_mean:.4f}")
        print(f"   Visual Baseline:  {int_mean:.4f}")
        print(f"   Model:            {mod_mean:.4f}")
        print(f"   Win vs Nearest:   {wins_nr}/{n} ({wins_nr/n*100:4.1f}%) | Win vs Interp: {wins_int}/{n} ({wins_int/n*100:4.1f}%)")
        print(f"   Paired t-test:    t={t_q:+.2f}, p={p_q:.2e}, d={d_q:+.3f}\n")

        summary_rows.append({
            "quartile": q_name,
            "n_clips": n,
            "difficulty_mean": d_mean,
            "nearest_boundary_cos": nr_mean,
            "visual_interp_cos": int_mean,
            "model_cos": mod_mean,
            "win_rate_vs_nearest": wins_nr / n,
            "win_rate_vs_interp": wins_int / n,
            "t_statistic": t_q,
            "p_value": p_q,
            "cohen_d": d_q
        })

    if args.summary_csv:
        Path(args.summary_csv).parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(summary_rows).to_csv(args.summary_csv, index=False)
        print(f"Saved quartile summary CSV to {args.summary_csv}")

    # =========================================================
    # ADVANTAGE SCALING CORRELATION
    # =========================================================
    r_nr, p_corr_nr = stats.pearsonr(df['difficulty'], df['adv_model'])
    print("="*85)
    print(f"Advantage Scaling Correlation: r = {r_nr:.4f} (p = {p_corr_nr:.2e})")
    print("="*85)

    # Breakdown by position
    print("\nBreakdown by Window Position:")
    for pos, p_df in df.groupby('position'):
        print(f"  • {pos.upper():6s} [N={len(p_df)}]: Diff={p_df['difficulty'].mean():.3f} | Nearest={p_df['nearest_cos'].mean():.4f} | Visual={p_df['interp_cos'].mean():.4f} | Model={p_df['model_cos'].mean():.4f} | WinVsNearest={p_df['win_vs_nearest'].mean()*100:.1f}%")

if __name__ == "__main__":
    main()
