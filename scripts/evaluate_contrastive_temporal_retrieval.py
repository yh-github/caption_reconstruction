#!/usr/bin/env python3
"""
Contrastive Temporal Retrieval Evaluator.

Evaluates how effectively LLM-reconstructed representations break 'visual/text inertia'
and isolate the true missing temporal segment against adjacent temporal distractors
(pre/post boundaries) and intra-video distractor pools (all 60 frames).

Tests the Predictability Spectrum hypothesis:
- Procedural domains (Farming, Military, Survival): LLM semantic inference breaks inertia,
  achieving superior contrastive margins and temporal retrieval accuracy over baselines.
- Stochastic domains (Nature, Scenery): Visual continuity models dominate due to smooth
  background dynamics and chaotic non-deterministic state changes.
"""

from __future__ import annotations
import argparse
import glob
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from llm.local_embedder import SiglipTextEmbedder

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Contrastive Temporal Retrieval")
    parser.add_argument(
        "--recon-dir",
        type=str,
        default="results/reconstruction/wild4_llama_w6/llama-3.1-8b__whole_window__t=0.6__fixed_fill(w=6, i=29)",
        help="Directory containing reconstructed JSON files",
    )
    parser.add_argument(
        "--captions-dir",
        type=str,
        default="datasets/wildQA/captions__wild4",
        help="Directory containing ground truth oracle captions",
    )
    parser.add_argument(
        "--video-embs-dir",
        type=str,
        default="local/wild_videos_embs_siglip",
        help="Directory containing per-second video frame embeddings (.npy)",
    )
    parser.add_argument(
        "--categories-file",
        type=str,
        default="results/video_categories.json",
        help="JSON file mapping video_id to category/domain",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/contrastive_temporal_retrieval",
        help="Directory to save output CSVs and summary markdown",
    )
    parser.add_argument(
        "--max-videos",
        type=int,
        default=None,
        help="Optional limit on number of videos to evaluate (for testing)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose debug logging",
    )
    return parser.parse_args()


def load_categories(cat_file: Path) -> dict[str, str]:
    if not cat_file.exists():
        logger.warning(f"Categories file not found at {cat_file}")
        return {}
    with open(cat_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {k: v.get("category", "Unknown") for k, v in data.items()}


def compute_retrieval_rank(query_vec: np.ndarray, candidate_pool: np.ndarray, target_idx: int) -> int:
    """
    Computes 1-based rank of candidate_pool[target_idx] when queried with query_vec.
    Rank 1 means candidate_pool[target_idx] has the highest similarity.
    """
    sims = np.dot(candidate_pool, query_vec)
    target_sim = sims[target_idx]
    # Count how many have strictly greater similarity (+ small eps to avoid float noise)
    rank = int(np.sum(sims > (target_sim + 1e-6))) + 1
    return rank


def evaluate_video(
    vid_id: str,
    recon_caps: dict[int, str],
    oracle_texts: list[str],
    video_embs: np.ndarray,
    embedder: SiglipTextEmbedder,
    category: str,
) -> list[dict[str, Any]]:
    gap_indices = sorted(recon_caps.keys())
    if not gap_indices:
        return []

    num_frames = 60
    video_embs = video_embs[:num_frames]
    oracle_texts = oracle_texts[:num_frames]
    pre_idx = max(0, gap_indices[0] - 1)
    post_idx = min(num_frames - 1, gap_indices[-1] + 1)

    # 1. Text Embeddings via SigLIP text tower
    recon_texts = [recon_caps[k] for k in gap_indices]
    all_texts = oracle_texts + recon_texts
    all_text_embs = np.array(embedder.get_embeddings(vid_id, all_texts))

    oracle_text_embs = all_text_embs[:num_frames]
    recon_text_embs = all_text_embs[num_frames:]

    # Boundaries
    pre_v = video_embs[pre_idx]
    post_v = video_embs[post_idx]
    pre_c = oracle_text_embs[pre_idx]
    post_c = oracle_text_embs[post_idx]

    records = []
    gap_len = len(gap_indices)

    for i, t in enumerate(gap_indices):
        tgt_v = video_embs[t]
        tgt_c = oracle_text_embs[t]

        # Normalized interpolation weights
        frac = (i + 1) / (gap_len + 1)

        # Nearest boundary for repeat baseline
        nearest_is_pre = (i < gap_len / 2)
        repeat_v = pre_v if nearest_is_pre else post_v
        repeat_c = pre_c if nearest_is_pre else post_c

        # Linear interpolation (LERP)
        v_lerp = (1.0 - frac) * pre_v + frac * post_v
        v_lerp = v_lerp / np.linalg.norm(v_lerp)

        c_lerp = (1.0 - frac) * pre_c + frac * post_c
        c_lerp = c_lerp / np.linalg.norm(c_lerp)

        r_llm = recon_text_embs[i]

        # ---------------------------------------------------------
        # 1. Cross-Modal Video Space (Target = True Video Frame V_tgt)
        # ---------------------------------------------------------
        # (a) LLM Text Recon -> Video Space
        sim_llm_v_tgt = float(np.dot(r_llm, tgt_v))
        sim_llm_v_pre = float(np.dot(r_llm, pre_v))
        sim_llm_v_post = float(np.dot(r_llm, post_v))
        margin_max_llm_v = sim_llm_v_tgt - max(sim_llm_v_pre, sim_llm_v_post)
        margin_mean_llm_v = sim_llm_v_tgt - 0.5 * (sim_llm_v_pre + sim_llm_v_post)
        rank_llm_v = compute_retrieval_rank(r_llm, video_embs, t)

        # (b) Visual LERP -> Video Space
        sim_vlerp_v_tgt = float(np.dot(v_lerp, tgt_v))
        sim_vlerp_v_pre = float(np.dot(v_lerp, pre_v))
        sim_vlerp_v_post = float(np.dot(v_lerp, post_v))
        margin_max_vlerp_v = sim_vlerp_v_tgt - max(sim_vlerp_v_pre, sim_vlerp_v_post)
        margin_mean_vlerp_v = sim_vlerp_v_tgt - 0.5 * (sim_vlerp_v_pre + sim_vlerp_v_post)
        rank_vlerp_v = compute_retrieval_rank(v_lerp, video_embs, t)

        # (c) Visual Repeat -> Video Space
        sim_vrep_v_tgt = float(np.dot(repeat_v, tgt_v))
        sim_vrep_v_pre = float(np.dot(repeat_v, pre_v))
        sim_vrep_v_post = float(np.dot(repeat_v, post_v))
        margin_max_vrep_v = sim_vrep_v_tgt - max(sim_vrep_v_pre, sim_vrep_v_post)
        rank_vrep_v = compute_retrieval_rank(repeat_v, video_embs, t)

        # (d) Oracle Text -> Video Space (Ceiling)
        sim_oracle_v_tgt = float(np.dot(tgt_c, tgt_v))
        sim_oracle_v_pre = float(np.dot(tgt_c, pre_v))
        sim_oracle_v_post = float(np.dot(tgt_c, post_v))
        margin_max_oracle_v = sim_oracle_v_tgt - max(sim_oracle_v_pre, sim_oracle_v_post)
        rank_oracle_v = compute_retrieval_rank(tgt_c, video_embs, t)

        # ---------------------------------------------------------
        # 2. Text Semantic Space (Target = True Caption C_tgt)
        # ---------------------------------------------------------
        # (a) LLM Text Recon -> Text Space
        sim_llm_c_tgt = float(np.dot(r_llm, tgt_c))
        sim_llm_c_pre = float(np.dot(r_llm, pre_c))
        sim_llm_c_post = float(np.dot(r_llm, post_c))
        margin_max_llm_c = sim_llm_c_tgt - max(sim_llm_c_pre, sim_llm_c_post)
        margin_mean_llm_c = sim_llm_c_tgt - 0.5 * (sim_llm_c_pre + sim_llm_c_post)
        rank_llm_c = compute_retrieval_rank(r_llm, oracle_text_embs, t)

        # (b) Text LERP -> Text Space
        sim_clerp_c_tgt = float(np.dot(c_lerp, tgt_c))
        sim_clerp_c_pre = float(np.dot(c_lerp, pre_c))
        sim_clerp_c_post = float(np.dot(c_lerp, post_c))
        margin_max_clerp_c = sim_clerp_c_tgt - max(sim_clerp_c_pre, sim_clerp_c_post)
        margin_mean_clerp_c = sim_clerp_c_tgt - 0.5 * (sim_clerp_c_pre + sim_clerp_c_post)
        rank_clerp_c = compute_retrieval_rank(c_lerp, oracle_text_embs, t)

        # (c) Text Repeat -> Text Space
        sim_crep_c_tgt = float(np.dot(repeat_c, tgt_c))
        sim_crep_c_pre = float(np.dot(repeat_c, pre_c))
        sim_crep_c_post = float(np.dot(repeat_c, post_c))
        margin_max_crep_c = sim_crep_c_tgt - max(sim_crep_c_pre, sim_crep_c_post)
        rank_crep_c = compute_retrieval_rank(repeat_c, oracle_text_embs, t)

        records.append({
            "video_id": vid_id,
            "category": category,
            "second": t,
            "gap_relative_idx": i,
            "gap_fraction": frac,
            # Target captions for inspection
            "oracle_caption": oracle_texts[t],
            "recon_caption": recon_caps[t],
            # Video Space Margins
            "margin_max_llm_v": margin_max_llm_v,
            "margin_mean_llm_v": margin_mean_llm_v,
            "margin_max_vlerp_v": margin_max_vlerp_v,
            "margin_mean_vlerp_v": margin_mean_vlerp_v,
            "margin_max_vrep_v": margin_max_vrep_v,
            "margin_max_oracle_v": margin_max_oracle_v,
            # Video Space Ranks (out of 60 frames)
            "rank_llm_v": rank_llm_v,
            "mrr_llm_v": 1.0 / rank_llm_v,
            "r1_llm_v": 1 if rank_llm_v == 1 else 0,
            "r5_llm_v": 1 if rank_llm_v <= 5 else 0,
            "rank_vlerp_v": rank_vlerp_v,
            "mrr_vlerp_v": 1.0 / rank_vlerp_v,
            "r1_vlerp_v": 1 if rank_vlerp_v == 1 else 0,
            "r5_vlerp_v": 1 if rank_vlerp_v <= 5 else 0,
            "rank_vrep_v": rank_vrep_v,
            "mrr_vrep_v": 1.0 / rank_vrep_v,
            "r1_vrep_v": 1 if rank_vrep_v == 1 else 0,
            "r5_vrep_v": 1 if rank_vrep_v <= 5 else 0,
            "rank_oracle_v": rank_oracle_v,
            "mrr_oracle_v": 1.0 / rank_oracle_v,
            # Text Space Margins
            "margin_max_llm_c": margin_max_llm_c,
            "margin_mean_llm_c": margin_mean_llm_c,
            "margin_max_clerp_c": margin_max_clerp_c,
            "margin_mean_clerp_c": margin_mean_clerp_c,
            "margin_max_crep_c": margin_max_crep_c,
            # Text Space Ranks (out of 60 captions)
            "rank_llm_c": rank_llm_c,
            "mrr_llm_c": 1.0 / rank_llm_c,
            "r1_llm_c": 1 if rank_llm_c == 1 else 0,
            "r5_llm_c": 1 if rank_llm_c <= 5 else 0,
            "rank_clerp_c": rank_clerp_c,
            "mrr_clerp_c": 1.0 / rank_clerp_c,
            "r1_clerp_c": 1 if rank_clerp_c == 1 else 0,
            "r5_clerp_c": 1 if rank_clerp_c <= 5 else 0,
            "rank_crep_c": rank_crep_c,
            "mrr_crep_c": 1.0 / rank_crep_c,
            "r1_crep_c": 1 if rank_crep_c == 1 else 0,
            "r5_crep_c": 1 if rank_crep_c <= 5 else 0,
        })

    return records


def generate_summary_report(df: pd.DataFrame, output_path: Path):
    """Generates comprehensive markdown report with tables, domain breakdown, and stats."""
    procedural_cats = {"Farming", "Military", "Survival"}
    df["macro_domain"] = df["category"].apply(lambda c: "Procedural" if c in procedural_cats else "Stochastic")

    by_category = df.groupby("category").agg({
        "video_id": "nunique",
        "margin_max_llm_c": "mean",
        "margin_max_clerp_c": "mean",
        "mrr_llm_c": "mean",
        "mrr_clerp_c": "mean",
        "margin_max_llm_v": "mean",
        "margin_max_vlerp_v": "mean",
        "mrr_llm_v": "mean",
        "mrr_vlerp_v": "mean",
    }).rename(columns={"video_id": "N_videos"})

    by_macro = df.groupby("macro_domain").agg({
        "video_id": "nunique",
        "margin_max_llm_c": "mean",
        "margin_max_clerp_c": "mean",
        "mrr_llm_c": "mean",
        "mrr_clerp_c": "mean",
        "margin_max_llm_v": "mean",
        "margin_max_vlerp_v": "mean",
        "mrr_llm_v": "mean",
        "mrr_vlerp_v": "mean",
    }).rename(columns={"video_id": "N_videos"})

    # Statistical tests
    try:
        w_text_stat, w_text_p = stats.wilcoxon(df["margin_max_llm_c"], df["margin_max_clerp_c"], alternative="two-sided")
    except Exception:
        w_text_stat, w_text_p = 0.0, 1.0
    try:
        w_vid_stat, w_vid_p = stats.wilcoxon(df["margin_max_llm_v"], df["margin_max_vlerp_v"], alternative="two-sided")
    except Exception:
        w_vid_stat, w_vid_p = 0.0, 1.0

    llm_win_text = (df["margin_max_llm_c"] > df["margin_max_clerp_c"]).mean() * 100
    llm_win_vid = (df["margin_max_llm_v"] > df["margin_max_vlerp_v"]).mean() * 100

    report = f"""# Contrastive Temporal Retrieval Benchmark Report

## 1. Overview
Evaluated **{df['video_id'].nunique()} unique videos** across **{len(df)} total masked seconds**.
Target task: Can the reconstruction break boundary inertia and match the true missing temporal segment ($V_{{target}}$ / $C_{{target}}$) better than adjacent temporal boundaries ($Pre$, $Post$)?

---

## 2. Macro Domain Breakdown (Procedural vs. Stochastic)

| Macro Domain | N Videos | LLM Text Margin | Text LERP Margin | LLM Margin Advantage (Δ) | LLM Video MRR | Vis LERP Video MRR |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
"""
    for domain, row in by_macro.iterrows():
        delta_margin = row['margin_max_llm_c'] - row['margin_max_clerp_c']
        report += (
            f"| **{domain}** | {int(row['N_videos'])} | "
            f"`{row['margin_max_llm_c']:+.4f}` | `{row['margin_max_clerp_c']:+.4f}` | "
            f"**`{delta_margin:+.4f}`** | `{row['mrr_llm_v']:.4f}` | `{row['mrr_vlerp_v']:.4f}` |\n"
        )

    report += """
---

## 3. Granular Category Breakdown

| Category | N Videos | LLM Text Margin | Text LERP Margin | Text Margin Δ | LLM Video Margin | Vis LERP Video Margin | Video Margin Δ |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
"""
    for cat, row in by_category.iterrows():
        delta_text = row['margin_max_llm_c'] - row['margin_max_clerp_c']
        delta_vid = row['margin_max_llm_v'] - row['margin_max_vlerp_v']
        report += (
            f"| **{cat}** | {int(row['N_videos'])} | "
            f"`{row['margin_max_llm_c']:+.4f}` | `{row['margin_max_clerp_c']:+.4f}` | **`{delta_text:+.4f}`** | "
            f"`{row['margin_max_llm_v']:+.4f}` | `{row['margin_max_vlerp_v']:+.4f}` | **`{delta_vid:+.4f}`** |\n"
        )

    report += f"""
---

## 4. Overall Benchmark Metrics & Statistical Significance

### (A) Text Semantic Space: LLM Recon vs. Text LERP
* **Overall LLM Mean Margin**: `{df['margin_max_llm_c'].mean():+.4f}`
* **Overall Text LERP Mean Margin**: `{df['margin_max_clerp_c'].mean():+.4f}`
* **LLM Margin Win Rate**: **{llm_win_text:.1f}%**
* **Wilcoxon Signed-Rank Test**: \\(p = {w_text_p:.4e}\\) ({'Significant' if w_text_p < 0.05 else 'Not significant'})

### (B) Cross-Modal Video Space: LLM Recon vs. Visual LERP
* **Overall LLM Mean Margin**: `{df['margin_max_llm_v'].mean():+.4f}`
* **Overall Visual LERP Mean Margin**: `{df['margin_max_vlerp_v'].mean():+.4f}`
* **LLM Margin Win Rate**: **{llm_win_vid:.1f}%**
* **Wilcoxon Signed-Rank Test**: \\(p = {w_vid_p:.4e}\\) ({'Significant' if w_vid_p < 0.05 else 'Not significant'})

---

## 5. Temporal Frame Retrieval (60-Frame Intra-Video Pool)

| Condition | Target Modality | MRR | Recall@1 | Recall@5 |
|---|---|:---:|:---:|:---:|
| **LLM Recon (Text)** | Text Oracle | `{df['mrr_llm_c'].mean():.4f}` | `{df['r1_llm_c'].mean()*100:.1f}%` | `{df['r5_llm_c'].mean()*100:.1f}%` |
| **Text LERP Baseline** | Text Oracle | `{df['mrr_clerp_c'].mean():.4f}` | `{df['r1_clerp_c'].mean()*100:.1f}%` | `{df['r5_clerp_c'].mean()*100:.1f}%` |
| **Text Repeat Baseline** | Text Oracle | `{df['mrr_crep_c'].mean():.4f}` | `{df['r1_crep_c'].mean()*100:.1f}%` | `{df['r5_crep_c'].mean()*100:.1f}%` |
| **LLM Recon (Cross-Modal)** | Video Frames | `{df['mrr_llm_v'].mean():.4f}` | `{df['r1_llm_v'].mean()*100:.1f}%` | `{df['r5_llm_v'].mean()*100:.1f}%` |
| **Visual LERP Baseline** | Video Frames | `{df['mrr_vlerp_v'].mean():.4f}` | `{df['r1_vlerp_v'].mean()*100:.1f}%` | `{df['r5_vlerp_v'].mean()*100:.1f}%` |
| **Visual Repeat Baseline** | Video Frames | `{df['mrr_vrep_v'].mean():.4f}` | `{df['r1_vrep_v'].mean()*100:.1f}%` | `{df['r5_vrep_v'].mean()*100:.1f}%` |
| **Oracle Caption (Ceiling)** | Video Frames | `{df['mrr_oracle_v'].mean():.4f}` | — | — |
"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report)
    logger.info(f"Summary report saved to {output_path}")


def main():
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    recon_dir = Path(args.recon_dir)
    captions_dir = Path(args.captions_dir)
    video_embs_dir = Path(args.video_embs_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    categories = load_categories(Path(args.categories_file))
    embedder = SiglipTextEmbedder()

    recon_files = sorted(glob.glob(str(recon_dir / "*.json")))
    if args.max_videos:
        recon_files = recon_files[:args.max_videos]

    logger.info(f"Found {len(recon_files)} reconstruction files in {recon_dir}")

    all_records = []
    skipped = 0

    for i, rf in enumerate(recon_files):
        vid_id = os.path.splitext(os.path.basename(rf))[0]
        vid_npy = video_embs_dir / f"{vid_id}.npy"
        cap_json = captions_dir / f"{vid_id}.json"

        if not vid_npy.exists() or not cap_json.exists():
            skipped += 1
            continue

        try:
            video_embs = np.load(vid_npy)
            with open(cap_json, "r", encoding="utf-8") as f:
                cap_data = json.load(f)
            oracle_texts = [c["caption"] for c in cap_data.get("captions", [])[:60]]

            if len(oracle_texts) < 60 or len(video_embs) < 60:
                skipped += 1
                continue

            with open(rf, "r", encoding="utf-8") as f:
                recon_data = json.load(f)
            recon_caps_raw = recon_data.get("reconstructed_captions", {})
            recon_caps = {int(k): v for k, v in recon_caps_raw.items()}

            category = categories.get(vid_id, "Unknown")

            vid_records = evaluate_video(
                vid_id=vid_id,
                recon_caps=recon_caps,
                oracle_texts=oracle_texts,
                video_embs=video_embs,
                embedder=embedder,
                category=category,
            )
            all_records.extend(vid_records)

            if (i + 1) % 10 == 0 or (i + 1) == len(recon_files):
                logger.info(f"Processed {i + 1}/{len(recon_files)} videos ({len(all_records)} seconds evaluated)")

        except Exception as e:
            logger.warning(f"Error evaluating {vid_id}: {e}")
            skipped += 1

    if not all_records:
        logger.error("No records successfully evaluated!")
        sys.exit(1)

    df = pd.DataFrame(all_records)
    csv_path = output_dir / "contrastive_metrics_per_second.csv"
    df.to_csv(csv_path, index=False)
    logger.info(f"Per-second metrics saved to {csv_path}")

    # Video-level aggregation
    video_df = df.groupby(["video_id", "category"]).agg({
        "margin_max_llm_c": "mean",
        "margin_max_clerp_c": "mean",
        "margin_max_crep_c": "mean",
        "mrr_llm_c": "mean",
        "mrr_clerp_c": "mean",
        "margin_max_llm_v": "mean",
        "margin_max_vlerp_v": "mean",
        "margin_max_vrep_v": "mean",
        "mrr_llm_v": "mean",
        "mrr_vlerp_v": "mean",
    }).reset_index()
    video_csv_path = output_dir / "contrastive_metrics_per_video.csv"
    video_df.to_csv(video_csv_path, index=False)
    logger.info(f"Per-video metrics saved to {video_csv_path}")

    # Generate Summary Report
    summary_path = output_dir / "contrastive_retrieval_summary.md"
    generate_summary_report(df, summary_path)
    logger.info("Done!")


if __name__ == "__main__":
    main()
