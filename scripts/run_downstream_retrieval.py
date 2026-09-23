#!/usr/bin/env python3
"""
Downstream Evidence Retrieval Experiment on WildQA.

Evaluates how effectively human-written WildQA questions retrieve their ground-truth
evidence intervals from video indices under different conditions:
- Oracle (Unmasked full captions / video vectors)
- Masked (Evidence interval blanked out)
- Baseline Repeat (Nearest known boundary frame / visual inertia analog)
- Reconstructed (LLM in-filled captions, when available)
- Visual Modality Baselines (Cross-modal SigLIP video vectors, when available)

Usage:
    PYTHONPATH=src python3.9 scripts/run_downstream_retrieval.py --split dev
    PYTHONPATH=src python3.9 scripts/run_downstream_retrieval.py --split dev --embedder siglip
"""
from __future__ import annotations
import argparse
import json
import logging
import os
import sys
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd

from data.wildqa_loader import WildQAPair, load_wildqa_dataset
from evaluations.evidence_retrieval import (
    calculate_evidence_retrieval_metrics,
    build_evidence_retrieval_index,
)

logger = logging.getLogger("downstream_retrieval")


def parse_args():
    parser = argparse.ArgumentParser(description="Run Downstream Evidence Retrieval on WildQA")
    parser.add_argument("--split", choices=["dev", "test", "both"], default="dev",
                        help="WildQA split to evaluate (dev=wild4, test=wild5)")
    parser.add_argument("--embedder", choices=["siglip", "all-MiniLM-L6-v2", "all-mpnet-base-v2"],
                        default="siglip", help="Embedding model for text & queries")
    parser.add_argument("--clean-only", action="store_true", default=True,
                        help="Filter for single-interval evidence < 15s (default True)")
    parser.add_argument("--all-intervals", dest="clean_only", action="store_false",
                        help="Include multi-interval and longer evidence spans")
    parser.add_argument("--max-items", type=int, default=None,
                        help="Limit number of QA pairs to process (for quick testing)")
    parser.add_argument("--recon-source", type=str, default=None,
                        help="Path to local directory or HF run path containing reconstructed JSON files")
    parser.add_argument("--output-dir", type=str, default="results/downstream_retrieval",
                        help="Directory to save output CSV and summary reports")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    return parser.parse_args()


def load_reconstructed_clips(source: str | None, video_id: str) -> dict[int, str] | None:
    """Loads reconstructed captions for a video from local disk or Hugging Face dataset."""
    if not source:
        return None
    p = Path(source)
    local_file = p / f"{video_id}.json"
    if local_file.exists():
        try:
            with open(local_file, "r", encoding="utf-8") as f:
                d = json.load(f)
            return {int(k): v for k, v in d.get("reconstructed_captions", {}).items()}
        except Exception as e:
            logger.debug(f"Failed to load local reconstruction for {video_id}: {e}")
            return None

    try:
        from huggingface_hub import hf_hub_download
        hf_path = f"{source.rstrip('/')}/{video_id}.json"
        cached_p = hf_hub_download("Y3/dense_video_captions", hf_path, repo_type="dataset")
        with open(cached_p, "r", encoding="utf-8") as f:
            d = json.load(f)
        return {int(k): v for k, v in d.get("reconstructed_captions", {}).items()}
    except Exception as e:
        logger.debug(f"Failed to load HF reconstruction for {video_id} from {source}: {e}")
        return None


def get_embedder(embedder_name: str):
    """Initializes the requested text embedder."""
    if embedder_name == "siglip":
        from llm.local_embedder import SiglipTextEmbedder
        return SiglipTextEmbedder()
    else:
        from llm.local_embedder import LocalEmbedder
        return LocalEmbedder(embedder_name)


def load_video_captions(captions_dir: Path, video_id: str) -> list[str] | None:
    """Loads 60-second dense captions for a video if available."""
    cap_file = captions_dir / f"{video_id}.json"
    if not cap_file.exists():
        return None
    try:
        with open(cap_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        captions = [c["caption"] for c in data.get("captions", [])[:60]]
        return captions if len(captions) == 60 else None
    except Exception as e:
        logger.warning(f"Error loading captions for {video_id}: {e}")
        return None


def run_experiment():
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    splits = ["dev"] if args.split == "dev" else (["test"] if args.split == "test" else ["dev", "test"])
    embedder = get_embedder(args.embedder)
    video_embs_dir = Path("local/wild_videos_embs_siglip")
    is_siglip = (args.embedder == "siglip")

    all_rows = []

    for split in splits:
        json_path = Path(f"datasets/wildQA/{split}.json")
        cap_dir = Path(f"datasets/wildQA/captions__wild{4 if split == 'dev' else 5}")

        logger.info(f"Loading {split} dataset from {json_path}...")
        qa_pairs = load_wildqa_dataset(
            json_path,
            filter_scene_only=True,
            max_duration=60.0,
            single_evidence_only=args.clean_only,
            max_evidence_duration=15.0 if args.clean_only else None
        )

        if args.max_items:
            qa_pairs = qa_pairs[:args.max_items]

        logger.info(f"Processing {len(qa_pairs)} questions on {split}...")

        # Cache video caption embeddings across questions for the same video
        cached_caption_embs = {}
        cached_video_embs = {}

        for idx, qa in enumerate(qa_pairs, start=1):
            vid = qa.video_id
            target_indices = qa.evidence_indices(max_duration=60)
            if not target_indices:
                continue

            # Load dense captions
            if vid not in cached_caption_embs:
                captions = load_video_captions(cap_dir, vid)
                if captions is None:
                    continue
                c_vecs = embedder.get_embeddings(f"{vid}_dense_caps", captions)
                cached_caption_embs[vid] = np.array(c_vecs, dtype=np.float32)

            text_index = cached_caption_embs[vid]

            # Embed query
            q_vecs = embedder.get_embeddings(f"{vid}_q_{qa.assignment_id or idx}", [qa.question])
            query_vec = np.array(q_vecs[0], dtype=np.float32)

            # Condition 1: Text Oracle (Full captions)
            oracle_m = calculate_evidence_retrieval_metrics(query_vec, text_index, target_indices)

            # Condition 2: Text Masked (Evidence zeroed out)
            masked_idx = build_evidence_retrieval_index(text_index, target_indices, condition="masked")
            masked_m = calculate_evidence_retrieval_metrics(query_vec, masked_idx, target_indices)

            # Condition 3: Text Baseline Repeat (Nearest neighbor caption repeated)
            repeat_idx = build_evidence_retrieval_index(text_index, target_indices, condition="baseline_repeat")
            repeat_m = calculate_evidence_retrieval_metrics(query_vec, repeat_idx, target_indices)

            # Condition 3b: Text Baseline LERP (Linear interpolation weighted by distance)
            lerp_idx = build_evidence_retrieval_index(text_index, target_indices, condition="baseline_lerp")
            lerp_m = calculate_evidence_retrieval_metrics(query_vec, lerp_idx, target_indices)

            row = {
                "split": split,
                "video_id": vid,
                "domain": qa.domain,
                "question_type": ",".join(qa.question_type),
                "is_reasoning": any(t in ("Reasoning", "Motion", "Temporal relationship") for t in qa.question_type),
                "question": qa.question,
                "evidence_span": f"{min(target_indices)}-{max(target_indices)}",
                "evidence_duration": len(target_indices),
                
                # Text Oracle
                "oracle_rank": oracle_m["rank"],
                "oracle_mrr": oracle_m["mrr"],
                "oracle_r1": oracle_m["recall_at_1"],
                "oracle_r5": oracle_m["recall_at_5"],
                "oracle_sim": oracle_m["best_target_sim"],
                
                # Text Masked
                "masked_rank": masked_m["rank"],
                "masked_mrr": masked_m["mrr"],
                "masked_r1": masked_m["recall_at_1"],
                "masked_r5": masked_m["recall_at_5"],
                "masked_sim": masked_m["best_target_sim"],
                
                # Text Baseline Repeat
                "baseline_repeat_rank": repeat_m["rank"],
                "baseline_repeat_mrr": repeat_m["mrr"],
                "baseline_repeat_r1": repeat_m["recall_at_1"],
                "baseline_repeat_r5": repeat_m["recall_at_5"],
                "baseline_repeat_sim": repeat_m["best_target_sim"],

                # Text Baseline LERP
                "baseline_lerp_rank": lerp_m["rank"],
                "baseline_lerp_mrr": lerp_m["mrr"],
                "baseline_lerp_r1": lerp_m["recall_at_1"],
                "baseline_lerp_r5": lerp_m["recall_at_5"],
                "baseline_lerp_sim": lerp_m["best_target_sim"],
            }

            # Condition 4: Text Reconstructed (LLM in-filling if available)
            recon_caps = load_reconstructed_clips(args.recon_source, vid)
            if recon_caps:
                sorted_targets = sorted(target_indices)
                recon_texts = [recon_caps.get(t, "") for t in sorted_targets]
                if all(recon_texts):
                    recon_embs = np.array(embedder.get_embeddings(f"{vid}_recon", recon_texts), dtype=np.float32)
                    recon_idx = build_evidence_retrieval_index(
                        text_index, target_indices, condition="reconstructed", reconstructed_embeddings=recon_embs
                    )
                    recon_m = calculate_evidence_retrieval_metrics(query_vec, recon_idx, target_indices)
                    row.update({
                        "recon_rank": recon_m["rank"],
                        "recon_mrr": recon_m["mrr"],
                        "recon_r1": recon_m["recall_at_1"],
                        "recon_r5": recon_m["recall_at_5"],
                        "recon_sim": recon_m["best_target_sim"],
                    })

            # Optional Visual Modality Condition (Cross-modal SigLIP)
            if is_siglip:
                vid_npy = video_embs_dir / f"{vid}.npy"
                if vid_npy.exists():
                    if vid not in cached_video_embs:
                        cached_video_embs[vid] = np.load(vid_npy)[:60].astype(np.float32)
                    v_index = cached_video_embs[vid]

                    # Visual Oracle
                    v_oracle_m = calculate_evidence_retrieval_metrics(query_vec, v_index, target_indices)
                    # Visual Masked
                    v_masked_idx = build_evidence_retrieval_index(v_index, target_indices, condition="masked")
                    v_masked_m = calculate_evidence_retrieval_metrics(query_vec, v_masked_idx, target_indices)
                    # Visual Interpolated (Repeat Boundary Vector)
                    v_repeat_idx = build_evidence_retrieval_index(v_index, target_indices, condition="baseline_repeat")
                    v_repeat_m = calculate_evidence_retrieval_metrics(query_vec, v_repeat_idx, target_indices)

                    row.update({
                        "vis_oracle_mrr": v_oracle_m["mrr"],
                        "vis_oracle_r1": v_oracle_m["recall_at_1"],
                        "vis_masked_mrr": v_masked_m["mrr"],
                        "vis_interp_mrr": v_repeat_m["mrr"],
                        "vis_interp_r1": v_repeat_m["recall_at_1"],
                        "vis_interp_sim": v_repeat_m["best_target_sim"],
                    })

            all_rows.append(row)
            if idx % 25 == 0 or idx == len(qa_pairs):
                logger.info(f"[{split}] Evaluated {idx}/{len(qa_pairs)} questions")

    if not all_rows:
        logger.error("No questions were successfully evaluated!")
        return

    df = pd.DataFrame(all_rows)
    csv_file = out_dir / f"retrieval_metrics_{args.split}_{args.embedder}.csv"
    df.to_csv(csv_file, index=False)
    logger.info(f"Saved detailed results to {csv_file}")

    # Generate Summary Analysis
    report_file = out_dir / f"retrieval_summary_{args.split}_{args.embedder}.md"
    generate_summary_report(df, report_file, is_siglip=is_siglip)
    print(f"\nReport generated at {report_file}")


def generate_summary_report(df: pd.DataFrame, report_path: Path, is_siglip: bool = True):
    """Computes aggregate metrics by domain and question type, saving a markdown report."""
    # Map domain to Procedural vs Stochastic
    procedural_domains = {"Agriculture", "Military", "Human Survival"}
    df["regime"] = df["domain"].apply(lambda d: "Procedural" if d in procedural_domains else "Stochastic")

    lines = []
    lines.append(f"# Downstream Evidence Retrieval Summary Report\n")
    lines.append(f"Total evaluated questions: **{len(df)}** across **{df['video_id'].nunique()}** unique videos.\n")

    # Overall Metrics Table
    lines.append("## 1. Overall Performance Across Conditions\n")
    lines.append("| Condition | MRR | Recall@1 | Recall@5 | Mean Sim |")
    lines.append("|---|:---:|:---:|:---:|:---:|")
    lines.append(f"| **Text Oracle (Ceiling)** | {df['oracle_mrr'].mean():.4f} | {df['oracle_r1'].mean():.4f} | {df['oracle_r5'].mean():.4f} | {df['oracle_sim'].mean():.4f} |")
    lines.append(f"| **Text Masked (Black Hole)** | {df['masked_mrr'].mean():.4f} | {df['masked_r1'].mean():.4f} | {df['masked_r5'].mean():.4f} | {df['masked_sim'].mean():.4f} |")
    lines.append(f"| **Text Baseline (Repeat Nearest)** | {df['baseline_repeat_mrr'].mean():.4f} | {df['baseline_repeat_r1'].mean():.4f} | {df['baseline_repeat_r5'].mean():.4f} | {df['baseline_repeat_sim'].mean():.4f} |")
    if "baseline_lerp_mrr" in df.columns:
        lines.append(f"| **Text Baseline (LERP Interp)** | {df['baseline_lerp_mrr'].mean():.4f} | {df['baseline_lerp_r1'].mean():.4f} | {df['baseline_lerp_r5'].mean():.4f} | {df['baseline_lerp_sim'].mean():.4f} |")

    if "recon_mrr" in df.columns:
        valid_r = df.dropna(subset=["recon_mrr"])
        lines.append(f"| **Text Reconstructed (LLM)** | {valid_r['recon_mrr'].mean():.4f} | {valid_r['recon_r1'].mean():.4f} | {valid_r['recon_r5'].mean():.4f} | {valid_r['recon_sim'].mean():.4f} |")

    if is_siglip and "vis_oracle_mrr" in df.columns:
        valid_v = df.dropna(subset=["vis_oracle_mrr"])
        lines.append(f"| **Visual Oracle (Video Frames)** | {valid_v['vis_oracle_mrr'].mean():.4f} | {valid_v['vis_oracle_r1'].mean():.4f} | — | — |")
        lines.append(f"| **Visual Interp (Frame Repeat)** | {valid_v['vis_interp_mrr'].mean():.4f} | {valid_v['vis_interp_r1'].mean():.4f} | — | {valid_v['vis_interp_sim'].mean():.4f} |")
    lines.append("\n")

    # Domain Breakdown Table
    lines.append("## 2. Performance by Domain (Procedural vs. Stochastic)\n")
    lines.append("| Domain | Regime | N | Oracle MRR | Masked MRR | Baseline MRR | Masking Damage (Δ) |")
    lines.append("|---|---|:---:|:---:|:---:|:---:|:---:|")
    for dom, group in df.groupby("domain"):
        regime = "Procedural" if dom in procedural_domains else "Stochastic"
        o_mrr = group["oracle_mrr"].mean()
        m_mrr = group["masked_mrr"].mean()
        b_mrr = group["baseline_repeat_mrr"].mean()
        damage = o_mrr - m_mrr
        lines.append(f"| **{dom}** | {regime} | {len(group)} | {o_mrr:.4f} | {m_mrr:.4f} | {b_mrr:.4f} | {damage:.4f} |")
    lines.append("\n")

    # Question Type Breakdown
    lines.append("## 3. Performance by Question Category\n")
    lines.append("| Category | N | Oracle MRR | Masked MRR | Baseline MRR |")
    lines.append("|---|:---:|:---:|:---:|:---:|")
    for is_r, group in df.groupby("is_reasoning"):
        cat_name = "Reasoning / Motion / Script" if is_r else "Perceptual / Existence / Static"
        lines.append(f"| **{cat_name}** | {len(group)} | {group['oracle_mrr'].mean():.4f} | {group['masked_mrr'].mean():.4f} | {group['baseline_repeat_mrr'].mean():.4f} |")
    lines.append("\n")

    report_text = "\n".join(lines)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)
    print(report_text)


if __name__ == "__main__":
    run_experiment()
