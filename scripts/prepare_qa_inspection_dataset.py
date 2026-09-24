#!/usr/bin/env python3
"""
Prepare QA Inspection Dataset for Manual and LLM Evaluation.

Constructs comprehensive inspection records for downstream WildQA questions,
comparing Oracle, Masked, Baseline Repeat, and LLM Reconstructed captions.

For each question, extracts:
1. Question and ground-truth human reference answers.
2. Boundary context: pre-gap and post-gap unmasked captions.
3. In-gap captions under:
   - Oracle (unmasked VLM captions)
   - Baseline Repeat (nearest neighbor visual inertia)
   - Reconstructed (LLM in-filled captions, e.g. LLaMA-3.1-8B)
4. (Optional) Top-K retrieved captions from the 60s temporal index under each condition.
5. Automated heuristic flags (e.g. lexical leakage in boundary frames).

Outputs:
- JSONL format: For automated LLM-as-a-judge pipelines.
- HTML report: Interactive, human-readable inspection dashboard.
- Markdown summary: Clean documentation of inspected samples.

Usage:
    # Quick dry-run (fast, no embedder, first 5 questions):
    PYTHONPATH=src python3.9 scripts/prepare_qa_inspection_dataset.py --split dev --embedder none --max-items 5

    # Full inspection run with SigLIP retrieval on dev set:
    PYTHONPATH=src python3.9 scripts/prepare_qa_inspection_dataset.py --split dev --embedder siglip

    # Full inspection run on test set:
    PYTHONPATH=src python3.9 scripts/prepare_qa_inspection_dataset.py --split test --embedder siglip
"""
from __future__ import annotations

import argparse
import html
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from data.wildqa_loader import WildQAPair, load_wildqa_dataset
from evaluations.evidence_retrieval import (
    build_evidence_retrieval_index,
    calculate_evidence_retrieval_metrics,
)

logger = logging.getLogger("qa_inspection")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare QA Inspection Dataset for Manual & LLM Review"
    )
    parser.add_argument(
        "--split",
        choices=["dev", "test", "both"],
        default="dev",
        help="WildQA split to evaluate (dev=wild4, test=wild5, default: dev)",
    )
    parser.add_argument(
        "--recon-source",
        type=str,
        default=None,
        help="Path to directory containing reconstructed JSON files (auto-resolved if not specified)",
    )
    parser.add_argument(
        "--embedder",
        choices=["siglip", "all-mpnet-base-v2", "none"],
        default="siglip",
        help="Embedder to compute temporal retrieval and top-k candidates (default: siglip)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Number of top retrieved candidates to extract for each condition (default: 3)",
    )
    parser.add_argument(
        "--boundary-window",
        type=int,
        default=3,
        help="Seconds of boundary context before and after the gap to include (default: 3)",
    )
    parser.add_argument(
        "--clean-only",
        action="store_true",
        default=True,
        help="Filter for single-interval evidence < 15s (default True)",
    )
    parser.add_argument(
        "--all-intervals",
        dest="clean_only",
        action="store_false",
        help="Include multi-interval and longer evidence spans",
    )
    parser.add_argument(
        "--max-items",
        type=int,
        default=None,
        help="Limit number of QA pairs to process (for quick testing)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/qa_inspection",
        help="Directory to save output inspection files (default: results/qa_inspection)",
    )
    parser.add_argument(
        "--format",
        choices=["all", "jsonl", "html", "markdown"],
        default="all",
        help="Output formats to generate (default: all)",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    return parser.parse_args()


def resolve_recon_source(split: str, user_source: str | None) -> Path | None:
    """
    Auto-detects the reconstructed captions path for the given split if not explicitly provided.
    Checks local results directory and Hugging Face dataset snapshots.
    """
    if user_source:
        p = Path(user_source)
        if p.exists():
            return p
        logger.warning(f"Specified recon-source path '{user_source}' does not exist.")
        return None

    dataset_name = "dev" if split == "dev" else "test"
    wild_num = "4" if split == "dev" else "5"
    target_subpath = f"reconstruction/wild{wild_num}_llama_evidence/llama-3.1-8b__whole_window__t=0.6__evidence(dataset={dataset_name})"

    # 1. Check local results folder
    local_candidate = PROJECT_ROOT / "results" / target_subpath
    if local_candidate.exists():
        logger.info(f"Auto-resolved local recon-source: {local_candidate}")
        return local_candidate

    # 2. Check Hugging Face hub snapshots cache
    hf_cache_dir = Path.home() / ".cache" / "huggingface" / "hub" / "datasets--Y3--dense_video_captions" / "snapshots"
    if hf_cache_dir.exists():
        for snap in sorted(hf_cache_dir.iterdir(), reverse=True):
            if snap.is_dir():
                cand = snap / target_subpath
                if cand.exists():
                    logger.info(f"Auto-resolved HF snapshot recon-source: {cand}")
                    return cand

    logger.warning(
        f"Could not auto-resolve reconstructed captions directory for split '{split}'. "
        f"Reconstruction fields will be marked unavailable."
    )
    return None


def load_reconstructed_captions(source_dir: Path | None, video_id: str) -> dict[int, str] | None:
    """Loads reconstructed captions for a video ID from a directory of JSON files."""
    if not source_dir:
        return None
    file_path = source_dir / f"{video_id}.json"
    if not file_path.exists():
        return None
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        recons = data.get("reconstructed_captions", {})
        return {int(k): str(v) for k, v in recons.items()}
    except Exception as e:
        logger.warning(f"Error loading reconstructed captions for {video_id}: {e}")
        return None


def load_video_captions(captions_dir: Path, video_id: str) -> list[str] | None:
    """Loads dense 60-second captions for a video from disk."""
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


def get_embedder_instance(embedder_name: str):
    """Initializes the requested text embedder or returns None."""
    if embedder_name == "none":
        return None
    elif embedder_name == "siglip":
        from llm.local_embedder import SiglipTextEmbedder
        return SiglipTextEmbedder()
    else:
        from llm.local_embedder import LocalEmbedder
        return LocalEmbedder(embedder_name)


def extract_top_k_candidates(
    index_vectors: np.ndarray,
    query_vector: np.ndarray,
    all_captions: list[str],
    target_indices: set[int],
    top_k: int = 3,
) -> list[dict[str, Any]]:
    """
    Returns the top-K ranked captions from the 60-second index along with similarity scores.
    """
    q_norm = np.linalg.norm(query_vector)
    q_unit = query_vector / (q_norm + 1e-12)

    i_norms = np.linalg.norm(index_vectors, axis=1, keepdims=True)
    i_norms = np.where(i_norms == 0, 1.0, i_norms)
    i_unit = index_vectors / i_norms

    sims = np.dot(i_unit, q_unit)  # Shape: (60,)
    sorted_indices = np.argsort(-sims)[:top_k]

    results = []
    for rank_idx, sec in enumerate(sorted_indices, start=1):
        sec_int = int(sec)
        results.append({
            "rank": rank_idx,
            "second": sec_int,
            "timestamp": f"00:{sec_int:02d}",
            "caption": all_captions[sec_int] if sec_int < len(all_captions) else "",
            "similarity": round(float(sims[sec_int]), 4),
            "is_in_evidence": sec_int in target_indices,
        })
    return results


def check_lexical_leakage(text_list: list[str], answers: list[str]) -> bool:
    """Checks if any significant answer token or substring is present in the surrounding text."""
    joined_text = " ".join(text_list).lower()
    for ans in answers:
        clean_ans = ans.lower().strip()
        if len(clean_ans) > 2 and clean_ans in joined_text:
            return True
        # Check individual non-stopword tokens if multi-word
        tokens = [w for w in clean_ans.split() if len(w) > 3 and w not in {"the", "and", "with", "from", "that", "this"}]
        for t in tokens:
            if t in joined_text:
                return True
    return False


def build_inspection_item(
    qa: WildQAPair,
    split: str,
    original_captions: list[str],
    recon_captions: dict[int, str] | None,
    embedder: Any | None,
    boundary_window: int = 3,
    top_k: int = 3,
    cached_caption_embs: dict[str, np.ndarray] | None = None,
) -> dict[str, Any]:
    """Builds a single comprehensive inspection record for a QA pair."""
    target_indices = sorted(qa.evidence_indices(max_duration=60))
    target_set = set(target_indices)
    t_min = min(target_indices)
    t_max = max(target_indices)

    # 1. Boundary Context (Pre-gap and Post-gap)
    pre_gap_indices = [t for t in range(t_min - boundary_window, t_min) if 0 <= t < 60]
    post_gap_indices = [t for t in range(t_max + 1, t_max + 1 + boundary_window) if 0 <= t < 60]

    pre_gap_caps = [
        {"second": t, "timestamp": f"00:{t:02d}", "caption": original_captions[t]}
        for t in pre_gap_indices
    ]
    post_gap_caps = [
        {"second": t, "timestamp": f"00:{t:02d}", "caption": original_captions[t]}
        for t in post_gap_indices
    ]

    # Check for lexical leakage in boundary frames
    boundary_texts = [c["caption"] for c in pre_gap_caps + post_gap_caps]
    leakage_detected = check_lexical_leakage(boundary_texts, qa.all_reference_answers)

    # 2. Evidence Gap Captions under Different Conditions
    oracle_gap = [
        {"second": t, "timestamp": f"00:{t:02d}", "caption": original_captions[t]}
        for t in target_indices
    ]

    # Baseline Repeat: Repeat nearest unmasked boundary frame
    left_boundary = t_min - 1 if t_min > 0 else (t_max + 1 if t_max + 1 < 60 else None)
    repeat_text = original_captions[left_boundary] if left_boundary is not None else ""
    baseline_repeat_gap = [
        {"second": t, "timestamp": f"00:{t:02d}", "caption": repeat_text}
        for t in target_indices
    ]

    # Reconstructed Captions
    recon_available = False
    reconstructed_gap = []
    if recon_captions:
        recon_texts = [recon_captions.get(t, "") for t in target_indices]
        if any(recon_texts):
            recon_available = True
            reconstructed_gap = [
                {"second": t, "timestamp": f"00:{t:02d}", "caption": recon_captions.get(t, "")}
                for t in target_indices
            ]

    # 3. Retrieval & Top-K Candidates (if embedder is active)
    retrieval_info = {}
    top_candidates = {}

    if embedder is not None and cached_caption_embs is not None:
        vid = qa.video_id
        if vid not in cached_caption_embs:
            vecs = embedder.get_embeddings(f"{vid}_dense_caps", original_captions)
            cached_caption_embs[vid] = np.array(vecs, dtype=np.float32)

        text_index = cached_caption_embs[vid]
        q_vec = np.array(
            embedder.get_embeddings(f"{vid}_q_{qa.assignment_id or 'q'}", [qa.question])[0],
            dtype=np.float32,
        )

        # Oracle Condition
        oracle_m = calculate_evidence_retrieval_metrics(q_vec, text_index, target_set, top_k=top_k)
        retrieval_info["oracle"] = {
            "rank": oracle_m["rank"],
            "mrr": round(oracle_m["mrr"], 4),
            "recall_at_1": oracle_m["recall_at_1"],
            f"recall_at_{top_k}": oracle_m[f"recall_at_{top_k}"],
            "best_sim": oracle_m["best_target_sim"],
        }
        top_candidates["oracle"] = extract_top_k_candidates(
            text_index, q_vec, original_captions, target_set, top_k=top_k
        )

        # Masked Condition
        masked_idx = build_evidence_retrieval_index(text_index, target_set, condition="masked")
        masked_m = calculate_evidence_retrieval_metrics(q_vec, masked_idx, target_set, top_k=top_k)
        retrieval_info["masked"] = {
            "rank": masked_m["rank"],
            "mrr": round(masked_m["mrr"], 4),
            "recall_at_1": masked_m["recall_at_1"],
            f"recall_at_{top_k}": masked_m[f"recall_at_{top_k}"],
            "best_sim": masked_m["best_target_sim"],
        }
        masked_caps = list(original_captions)
        for t in target_set:
            masked_caps[t] = "[MASKED EVIDENCE INTERVAL - NO TEXT]"
        top_candidates["masked"] = extract_top_k_candidates(
            masked_idx, q_vec, masked_caps, target_set, top_k=top_k
        )

        # Baseline Repeat Condition
        repeat_idx = build_evidence_retrieval_index(text_index, target_set, condition="baseline_repeat")
        repeat_m = calculate_evidence_retrieval_metrics(q_vec, repeat_idx, target_set, top_k=top_k)
        retrieval_info["baseline_repeat"] = {
            "rank": repeat_m["rank"],
            "mrr": round(repeat_m["mrr"], 4),
            "recall_at_1": repeat_m["recall_at_1"],
            f"recall_at_{top_k}": repeat_m[f"recall_at_{top_k}"],
            "best_sim": repeat_m["best_target_sim"],
        }
        repeat_caps = list(original_captions)
        for t in target_set:
            repeat_caps[t] = repeat_text
        top_candidates["baseline_repeat"] = extract_top_k_candidates(
            repeat_idx, q_vec, repeat_caps, target_set, top_k=top_k
        )

        # Reconstructed Condition (if available)
        if recon_available:
            r_texts = [c["caption"] for c in reconstructed_gap]
            r_embs = np.array(
                embedder.get_embeddings(f"{vid}_recon", r_texts), dtype=np.float32
            )
            recon_idx = build_evidence_retrieval_index(
                text_index, target_set, condition="reconstructed", reconstructed_embeddings=r_embs
            )
            recon_m = calculate_evidence_retrieval_metrics(q_vec, recon_idx, target_set, top_k=top_k)
            retrieval_info["reconstructed"] = {
                "rank": recon_m["rank"],
                "mrr": round(recon_m["mrr"], 4),
                "recall_at_1": recon_m["recall_at_1"],
                f"recall_at_{top_k}": recon_m[f"recall_at_{top_k}"],
                "best_sim": recon_m["best_target_sim"],
            }
            recon_full_caps = list(original_captions)
            for t, cap_obj in zip(target_indices, reconstructed_gap):
                recon_full_caps[t] = cap_obj["caption"]
            top_candidates["reconstructed"] = extract_top_k_candidates(
                recon_idx, q_vec, recon_full_caps, target_set, top_k=top_k
            )

    return {
        "assignment_id": qa.assignment_id,
        "video_id": qa.video_id,
        "split": split,
        "domain": qa.domain,
        "question_type": qa.question_type,
        "is_reasoning": any(
            t in ("Reasoning", "Motion", "Temporal relationship") for t in qa.question_type
        ),
        "question": qa.question,
        "ground_truth_answer": qa.answer,
        "all_reference_answers": qa.all_reference_answers,
        "evidence_span": f"{t_min:02d}-{t_max:02d}",
        "evidence_duration_sec": len(target_indices),
        "boundary_context": {
            "pre_gap": pre_gap_caps,
            "post_gap": post_gap_caps,
            "leakage_suspected": leakage_detected,
        },
        "gap_conditions": {
            "oracle": oracle_gap,
            "baseline_repeat": baseline_repeat_gap,
            "reconstructed": reconstructed_gap if recon_available else None,
        },
        "recon_available": recon_available,
        "retrieval_metrics": retrieval_info,
        "top_k_candidates": top_candidates,
    }


def export_html_report(items: list[dict[str, Any]], output_path: Path, embedder_name: str):
    """Generates an interactive, styled HTML dashboard for manual audit and inspection."""
    total = len(items)
    with_recon = sum(1 for it in items if it["recon_available"])
    leakage_count = sum(1 for it in items if it["boundary_context"]["leakage_suspected"])

    html_parts = [
        "<!DOCTYPE html>",
        "<html lang='en'>",
        "<head>",
        "<meta charset='UTF-8'>",
        "<title>WildQA Caption Reconstruction - Inspection Dashboard</title>",
        "<style>",
        "  body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; margin: 0; padding: 20px; background: #f8fafc; color: #1e293b; line-height: 1.5; }",
        "  .container { max-width: 1200px; margin: 0 auto; }",
        "  header { background: #ffffff; padding: 24px; border-radius: 12px; border: 1px solid #e2e8f0; margin-bottom: 24px; box-shadow: 0 1px 3px rgba(0,0,0,0.05); }",
        "  h1 { margin-top: 0; color: #0f172a; font-size: 24px; }",
        "  .stats-bar { display: flex; gap: 20px; flex-wrap: wrap; margin-top: 12px; }",
        "  .stat-card { background: #f1f5f9; padding: 12px 18px; border-radius: 8px; font-size: 14px; }",
        "  .stat-card strong { display: block; font-size: 20px; color: #2563eb; }",
        "  .rubric-box { background: #eff6ff; border-left: 4px solid #3b82f6; padding: 16px; border-radius: 0 8px 8px 0; margin-top: 16px; font-size: 14px; }",
        "  .qa-card { background: #ffffff; border: 1px solid #cbd5e1; border-radius: 12px; padding: 20px; margin-bottom: 24px; box-shadow: 0 2px 4px rgba(0,0,0,0.04); }",
        "  .qa-header { display: flex; justify-content: space-between; align-items: baseline; flex-wrap: wrap; border-bottom: 1px solid #f1f5f9; padding-bottom: 12px; margin-bottom: 16px; }",
        "  .badge { display: inline-block; padding: 4px 10px; border-radius: 12px; font-size: 12px; font-weight: 600; text-transform: uppercase; margin-right: 6px; }",
        "  .badge-domain { background: #e0e7ff; color: #3730a3; }",
        "  .badge-type { background: #fef3c7; color: #92400e; }",
        "  .badge-evidence { background: #f1f5f9; color: #475569; }",
        "  .badge-leak { background: #fee2e2; color: #b91c1c; }",
        "  .question-text { font-size: 18px; font-weight: 600; color: #0f172a; margin: 8px 0; }",
        "  .answer-box { background: #f0fdf4; border: 1px solid #bbf7d0; border-radius: 8px; padding: 10px 14px; margin-bottom: 16px; font-size: 14px; color: #166534; }",
        "  .grid-2 { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }",
        "  @media (max-width: 800px) { .grid-2 { grid-template-columns: 1fr; } }",
        "  .section-title { font-size: 13px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.05em; color: #64748b; margin-bottom: 8px; }",
        "  .context-box { background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; padding: 12px; font-size: 13px; }",
        "  .caption-item { margin-bottom: 6px; }",
        "  .sec-tag { font-family: monospace; font-size: 11px; background: #e2e8f0; padding: 2px 5px; border-radius: 4px; color: #334155; margin-right: 6px; }",
        "  .table-candidates { width: 100%; border-collapse: collapse; font-size: 12px; margin-top: 6px; }",
        "  .table-candidates th, .table-candidates td { border: 1px solid #e2e8f0; padding: 6px 8px; text-align: left; }",
        "  .table-candidates th { background: #f1f5f9; font-weight: 600; color: #475569; }",
        "  .hit-target { background: #dcfce7; font-weight: 600; }",
        "  .eval-form { margin-top: 16px; padding: 12px; background: #fafafa; border-radius: 8px; border: 1px dashed #cbd5e1; font-size: 13px; }",
        "  .eval-form label { margin-right: 18px; font-weight: 500; cursor: pointer; }",
        "</style>",
        "</head>",
        "<body>",
        "<div class='container'>",
        "<header>",
        "  <h1>WildQA Temporal Caption Reconstruction & Downstream Q&A Audit</h1>",
        f"  <p style='color: #64748b; margin: 4px 0 0 0;'>Generated with embedder: <code>{embedder_name}</code> | Evaluation Split: <strong>{items[0]['split'] if items else 'N/A'}</strong></p>",
        "  <div class='stats-bar'>",
        f"    <div class='stat-card'><strong>{total}</strong> Total Questions</div>",
        f"    <div class='stat-card'><strong>{with_recon}</strong> With LLM Reconstruction</div>",
        f"    <div class='stat-card'><strong>{leakage_count}</strong> Lexical Leakage Hints</div>",
        "  </div>",
        "  <div class='rubric-box'>",
        "    <strong>Inspection Rubric:</strong><br>",
        "    • <strong>Q1 (Boundary Leakage)</strong>: Was the exact fact already revealed in the unmasked pre/post context frames?<br>",
        "    • <strong>Q2 (Intrinsic Reconstruction)</strong>: Did the LLM in-fill produce the <em>Exact Fact</em>, the <em>Semantic/Type Class</em>, a <em>Plausible Hallucination</em>, or a <em>Miss</em>?<br>",
        "    • <strong>Q3 (Downstream Feasibility)</strong>: Can a reader model answer the question using the Top-K candidates retrieved under <em>Reconstructed</em> vs <em>Masked</em>?",
        "  </div>",
        "</header>",
    ]

    for idx, it in enumerate(items, start=1):
        q = html.escape(it["question"])
        ans = html.escape(it["ground_truth_answer"])
        alt_ans = [html.escape(a) for a in it["all_reference_answers"] if a != it["ground_truth_answer"]]
        dom = html.escape(it["domain"])
        types = html.escape(", ".join(it["question_type"]))
        vid = html.escape(it["video_id"])
        span = it["evidence_span"]
        leak_badge = "<span class='badge badge-leak'>Leakage Hint</span>" if it["boundary_context"]["leakage_suspected"] else ""

        # Pre/Post Gap HTML
        pre_lines = "".join(
            f"<div class='caption-item'><span class='sec-tag'>{c['timestamp']}</span>{html.escape(c['caption'])}</div>"
            for c in it["boundary_context"]["pre_gap"]
        ) or "<em style='color:#94a3b8;'>None (start of video)</em>"

        post_lines = "".join(
            f"<div class='caption-item'><span class='sec-tag'>{c['timestamp']}</span>{html.escape(c['caption'])}</div>"
            for c in it["boundary_context"]["post_gap"]
        ) or "<em style='color:#94a3b8;'>None (end of video)</em>"

        # Gap Conditions HTML
        oracle_lines = "".join(
            f"<div class='caption-item'><span class='sec-tag'>{c['timestamp']}</span>{html.escape(c['caption'])}</div>"
            for c in it["gap_conditions"]["oracle"]
        )
        baseline_lines = "".join(
            f"<div class='caption-item'><span class='sec-tag'>{c['timestamp']}</span>{html.escape(c['caption'])}</div>"
            for c in it["gap_conditions"]["baseline_repeat"]
        )
        recon_data = it["gap_conditions"]["reconstructed"]
        if recon_data:
            recon_lines = "".join(
                f"<div class='caption-item'><span class='sec-tag'>{c['timestamp']}</span>{html.escape(c['caption'])}</div>"
                for c in recon_data
            )
        else:
            recon_lines = "<em style='color:#dc2626;'>Reconstruction not available for this sample</em>"

        # Top-K candidate table (if embedder was run)
        candidates_html = ""
        if it["top_k_candidates"]:
            cand_rows = []
            cand_rows.append("<div style='margin-top: 12px;'><div class='section-title'>Top Retrieved Candidates from 60s Video Index</div>")
            cand_rows.append("<table class='table-candidates'><tr><th>Condition</th><th>Rank</th><th>Time</th><th>Retrieved Caption</th><th>Similarity</th><th>Hit?</th></tr>")

            for cond_name, label in [("oracle", "Oracle"), ("masked", "Masked"), ("baseline_repeat", "Repeat"), ("reconstructed", "Recon (LLM)")]:
                cands = it["top_k_candidates"].get(cond_name, [])
                for c in cands:
                    hit_cls = "class='hit-target'" if c["is_in_evidence"] else ""
                    hit_txt = "Evidence Hit" if c["is_in_evidence"] else "Distractor"
                    cand_rows.append(
                        f"<tr {hit_cls}><td><strong>{label}</strong></td><td>#{c['rank']}</td><td>{c['timestamp']}</td><td>{html.escape(c['caption'])}</td><td>{c['similarity']:.4f}</td><td>{hit_txt}</td></tr>"
                    )
            cand_rows.append("</table></div>")
            candidates_html = "".join(cand_rows)

        card_html = f"""
        <div class='qa-card' id='sample-{idx}'>
          <div class='qa-header'>
            <div>
              <span class='badge badge-domain'>{dom}</span>
              <span class='badge badge-type'>{types}</span>
              <span class='badge badge-evidence'>Evidence: {span}s ({it['evidence_duration_sec']}s)</span>
              {leak_badge}
            </div>
            <div style='color: #64748b; font-size: 13px;'>Sample #{idx} | Video: <code>{vid}</code></div>
          </div>

          <div class='question-text'>Q: {q}</div>
          <div class='answer-box'>
            <strong>Ground-Truth Answer:</strong> {ans}
            {f"<div style='font-size: 12px; color: #15803d; margin-top: 4px;'><em>Alternates: {', '.join(alt_ans)}</em></div>" if alt_ans else ""}
          </div>

          <div class='grid-2'>
            <div class='context-box'>
              <div class='section-title'>Pre-Gap Boundary Context (Unmasked)</div>
              {pre_lines}
            </div>
            <div class='context-box'>
              <div class='section-title'>Post-Gap Boundary Context (Unmasked)</div>
              {post_lines}
            </div>
          </div>

          <div style='margin-top: 14px;' class='grid-2'>
            <div class='context-box' style='background: #f0fdf4;'>
              <div class='section-title' style='color: #166534;'>Condition: Reconstructed (LLaMA 3.1 8B)</div>
              {recon_lines}
            </div>
            <div class='context-box'>
              <div class='section-title'>Condition: Baseline Repeat (Boundary Frame)</div>
              {baseline_lines}
            </div>
          </div>

          <details style='margin-top: 10px; font-size: 13px;'>
            <summary style='cursor: pointer; color: #475569; font-weight: 600;'>Show Original Oracle Captions (Ground Truth)</summary>
            <div class='context-box' style='margin-top: 6px; background: #fafafa;'>
              {oracle_lines}
            </div>
          </details>

          {candidates_html}

          <div class='eval-form'>
            <strong>Manual / Auditor Rating:</strong><br>
            <span style='margin-right: 12px;'>Factual Recovery:</span>
            <label><input type='radio' name='r_{idx}'> Exact Fact</label>
            <label><input type='radio' name='r_{idx}'> Superclass/Type</label>
            <label><input type='radio' name='r_{idx}'> Plausible Hallucination</label>
            <label><input type='radio' name='r_{idx}'> Missed</label>
            <span style='margin-left: 20px; margin-right: 12px;'>Boundary Leakage:</span>
            <label><input type='checkbox' name='leak_{idx}'> Answer was in Pre/Post</label>
          </div>
        </div>
        """
        html_parts.append(card_html)

    html_parts.extend(["</div>", "</body>", "</html>"])
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(html_parts))
    logger.info(f"Saved interactive HTML inspection dashboard to {output_path}")


def export_markdown_summary(items: list[dict[str, Any]], output_path: Path):
    """Generates a clean GitHub-flavored markdown report summarizing the samples."""
    lines = [
        "# WildQA Temporal Reconstruction - QA Inspection Summary\n",
        f"Total Inspected Samples: **{len(items)}** | Split: **{items[0]['split'] if items else 'N/A'}**\n",
        "## Summary of First Samples\n",
    ]

    for idx, it in enumerate(items[:20], start=1):
        lines.append(f"### Sample #{idx}: {it['question']}\n")
        lines.append(f"- **Video ID**: `{it['video_id']}` | **Domain**: {it['domain']} | **Span**: `{it['evidence_span']}s`")
        lines.append(f"- **Ground-Truth Answer**: **{it['ground_truth_answer']}**")
        if it["boundary_context"]["leakage_suspected"]:
            lines.append("- ⚠️ *Lexical hint: Key answer tokens detected in boundary frames.*")

        lines.append("\n**Gap In-Fill Comparison:**")
        lines.append("| Condition | First Caption in Gap |")
        lines.append("|---|---|")
        lines.append(f"| **Oracle** | {it['gap_conditions']['oracle'][0]['caption'] if it['gap_conditions']['oracle'] else 'N/A'} |")
        lines.append(f"| **Baseline Repeat** | {it['gap_conditions']['baseline_repeat'][0]['caption'] if it['gap_conditions']['baseline_repeat'] else 'N/A'} |")
        recon_c = it["gap_conditions"]["reconstructed"]
        recon_txt = recon_c[0]["caption"] if recon_c else "*Unavailable*"
        lines.append(f"| **Reconstructed (LLM)** | {recon_txt} |")
        lines.append("\n---\n")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    logger.info(f"Saved Markdown summary report to {output_path}")


def run_pipeline():
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    splits = ["dev"] if args.split == "dev" else (["test"] if args.split == "test" else ["dev", "test"])
    embedder = get_embedder_instance(args.embedder)

    for split in splits:
        json_path = PROJECT_ROOT / f"datasets/wildQA/{split}.json"
        cap_dir = PROJECT_ROOT / f"datasets/wildQA/captions__wild{4 if split == 'dev' else 5}"

        logger.info(f"Loading {split} QA pairs from {json_path}...")
        qa_pairs = load_wildqa_dataset(
            json_path,
            filter_scene_only=True,
            max_duration=60.0,
            single_evidence_only=args.clean_only,
            max_evidence_duration=15.0 if args.clean_only else None,
        )

        if args.max_items:
            qa_pairs = qa_pairs[:args.max_items]

        logger.info(f"Processing {len(qa_pairs)} questions on {split}...")

        recon_source = resolve_recon_source(split, args.recon_source)
        logger.info(f"Using recon source: {recon_source}")

        cached_caption_embs: dict[str, np.ndarray] = {}
        cached_gt_captions: dict[str, list[str]] = {}
        cached_recon_captions: dict[str, dict[int, str] | None] = {}

        inspection_items: list[dict[str, Any]] = []

        for idx, qa in enumerate(qa_pairs, start=1):
            vid = qa.video_id
            target_indices = qa.evidence_indices(max_duration=60)
            if not target_indices:
                continue

            # Load 60s dense captions
            if vid not in cached_gt_captions:
                caps = load_video_captions(cap_dir, vid)
                if caps is None:
                    continue
                cached_gt_captions[vid] = caps
            original_captions = cached_gt_captions[vid]

            # Load reconstructions
            if vid not in cached_recon_captions:
                cached_recon_captions[vid] = load_reconstructed_captions(recon_source, vid)
            recon_caps = cached_recon_captions[vid]

            item = build_inspection_item(
                qa=qa,
                split=split,
                original_captions=original_captions,
                recon_captions=recon_caps,
                embedder=embedder,
                boundary_window=args.boundary_window,
                top_k=args.top_k,
                cached_caption_embs=cached_caption_embs,
            )
            inspection_items.append(item)

            if idx % 20 == 0 or idx == len(qa_pairs):
                logger.info(f"[{split}] Processed {idx}/{len(qa_pairs)} items...")

        if not inspection_items:
            logger.warning(f"No inspection items generated for split {split}.")
            continue

        # Export outputs
        suffix = f"{split}_{args.embedder}"
        if args.format in ("all", "jsonl"):
            jsonl_file = out_dir / f"qa_inspection_{suffix}.jsonl"
            with open(jsonl_file, "w", encoding="utf-8") as f:
                for it in inspection_items:
                    f.write(json.dumps(it, ensure_ascii=False) + "\n")
            logger.info(f"Saved {len(inspection_items)} items to JSONL: {jsonl_file}")

        if args.format in ("all", "html"):
            html_file = out_dir / f"qa_inspection_{suffix}.html"
            export_html_report(inspection_items, html_file, args.embedder)

        if args.format in ("all", "markdown"):
            md_file = out_dir / f"qa_inspection_{suffix}.md"
            export_markdown_summary(inspection_items, md_file)

    logger.info("Done preparing QA inspection dataset.")


if __name__ == "__main__":
    run_pipeline()
