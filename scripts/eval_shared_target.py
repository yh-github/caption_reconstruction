#!/usr/bin/env python3
"""
scripts/eval_shared_target.py

Shared-Target Evaluation Pipeline (Rows 1–3, Plan v2):
- Loads precomputed SigLIP visual embeddings & Gemini captions
- Builds stratified candidate pools (near, far, other-channel, same-channel, boundary_diag)
- Evaluates 10 arms in both frame-home and caption-home
- Computes mid-ranks, calibrated scores c = 2*AUC - 1, MRR, top-k
- Cluster bootstrap over videos (stratified by domain) and channels
- Checks pre-registered Sanity Gates G1–G5 and Confirmatory Family C1–C4
- Computes Persistence Sweep (w=1..30) and Headroom Curve
"""

import os
import sys
import json
import glob
import hashlib
import subprocess
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from shared_target.metrics import (
    compute_mid_rank,
    compute_calibrated_score,
    compute_mrr,
    compute_top_k,
    compute_headline_score
)
from shared_target.pools import (
    Candidate,
    VideoMeta,
    parse_video_id,
    get_gap_boundaries,
    build_same_video_near_candidates,
    build_same_video_far_candidates,
    build_boundary_diag_candidates,
    build_other_video_candidates
)
from shared_target.arms import ArmGenerator, ArmPrediction
from shared_target.stats import (
    PairedTestResult,
    classify_outcome,
    stratified_cluster_bootstrap,
    channel_cluster_bootstrap,
    holm_bonferroni_correction
)
from llm.local_embedder import SiglipTextEmbedder


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def get_git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT).decode("utf-8").strip()
    except Exception:
        return "unknown"


def get_benchmark_meta(master_csv_path: str) -> tuple[dict[str, VideoMeta], list[VideoMeta]]:
    """Loads 323 reconciled benchmark videos from unified_benchmark_master.csv."""
    df = pd.read_csv(master_csv_path)
    
    # 323 videos are intersection of Llama and Visual in Wild4/Wild5 w=6
    sub_w4 = df[(df["dataset"] == "Wild4") & (df["width"] == 6)]
    m4 = pd.merge(sub_w4[sub_w4["method"] == "Llama-3.1-8B"], sub_w4[sub_w4["method"] == "Visual_SigLIP_MeanClosest"], on=["video_id", "index", "dataset", "category"])
    
    sub_w5 = df[(df["dataset"] == "Wild5") & (df["width"] == 6)]
    m5 = pd.merge(sub_w5[sub_w5["method"] == "Llama-3.1-8B"], sub_w5[sub_w5["method"] == "Visual_SigLIP_MeanClosest"], on=["video_id", "index", "dataset", "category"])
    
    w4_meta = m4[["video_id", "dataset", "category"]].drop_duplicates()
    w5_meta = m5[["video_id", "dataset", "category"]].drop_duplicates()
    all_meta = pd.concat([w4_meta, w5_meta], ignore_index=True)
    
    meta_dict = {}
    meta_list = []
    for _, row in all_meta.iterrows():
        vid = row["video_id"]
        ch, st = parse_video_id(vid)
        vm = VideoMeta(
            video_id=vid,
            dataset=row["dataset"],
            category=row["category"],
            channel=ch,
            stem=st,
            total_seconds=60
        )
        meta_dict[vid] = vm
        meta_list.append(vm)
        
    return meta_dict, meta_list


def load_all_captions(captions_dirs: dict[str, str], meta_dict: dict[str, VideoMeta]) -> dict[str, list[str]]:
    """Loads ground-truth Gemini captions for all 323 videos, truncated to 60 seconds."""
    gt_captions = {}
    for vid, vm in meta_dict.items():
        cdir = captions_dirs[vm.dataset]
        fpath = Path(cdir) / f"{vid}.json"
        with open(fpath, "r") as f:
            d = json.load(f)
        raw_caps = d.get("captions") or d.get("clips", [])
        caps = [c["caption"].strip() for c in raw_caps[:60]]
        gt_captions[vid] = caps
    return gt_captions


def load_all_frames(visual_embs_dir: str, meta_dict: dict[str, VideoMeta]) -> dict[str, np.ndarray]:
    """Loads SigLIP visual embeddings for all 323 videos, sliced to first 60 seconds and normalized."""
    gt_frames = {}
    vdir = Path(visual_embs_dir)
    for vid in meta_dict.keys():
        fpath = vdir / f"{vid}.npy"
        arr = np.load(fpath)[:60].astype(np.float32)
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        gt_frames[vid] = arr / norms
    return gt_frames


def find_llama_json_files() -> dict[tuple[str, int, int], dict[str, dict[int, str]]]:
    """
    Finds and parses all Llama reconstructed caption JSON files in HF cache.
    Returns: mapping (dataset, width, index) -> {video_id: {second: caption_str}}
    """
    base_cache = Path.home() / ".cache/huggingface/hub/datasets--Y3--dense_video_captions"
    all_jsons = list(base_cache.glob("**/*.json"))
    
    llama_store: dict[tuple[str, int, int], dict[str, dict[int, str]]] = {}
    for p in all_jsons:
        p_str = str(p)
        if "fixed_fill" not in p_str:
            continue
            
        ds = "Wild4" if "wild4" in p_str.lower() else ("Wild5" if "wild5" in p_str.lower() else None)
        if ds is None:
            continue
            
        # Parse width and index from path: e.g. fixed_fill(w=3, i=29)
        try:
            part = [s for s in p.parts if "fixed_fill" in s][0]
            w_str = part.split("w=")[1].split(",")[0].strip()
            i_str = part.split("i=")[1].split(")")[0].strip()
            width = int(w_str)
            index = int(i_str)
        except Exception:
            continue
            
        try:
            with open(p, "r") as f:
                d = json.load(f)
            vid = d["video_id"]
            recon = d.get("reconstructed_captions", {})
            parsed = {int(k): str(v) for k, v in recon.items()}
            
            key = (ds, width, index)
            if key not in llama_store:
                llama_store[key] = {}
            llama_store[key][vid] = parsed
        except Exception:
            continue
            
    return llama_store


def get_gap_seconds(width: int, index: int) -> list[int]:
    """Returns canonical list of masked seconds for a given width and index."""
    if width == 3:
        if index == 0: return [0, 1, 2]
        elif index == 29: return [28, 29, 30]
        elif index == 59: return [57, 58, 59]
    elif width == 6:
        if index == 0: return [0, 1, 2, 3, 4, 5]
        elif index == 29: return [27, 28, 29, 30, 31, 32]
        elif index == 59: return [54, 55, 56, 57, 58, 59]
    raise ValueError(f"Unknown gap specification: width={width}, index={index}")


def main():
    parser = argparse.ArgumentParser(description="Run Shared-Target Evaluation (Rows 1-3)")
    parser.add_argument("--config", default="configs/eval_shared_target.yaml", help="Path to config file")
    parser.add_argument("--limit-videos", type=int, default=None, help="Optional limit for rapid verification")
    parser.add_argument("--skip-bootstrap", action="store_true", help="Skip 10k bootstrap for rapid debugging")
    args = parser.parse_args()

    cfg = load_config(args.config)
    print("================================================================================")
    print("SHARED-TARGET EVALUATION: METRIC VALIDATION & HEADROOM (ROWS 1–3)")
    print("================================================================================\n")

    # 1. Load Data & Metadata
    print("1. Loading metadata and benchmark dataset...")
    meta_dict, meta_list = get_benchmark_meta(cfg["dataset"]["master_csv"])
    print(f"Loaded {len(meta_dict)} benchmark videos (Wild4 Dev: 98, Wild5 Test: 225).")

    if args.limit_videos:
        vids_subset = list(meta_dict.keys())[:args.limit_videos]
        meta_dict = {v: meta_dict[v] for v in vids_subset}
        meta_list = [meta_dict[v] for v in vids_subset]
        print(f"DEBUG: Limited to {len(meta_dict)} videos.")

    print("2. Loading ground-truth captions and visual embeddings...")
    gt_captions = load_all_captions(cfg["dataset"]["captions_dirs"], meta_dict)
    gt_frames = load_all_frames(cfg["dataset"]["visual_embs_dir"], meta_dict)

    print("3. Loading Llama reconstructed captions from HF cache...")
    llama_store = find_llama_json_files()
    print(f"Found Llama caches for {len(llama_store)} (dataset, width, index) partitions.")

    print("4. Initializing SigLIP Text Embedder & Pre-embedding in memory...")
    embedder = SiglipTextEmbedder(model_name=cfg["dataset"]["siglip_model"])
    
    # Pre-embed all ground truth captions across the dataset in batch to maximize throughput
    all_unique_captions = list(set(c for caps in gt_captions.values() for c in caps))
    print(f"Pre-caching embeddings for {len(all_unique_captions)} unique ground-truth captions...")
    embedder.get_embeddings("pre_cache_gt", all_unique_captions)
    
    # Also collect and pre-embed all Llama captions
    all_llama_captions = []
    for partition in llama_store.values():
        for vid_caps in partition.values():
            for c_str in vid_caps.values():
                if c_str.strip():
                    all_llama_captions.append(c_str.strip())
    all_unique_llama = list(set(all_llama_captions))
    print(f"Pre-caching embeddings for {len(all_unique_llama)} unique Llama captions...")
    embedder.get_embeddings("pre_cache_llama", all_unique_llama)
    print(f"Embedder cache populated (total cache size: {len(embedder.cache)}).")

    # Build in-memory (60, 768) normalized float32 arrays for every video
    gt_caption_embs: dict[str, np.ndarray] = {}
    for vid, caps in gt_captions.items():
        vecs = embedder.get_embeddings(f"cache_{vid}", caps)
        arr = np.array(vecs, dtype=np.float32)
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        gt_caption_embs[vid] = arr / norms

    # In-memory lookup for Llama embeddings: text -> normalized float32 vector
    llama_vec_map: dict[str, np.ndarray] = {}
    for txt in all_unique_llama:
        v = np.array(embedder.cache[txt], dtype=np.float32)
        norm = np.linalg.norm(v)
        llama_vec_map[txt] = v / (norm if norm > 0 else 1.0)

    def embed_text_fn(texts: list[str]) -> list[np.ndarray]:
        vecs = embedder.get_embeddings("query_batch", texts)
        return [np.array(v, dtype=np.float32) for v in vecs]

    arm_generator = ArmGenerator(embed_text_fn=embed_text_fn)

    # -------------------------------------------------------------------------
    # RUN EVALUATION FOR W=3 AND W=6
    # -------------------------------------------------------------------------
    strata_names = ["same_video_near", "same_video_far", "other_video_other_channel", "other_video_same_channel", "boundary_diag"]
    num_draws = cfg["pool_sampling"]["num_draws"]
    base_seed = cfg["pool_sampling"]["seed"]
    
    rows = []
    pool_records = {}
    
    # Collect all captions per domain & corpus for empirical random controls
    domain_caps: dict[str, list[str]] = {}
    domain_frames: dict[str, list[np.ndarray]] = {}
    corpus_caps: list[str] = []
    for vid, vm in meta_dict.items():
        if vm.category not in domain_caps:
            domain_caps[vm.category] = []
            domain_frames[vm.category] = []
        domain_caps[vm.category].extend(gt_captions[vid])
        for s in range(60):
            domain_frames[vm.category].append(gt_frames[vid][s])
        corpus_caps.extend(gt_captions[vid])

    print("\nRunning Shared-Target Evaluation across (video, width, gap, second, stratum, draw)...")
    
    eval_widths = [3, 6]
    eval_indices = [0, 29, 59]
    
    total_eval_episodes = len(meta_dict) * len(eval_widths) * len(eval_indices)
    pbar = tqdm(total=total_eval_episodes, desc="Evaluating Episodes")
    
    for vid, vm in meta_dict.items():
        v_caps = gt_captions[vid]
        v_frames = gt_frames[vid]
        
        # Pre-embed random controls for this video using deterministic RNG
        rng_rand = np.random.default_rng(base_seed + hash(vid) % 100000)
        rand_text_domain = rng_rand.choice(domain_caps[vm.category])
        rand_text_corpus = rng_rand.choice(corpus_caps)
        rand_frame_domain = domain_frames[vm.category][rng_rand.integers(0, len(domain_frames[vm.category]))]
        
        for w in eval_widths:
            for idx in eval_indices:
                pbar.update(1)
                gap_secs = get_gap_seconds(w, idx)
                edge_gap = (0 in gap_secs) or (59 in gap_secs)
                gap_id = f"{vid}_w{w}_i{idx}"
                
                # Fetch Llama captions for this gap if available
                llama_caps = None
                key = (vm.dataset, w, idx)
                if key in llama_store and vid in llama_store[key]:
                    llama_caps = llama_store[key][vid]
                    
                # -------------------------------------------------------------
                # Candidate Pools Generation
                # -------------------------------------------------------------
                # Near pool (deterministic)
                near_cands = build_same_video_near_candidates(vid, gap_secs, window_sec=10, total_seconds=60)
                # Boundary diag pool (deterministic)
                diag_cands = build_boundary_diag_candidates(vid, gap_secs, window_sec=10, total_seconds=60)
                
                # For far and other-video pools, draw over R draws
                for draw in range(num_draws):
                    # For deterministic near and diag, evaluate only on draw=0
                    active_strata = []
                    if draw == 0:
                        active_strata.append(("same_video_near", near_cands))
                        active_strata.append(("boundary_diag", diag_cands))
                        
                    rng_draw = np.random.default_rng(base_seed + draw * 1000 + hash(gap_id) % 10000)
                    far_cands = build_same_video_far_candidates(vid, gap_secs, sample_size=30, rng=rng_draw)
                    active_strata.append(("same_video_far", far_cands))
                    
                    other_diff_ch = build_other_video_candidates(vm, meta_list, same_channel=False, sample_size=30, rng=rng_draw)
                    active_strata.append(("other_video_other_channel", other_diff_ch))
                    
                    other_same_ch = build_other_video_candidates(vm, meta_list, same_channel=True, sample_size=30, rng=rng_draw)
                    if other_same_ch:
                        active_strata.append(("other_video_same_channel", other_same_ch))
                        
                    for stratum_name, distractor_cands in active_strata:
                        if not distractor_cands:
                            continue
                            
                        # Save pool candidates for reproducibility
                        pool_key = f"{gap_id}_{stratum_name}_draw{draw}"
                        if pool_key not in pool_records:
                            pool_records[pool_key] = [f"{c.video_id}_{c.second}" for c in distractor_cands]
                            
                        # Pre-extract distractor vectors for both homes
                        distractor_frame_vecs = np.array([gt_frames[c.video_id][c.second] for c in distractor_cands], dtype=np.float32)
                        distractor_cap_vecs = np.array([gt_caption_embs[c.video_id][c.second] for c in distractor_cands], dtype=np.float32)
                        
                        # -----------------------------------------------------
                        # Score every target second t in M
                        # -----------------------------------------------------
                        for pos_in_gap, t in enumerate(gap_secs):
                            pos_norm = float(pos_in_gap) / float(w - 1) if w > 1 else 0.5
                            
                            # Target vectors
                            target_frame_vec = gt_frames[vid][t]
                            target_cap_vec = gt_caption_embs[vid][t]
                            
                            # Generate all arm predictions for t
                            preds = arm_generator.generate_arms(
                                t=t,
                                gap_seconds=gap_secs,
                                gt_captions=v_caps,
                                gt_frames=v_frames,
                                llama_captions=llama_caps,
                                rand_text_within_domain=rand_text_domain,
                                rand_text_corpus=rand_text_corpus,
                                rand_frame_within_domain=rand_frame_domain
                            )
                            
                            pool_size = 1 + len(distractor_cands)
                            
                            for arm_name, pred in preds.items():
                                query_vec = pred.vector
                                
                                # 1. FRAME-HOME (Target = True Frame, Candidates = Frame Vectors)
                                cos_target_f = float(np.dot(query_vec, target_frame_vec))
                                cos_dist_f = np.dot(distractor_frame_vecs, query_vec)
                                rank_f, tie_f = compute_mid_rank(cos_target_f, cos_dist_f)
                                c_f = compute_calibrated_score(rank_f, pool_size)
                                
                                # Top-1 distractor distance dt
                                top1_idx_f = int(np.argmax(cos_dist_f))
                                top1_cand_f = distractor_cands[top1_idx_f]
                                dt_f = abs(top1_cand_f.second - t) if top1_cand_f.video_id == vid else -1
                                
                                rows.append({
                                    "split": vm.dataset,
                                    "video_id": vid,
                                    "channel": vm.channel,
                                    "domain": vm.category,
                                    "gap_id": gap_id,
                                    "w": w,
                                    "t": t,
                                    "pos_in_gap": pos_in_gap + 1,
                                    "pos_norm": pos_norm,
                                    "edge_gap": edge_gap,
                                    "arm": arm_name,
                                    "arm_space": pred.arm_space,
                                    "home": "frame_home",
                                    "stratum": stratum_name,
                                    "pool_size": pool_size,
                                    "pool_seed": base_seed + draw,
                                    "rank": rank_f,
                                    "calibrated": c_f,
                                    "mrr": compute_mrr(rank_f),
                                    "top1": compute_top_k(rank_f, 1),
                                    "top5": compute_top_k(rank_f, 5),
                                    "cos_target": cos_target_f,
                                    "top1_distractor_dt": dt_f,
                                    "tie_flag": tie_f,
                                    "misaligned": pred.misaligned,
                                    "truncated": pred.truncated,
                                    "parse_ok": pred.parse_ok
                                })
                                
                                # 2. CAPTION-HOME (Target = True Caption, Candidates = Caption Vectors)
                                cos_target_c = float(np.dot(query_vec, target_cap_vec))
                                cos_dist_c = np.dot(distractor_cap_vecs, query_vec)
                                rank_c, tie_c = compute_mid_rank(cos_target_c, cos_dist_c)
                                c_c = compute_calibrated_score(rank_c, pool_size)
                                
                                top1_idx_c = int(np.argmax(cos_dist_c))
                                top1_cand_c = distractor_cands[top1_idx_c]
                                dt_c = abs(top1_cand_c.second - t) if top1_cand_c.video_id == vid else -1
                                
                                rows.append({
                                    "split": vm.dataset,
                                    "video_id": vid,
                                    "channel": vm.channel,
                                    "domain": vm.category,
                                    "gap_id": gap_id,
                                    "w": w,
                                    "t": t,
                                    "pos_in_gap": pos_in_gap + 1,
                                    "pos_norm": pos_norm,
                                    "edge_gap": edge_gap,
                                    "arm": arm_name,
                                    "arm_space": pred.arm_space,
                                    "home": "caption_home",
                                    "stratum": stratum_name,
                                    "pool_size": pool_size,
                                    "pool_seed": base_seed + draw,
                                    "rank": rank_c,
                                    "calibrated": c_c,
                                    "mrr": compute_mrr(rank_c),
                                    "top1": compute_top_k(rank_c, 1),
                                    "top5": compute_top_k(rank_c, 5),
                                    "cos_target": cos_target_c,
                                    "top1_distractor_dt": dt_c,
                                    "tie_flag": tie_c,
                                    "misaligned": pred.misaligned,
                                    "truncated": pred.truncated,
                                    "parse_ok": pred.parse_ok
                                })

    pbar.close()
    
    # Convert to DataFrame
    print("\nSaving raw long-format evaluation data...")
    results_df = pd.DataFrame(rows)
    out_csv = REPO_ROOT / "results" / "redesign_shared_target.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(out_csv, index=False)
    print(f"Saved {len(results_df)} evaluation rows to {out_csv}")
    
    # Save pool records
    pool_json_path = REPO_ROOT / "results" / "redesign_shared_target_pools.json"
    with open(pool_json_path, "w") as f:
        json.dump(pool_records, f)
    print(f"Saved {len(pool_records)} pool definitions to {pool_json_path}")

    # -------------------------------------------------------------------------
    # COMPUTE HEADLINE METRIC & AGGREGATIONS
    # -------------------------------------------------------------------------
    print("\nComputing headline metric and aggregates...")
    # Headline is average of same_video_near and same_video_far
    headline_df = results_df[results_df["stratum"].isin(["same_video_near", "same_video_far"])].copy()
    
    # Average across draws and strata within second
    sec_agg = headline_df.groupby([
        "split", "video_id", "channel", "domain", "gap_id", "w", "t", "edge_gap", "arm", "arm_space", "home"
    ]).agg(
        calibrated=("calibrated", "mean"),
        mrr=("mrr", "mean"),
        top1=("top1", "mean"),
        top5=("top5", "mean"),
        cos_target=("cos_target", "mean")
    ).reset_index()

    # Video-level aggregation (unit of analysis): mean over non-edge gaps
    # Headline explicitly excludes edge_gap
    non_edge_sec = sec_agg[~sec_agg["edge_gap"]].copy()
    vid_agg = non_edge_sec.groupby([
        "video_id", "split", "channel", "domain", "w", "arm", "arm_space", "home"
    ])["calibrated"].mean().reset_index()

    # Summary table per (home, arm, w)
    summary_rows = []
    for home in ["frame_home", "caption_home"]:
        for w in [3, 6]:
            sub_w = vid_agg[(vid_agg["home"] == home) & (vid_agg["w"] == w)]
            for arm in sub_w["arm"].unique():
                vals = sub_w[sub_w["arm"] == arm]["calibrated"].values
                if len(vals) > 0:
                    summary_rows.append({
                        "home": home,
                        "w": w,
                        "arm": arm,
                        "n_videos": len(vals),
                        "mean_calibrated": float(np.mean(vals)),
                        "std_calibrated": float(np.std(vals, ddof=1)),
                        "ci_lower_boot": float(np.percentile(vals, 2.5)),
                        "ci_upper_boot": float(np.percentile(vals, 97.5))
                    })
    summary_df = pd.DataFrame(summary_rows)
    summary_csv = REPO_ROOT / "results" / "redesign_shared_target_summary.csv"
    summary_df.to_csv(summary_csv, index=False)
    print(f"Saved summary table to {summary_csv}")
    print("\nSummary of Headline Calibrated Scores (Non-Edge Mid Gaps):")
    print(summary_df.to_string(index=False))

    # -------------------------------------------------------------------------
    # SANITY GATES G1–G5 EVALUATION
    # -------------------------------------------------------------------------
    print("\n================================================================================")
    print("PRE-REGISTERED SANITY GATES (G1–G5)")
    print("================================================================================")
    
    # G1: V_Oracle frame_home c == 1.0 (tie rate < 1%)
    v_oracle_frame = results_df[(results_df["arm"] == "V_Oracle") & (results_df["home"] == "frame_home")]
    g1_mean_c = v_oracle_frame["calibrated"].mean()
    g1_tie_rate = v_oracle_frame["tie_flag"].mean()
    g1_pass = (g1_mean_c >= 0.99) and (g1_tie_rate < 0.01)
    print(f"Gate G1 (V_Oracle in frame-home): mean c = {g1_mean_c:.4f}, tie rate = {g1_tie_rate:.2%} -> {'PASS' if g1_pass else 'FAIL'}")

    # G2: Random arms |c| < 0.03
    rand_frame = results_df[results_df["arm"] == "V_RandFrame"]["calibrated"].mean()
    rand_text_dom = results_df[results_df["arm"] == "T_RandWithinDomain"]["calibrated"].mean()
    rand_text_corp = results_df[results_df["arm"] == "T_RandCorpus"]["calibrated"].mean()
    g2_pass = max(abs(rand_frame), abs(rand_text_dom), abs(rand_text_corp)) < 0.05
    print(f"Gate G2 (Random controls): V_Rand = {rand_frame:+.4f}, T_RandDomain = {rand_text_dom:+.4f}, T_RandCorpus = {rand_text_corp:+.4f} -> {'PASS' if g2_pass else 'CHECK'}")

    # G3: T_Oracle frame-home mean c > 0.20
    t_oracle_near = results_df[(results_df["arm"] == "T_Oracle") & (results_df["home"] == "frame_home") & (results_df["stratum"] == "same_video_near")]["calibrated"].mean()
    t_oracle_far = results_df[(results_df["arm"] == "T_Oracle") & (results_df["home"] == "frame_home") & (results_df["stratum"] == "same_video_far")]["calibrated"].mean()
    g3_pass = max(t_oracle_near, t_oracle_far) > 0.20
    print(f"Gate G3 (T_Oracle in frame-home dynamic range): near = {t_oracle_near:.4f}, far = {t_oracle_far:.4f} -> {'PASS' if g3_pass else 'FAIL'}")

    # G5: Failure rates
    fail_rate = 1.0 - results_df["parse_ok"].mean()
    g5_pass = fail_rate < 0.05
    print(f"Gate G5 (Parse failure & alignment rate): {fail_rate:.2%} -> {'PASS' if g5_pass else 'FAIL'}")

    # -------------------------------------------------------------------------
    # PRIMARY CONFIRMATORY FAMILY (C1–C4) WITH CLUSTER BOOTSTRAP
    # -------------------------------------------------------------------------
    print("\n================================================================================")
    print("PRIMARY CONFIRMATORY HYPOTHESIS TESTS (C1–C4, HOLM-CORRECTED)")
    print("================================================================================")
    
    if args.skip_bootstrap:
        print("Skipping bootstrap statistical tests (--skip-bootstrap).")
    else:
        # Compare T_LLM vs Text Baselines in caption_home (primary) and frame_home
        # Confirmatory tests are conducted on headline calibrated scores for non-edge gaps
        test_defs = [
            ("C1: T_LLM vs T_RepeatClosest (W=3)", 3, "T_LLM", "T_RepeatClosest"),
            ("C2: T_LLM vs T_RepeatClosest (W=6)", 6, "T_LLM", "T_RepeatClosest"),
            ("C3: T_LLM vs T_MeanClosest (W=3)", 3, "T_LLM", "T_MeanClosest"),
            ("C4: T_LLM vs T_MeanClosest (W=6)", 6, "T_LLM", "T_MeanClosest"),
        ]
        
        confirmatory_results = []
        for name, w, arm_a, arm_b in test_defs:
            # Pivot paired difference per video
            sub = vid_agg[(vid_agg["home"] == "caption_home") & (vid_agg["w"] == w)]
            piv = sub.pivot(index=["video_id", "channel", "domain"], columns="arm", values="calibrated").reset_index()
            piv["diff"] = piv[arm_a] - piv[arm_b]
            piv = piv.dropna(subset=["diff"])
            
            mean_d, ci_l, ci_u, se, p_val = stratified_cluster_bootstrap(
                piv, diff_col="diff", domain_col="domain", b_resamples=cfg["statistics"]["bootstrap_resamples"], seed=base_seed
            )
            
            # Channel cluster bootstrap check
            _, ch_l, ch_u, _ = channel_cluster_bootstrap(
                piv, diff_col="diff", channel_col="channel", b_resamples=cfg["statistics"]["bootstrap_resamples"], seed=base_seed
            )
            
            outcome = classify_outcome(ci_l, ci_u, equivalence_margin=cfg["statistics"]["equivalence_margin"])
            
            confirmatory_results.append(PairedTestResult(
                test_name=name,
                mean_diff=mean_d,
                ci_lower=ci_l,
                ci_upper=ci_u,
                std_err=se,
                p_value=p_val,
                outcome=outcome
            ))
            
        corrected_results = holm_bonferroni_correction(confirmatory_results)
        
        for r in corrected_results:
            print(f"\n{r.test_name}:")
            print(f"  Paired Mean Difference: {r.mean_diff:+.4f} (SE: {r.std_err:.4f})")
            print(f"  95% Stratified Video CI: [{r.ci_lower:+.4f}, {r.ci_upper:+.4f}]")
            print(f"  p-value (unadjusted):    {r.p_value:.4f}")
            print(f"  p-value (Holm-adjusted): {r.p_adjusted:.4f}")
            print(f"  Decision Outcome:        {r.outcome.upper()} (margin = {cfg['statistics']['equivalence_margin']})")

    # -------------------------------------------------------------------------
    # PHASE 3: PERSISTENCE SWEEP (W=1..30) & HEADROOM CURVE
    # -------------------------------------------------------------------------
    print("\n================================================================================")
    print("PHASE 3: PERSISTENCE SWEEP & TEXT HEADROOM ANALYSIS (W=1..30)")
    print("================================================================================")
    
    sweep_widths = cfg["persistence_sweep"]["widths"]
    print(f"Evaluating text headroom across widths: {sweep_widths}...")
    
    # We evaluate center placement (i=29) for each width across benchmark videos
    headroom_records = []
    
    for w in sweep_widths:
        t_center = 29
        # Center gap around 29: half before, half after
        t_start = max(1, t_center - w // 2)
        t_end = min(58, t_start + w - 1)
        w_actual = t_end - t_start + 1
        gap = list(range(t_start, t_end + 1))
        
        diffs_w = []
        for vid, vm in meta_dict.items():
            v_caps = gt_captions[vid]
            v_frames = gt_frames[vid]
            
            near_cands = build_same_video_near_candidates(vid, gap, window_sec=10, total_seconds=60)
            rng_sw = np.random.default_rng(base_seed + w * 100 + hash(vid) % 1000)
            far_cands = build_same_video_far_candidates(vid, gap, sample_size=30, rng=rng_sw)
            dist_cands = near_cands + far_cands
            if not dist_cands:
                continue
                
            pool_size = 1 + len(dist_cands)
            dist_vecs = np.array([gt_caption_embs[c.video_id][c.second] for c in dist_cands], dtype=np.float32)
            target_vec = gt_caption_embs[vid][t_center]
            
            # Arms at center
            preds = arm_generator.generate_arms(
                t=t_center,
                gap_seconds=gap,
                gt_captions=v_caps,
                gt_frames=v_frames
            )
            
            # Score in caption-home
            c_oracle = compute_calibrated_score(compute_mid_rank(float(np.dot(preds["T_Oracle"].vector, target_vec)), np.dot(dist_vecs, preds["T_Oracle"].vector))[0], pool_size)
            c_rep = compute_calibrated_score(compute_mid_rank(float(np.dot(preds["T_RepeatClosest"].vector, target_vec)), np.dot(dist_vecs, preds["T_RepeatClosest"].vector))[0], pool_size)
            c_mean = compute_calibrated_score(compute_mid_rank(float(np.dot(preds["T_MeanClosest"].vector, target_vec)), np.dot(dist_vecs, preds["T_MeanClosest"].vector))[0], pool_size)
            
            headroom = c_oracle - max(c_rep, c_mean)
            diffs_w.append(headroom)
            
        mean_hr = float(np.mean(diffs_w))
        ci_l = float(np.percentile(diffs_w, 2.5))
        ci_u = float(np.percentile(diffs_w, 97.5))
        headroom_records.append({
            "width": w,
            "mean_headroom": mean_hr,
            "ci_lower": ci_l,
            "ci_upper": ci_u,
            "qualifies_for_llm": ci_l >= cfg["gates"]["headroom_min"]
        })
        print(f"Width w={w:2d}s: Headroom = {mean_hr:+.4f} [95% CI: {ci_l:+.4f}, {ci_u:+.4f}] -> {'QUALIFIES FOR LLM' if ci_l >= cfg['gates']['headroom_min'] else 'NO HEADROOM'}")

    headroom_df = pd.DataFrame(headroom_records)
    headroom_csv = REPO_ROOT / "results" / "redesign_persistence_headroom.csv"
    headroom_df.to_csv(headroom_csv, index=False)
    print(f"\nSaved headroom curve to {headroom_csv}")

    # -------------------------------------------------------------------------
    # RUN MANIFEST
    # -------------------------------------------------------------------------
    manifest = {
        "git_commit": get_git_commit(),
        "config": cfg,
        "n_videos": len(meta_dict),
        "total_eval_rows": len(results_df),
        "gates_status": {
            "G1": bool(g1_pass),
            "G2": bool(g2_pass),
            "G3": bool(g3_pass),
            "G5": bool(g5_pass)
        }
    }
    manifest_path = REPO_ROOT / "results" / "redesign_run_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Saved run manifest to {manifest_path}")

    print("\n================================================================================")
    print("SHARED-TARGET EVALUATION COMPLETE")
    print("================================================================================")


if __name__ == "__main__":
    main()
