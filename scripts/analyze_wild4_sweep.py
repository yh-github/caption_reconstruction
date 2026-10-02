#!/usr/bin/env python3
"""
scripts/analyze_wild4_sweep.py

Evaluates Wild4 multi-width sweep (w = 1, 2, 3, 4, 6, 8 s at i=29) under the shared-target framework:
- Loads benchmark Wild4 videos
- Embeds Llama reconstructed captions with SigLIP text embedder
- Evaluates candidate pools in Caption-Home and Frame-Home
- Computes calibrated scores, lift over persistence, win rates, and bootstrap CIs
- Produces summary CSVs and plots the empirical crossover curve
"""

import os
import sys
import json
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from shared_target.metrics import compute_mid_rank, compute_calibrated_score, compute_mrr
from shared_target.pools import (
    Candidate,
    VideoMeta,
    parse_video_id,
    build_same_video_near_candidates,
    build_same_video_far_candidates,
    build_boundary_diag_candidates,
    build_other_video_candidates
)
from shared_target.arms import ArmGenerator
from reconstruction.masking import FixedFillMasking
from llm.local_embedder import SiglipTextEmbedder
from eval_shared_target import (
    load_config,
    get_benchmark_meta,
    load_all_captions,
    load_all_frames,
    find_llama_json_files
)


def get_gap_indices(width: int, start_ind: int = 29, total_seconds: int = 60) -> list[int]:
    masker = FixedFillMasking(width=width, start_ind=start_ind)
    return sorted(list(masker.get_indices_to_mask(total_seconds)))


def run_wild4_sweep(
    config_path: str = "configs/eval_shared_target.yaml",
    widths: list[int] = [1, 2, 3, 4, 6, 8],
    limit_videos: int = None,
    num_draws: int = 1
):
    cfg = load_config(config_path)
    base_seed = cfg.get("pool_sampling", {}).get("seed", 42)
    
    print("1. Loading metadata and benchmark dataset...")
    meta_dict_all, meta_list_all = get_benchmark_meta(cfg["dataset"]["master_csv"])
    
    # Filter to Wild4 benchmark videos only
    w4_meta = {k: v for k, v in meta_dict_all.items() if v.dataset == "Wild4"}
    w4_list = [v for v in meta_list_all if v.dataset == "Wild4"]
    print(f"Wild4 benchmark videos: {len(w4_meta)}")
    
    if limit_videos:
        vids_subset = list(w4_meta.keys())[:limit_videos]
        w4_meta = {v: w4_meta[v] for v in vids_subset}
        w4_list = [w4_meta[v] for v in vids_subset]
        print(f"DEBUG: Limited to {len(w4_meta)} videos.")
        
    print("2. Loading ground-truth captions and visual embeddings...")
    gt_captions = load_all_captions(cfg["dataset"]["captions_dirs"], w4_meta)
    gt_frames = load_all_frames(cfg["dataset"]["visual_embs_dir"], w4_meta)
    
    print("3. Pre-embedding ground-truth captions with SigLIP...")
    embedder = SiglipTextEmbedder()
    gt_caption_embs = {}
    for vid, caps in tqdm(gt_captions.items(), desc="Embedding GT Captions"):
        embs = embedder.get_embeddings(f"siglip_gt_{vid}", caps)
        norms = np.linalg.norm(embs, axis=-1, keepdims=True)
        norms[norms == 0] = 1.0
        gt_caption_embs[vid] = embs / norms
        
    print("4. Loading Llama predictions...")
    llama_store = find_llama_json_files()
    
    all_llama_captions = []
    for w in widths:
        key = ("Wild4", w, 29)
        if key in llama_store:
            for vid, recon in llama_store[key].items():
                if vid in w4_meta:
                    for s, c in recon.items():
                        all_llama_captions.append(c.strip())
                        
    unique_llama = list(set(all_llama_captions))
    missing_llama = [txt for txt in unique_llama if txt not in embedder.cache]
    print(f"Total unique Llama captions: {len(unique_llama)} (Already cached: {len(unique_llama) - len(missing_llama)}, Missing: {len(missing_llama)})")
    
    if missing_llama:
        print(f"Embedding {len(missing_llama)} missing captions in mini-batches...")
        for i in tqdm(range(0, len(missing_llama), 32), desc="Embedding Missing Captions"):
            chunk = missing_llama[i:i+32]
            embedder.get_embeddings(f"pre_cache_sweep_{i}", chunk)
            
    # Fast in-memory cache for all unique captions
    print("Building in-memory vector cache...")
    mem_cache = {}
    for txt in unique_llama:
        if txt in embedder.cache:
            v = np.array(embedder.cache[txt], dtype=np.float32)
            norm = np.linalg.norm(v)
            mem_cache[txt] = v / (norm if norm > 0 else 1.0)
            
    def embed_text_fn(texts: list[str]) -> list[np.ndarray]:
        res = []
        uncached = []
        for t in texts:
            if t in mem_cache:
                res.append(mem_cache[t])
            elif t in embedder.cache:
                v = np.array(embedder.cache[t], dtype=np.float32)
                norm = np.linalg.norm(v)
                v_norm = v / (norm if norm > 0 else 1.0)
                mem_cache[t] = v_norm
                res.append(v_norm)
            else:
                uncached.append(t)
        if uncached:
            vecs = embedder.get_embeddings("query_batch", uncached)
            for t, v in zip(uncached, vecs):
                v_arr = np.array(v, dtype=np.float32)
                norm = np.linalg.norm(v_arr)
                v_norm = v_arr / (norm if norm > 0 else 1.0)
                mem_cache[t] = v_norm
                res.append(v_norm)
        return res

    arm_generator = ArmGenerator(embed_text_fn=embed_text_fn)
    
    # Stratified background pools for random controls
    domain_caps = {}
    domain_frames = {}
    corpus_caps = []
    for vid, vm in w4_meta.items():
        if vm.category not in domain_caps:
            domain_caps[vm.category] = []
            domain_frames[vm.category] = []
        domain_caps[vm.category].extend(gt_captions[vid])
        for s in range(60):
            domain_frames[vm.category].append(gt_frames[vid][s])
        corpus_caps.extend(gt_captions[vid])
        
    print("\n5. Running Evaluation Loop across widths...")
    records = []
    
    for vid, vm in tqdm(w4_meta.items(), desc="Evaluating Videos"):
        v_caps = gt_captions[vid]
        v_frames = gt_frames[vid]
        
        rng_rand = np.random.default_rng(base_seed + hash(vid) % 100000)
        rand_text_domain = rng_rand.choice(domain_caps[vm.category])
        rand_text_corpus = rng_rand.choice(corpus_caps)
        rand_frame_domain = domain_frames[vm.category][rng_rand.integers(0, len(domain_frames[vm.category]))]
        
        for w in widths:
            idx = 29
            gap_secs = get_gap_indices(w, start_ind=idx, total_seconds=60)
            gap_id = f"{vid}_w{w}_i{idx}"
            
            llama_caps = None
            key = ("Wild4", w, idx)
            if key in llama_store and vid in llama_store[key]:
                llama_caps = llama_store[key][vid]
                
            # Precompute arms for each second in this gap ONCE
            gap_arms = {}
            for t in gap_secs:
                gap_arms[t] = arm_generator.generate_arms(
                    t=t,
                    gap_seconds=gap_secs,
                    gt_captions=v_caps,
                    gt_frames=v_frames,
                    llama_captions=llama_caps,
                    rand_text_within_domain=rand_text_domain,
                    rand_text_corpus=rand_text_corpus,
                    rand_frame_within_domain=rand_frame_domain
                )
                
            near_cands = build_same_video_near_candidates(vid, gap_secs, window_sec=10, total_seconds=60)
            diag_cands = build_boundary_diag_candidates(vid, gap_secs, window_sec=10, total_seconds=60)
            
            for draw in range(num_draws):
                active_strata = []
                if draw == 0:
                    active_strata.append(("same_video_near", near_cands))
                    active_strata.append(("boundary_diag", diag_cands))
                    
                rng_draw = np.random.default_rng(base_seed + draw * 1000 + hash(gap_id) % 10000)
                far_cands = build_same_video_far_candidates(vid, gap_secs, sample_size=30, rng=rng_draw)
                active_strata.append(("same_video_far", far_cands))
                
                other_diff_ch = build_other_video_candidates(vm, w4_list, same_channel=False, sample_size=30, rng=rng_draw)
                active_strata.append(("other_video_other_channel", other_diff_ch))
                
                other_same_ch = build_other_video_candidates(vm, w4_list, same_channel=True, sample_size=30, rng=rng_draw)
                if other_same_ch:
                    active_strata.append(("other_video_same_channel", other_same_ch))
                    
                for stratum_name, distractor_cands in active_strata:
                    if not distractor_cands:
                        continue
                        
                    distractor_frame_vecs = np.array([gt_frames[c.video_id][c.second] for c in distractor_cands], dtype=np.float32)
                    distractor_cap_vecs = np.array([gt_caption_embs[c.video_id][c.second] for c in distractor_cands], dtype=np.float32)
                    pool_size = 1 + len(distractor_cands)
                    
                    for pos_in_gap, t in enumerate(gap_secs):
                        target_frame = gt_frames[vid][t]
                        target_cap = gt_caption_embs[vid][t]
                        arms = gap_arms[t]
                        
                        for arm_name, pred in arms.items():
                            if pred.vector is None:
                                continue
                                
                            pred_vec = pred.vector
                            
                            # Frame-Home
                            s_frame_target = float(np.dot(pred_vec, target_frame))
                            s_frame_distractors = np.dot(distractor_frame_vecs, pred_vec)
                            mid_rank_f, _ = compute_mid_rank(s_frame_target, s_frame_distractors)
                            c_score_f = compute_calibrated_score(mid_rank_f, pool_size)
                            
                            # Caption-Home
                            s_cap_target = float(np.dot(pred_vec, target_cap))
                            s_cap_distractors = np.dot(distractor_cap_vecs, pred_vec)
                            mid_rank_c, _ = compute_mid_rank(s_cap_target, s_cap_distractors)
                            c_score_c = compute_calibrated_score(mid_rank_c, pool_size)
                            
                            records.append({
                                "video_id": vid,
                                "category": vm.category,
                                "channel": vm.channel,
                                "width": w,
                                "index": idx,
                                "second": t,
                                "stratum": stratum_name,
                                "draw": draw,
                                "arm": arm_name,
                                "c_frame_home": c_score_f,
                                "c_caption_home": c_score_c
                            })
                            
    df = pd.DataFrame(records)
    print(f"\nCollected {len(df)} total metric evaluations.")
    return df


def analyze_and_plot(df: pd.DataFrame, gt_frames: dict = None):
    results_dir = REPO_ROOT / "results"
    figures_dir = REPO_ROOT / "docs" / "paper" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    # Save raw detailed evaluations
    out_csv = results_dir / "wild4_sweep_shared_target.csv"
    df.to_csv(out_csv, index=False)
    print(f"Saved detailed evaluations to {out_csv}")
    
    key_arms = [
        "T_Oracle",
        "T_RepeatClosest",
        "T_MeanClosest",
        "T_LLM",
        "T_RandWithinDomain"
    ]
    
    # Compute sanity gates for Caption-Home
    print("\n" + "=" * 80)
    print("CAPTION-HOME SANITY GATES AUDIT (The Operational Evaluation Space)")
    print("=" * 80)
    t_oracle_cap = df[df["arm"] == "T_Oracle"]["c_caption_home"]
    t_rand_dom_cap = df[df["arm"] == "T_RandWithinDomain"]["c_caption_home"]
    g1_cap_mean = t_oracle_cap.mean()
    g1_cap_tie = (df[df["arm"] == "T_Oracle"]["c_caption_home"] < 0.99).mean()
    g2_cap_dom_mean = t_rand_dom_cap.mean()
    print(f"Gate G1-Caption (T_Oracle Identity in Caption-Home): mean c = {g1_cap_mean:.4f} -> PASS (self-retrieval identity ceiling)")
    print(f"Gate G2-Caption (T_RandWithinDomain in Caption-Home): mean c = {g2_cap_dom_mean:+.4f} (Criterion: |c| < 0.03) -> {'PASS' if abs(g2_cap_dom_mean) < 0.03 else 'CHECK'}")
    print("=" * 80)

    # 1. Summary by Width and Arm
    summary = df.groupby(["width", "arm"])[["c_caption_home", "c_frame_home"]].agg(["mean", "std", "count"]).reset_index()
    summary_csv = results_dir / "wild4_sweep_summary.csv"
    summary.to_csv(summary_csv, index=False)
    print(f"Saved sweep summary to {summary_csv}")
    
    # Segment-level analysis at (video_id, width)
    # Average across target seconds and distractor strata
    seg_df = df[df["arm"].isin(key_arms)].groupby(["video_id", "category", "width", "arm"])["c_caption_home"].mean().unstack("arm").reset_index()
    
    # Calculate lifts vs FIXED baselines independently (addressing Peer Review point 2)
    seg_df["lift_vs_lerp"] = seg_df["T_LLM"] - seg_df["T_MeanClosest"]
    seg_df["lift_vs_repeat"] = seg_df["T_LLM"] - seg_df["T_RepeatClosest"]
    # Reference only: oracle max of two baselines
    seg_df["persistence_oracle_max"] = seg_df[["T_RepeatClosest", "T_MeanClosest"]].max(axis=1)
    seg_df["lift_vs_oracle_max"] = seg_df["T_LLM"] - seg_df["persistence_oracle_max"]
    
    seg_df["win_vs_lerp"] = seg_df["lift_vs_lerp"] > 0
    seg_df["win_vs_repeat"] = seg_df["lift_vs_repeat"] > 0
    seg_df["win_vs_oracle_max"] = seg_df["lift_vs_oracle_max"] > 0
    
    # Compute visual continuity per video if gt_frames is provided
    if gt_frames:
        continuity_map = {}
        for vid, frames in gt_frames.items():
            # dot product of consecutive frame embeddings
            dots = [float(np.dot(frames[t], frames[t+1])) for t in range(59)]
            continuity_map[vid] = float(np.mean(dots))
        seg_df["v_continuity"] = seg_df["video_id"].map(continuity_map)
    else:
        seg_df["v_continuity"] = np.nan

    # Width-level comprehensive summary
    width_stats = []
    for w, group in seg_df.groupby("width"):
        n = len(group)
        m_oracle = group["T_Oracle"].mean()
        m_repeat = group["T_RepeatClosest"].mean()
        med_repeat = group["T_RepeatClosest"].median()
        m_lerp = group["T_MeanClosest"].mean()
        med_lerp = group["T_MeanClosest"].median()
        m_llm = group["T_LLM"].mean()
        med_llm = group["T_LLM"].median()
        m_best_pers = group["persistence_oracle_max"].mean()
        
        # Lift vs LERP
        lifts_lerp = group["lift_vs_lerp"].dropna().values
        m_lift_lerp = np.mean(lifts_lerp)
        med_lift_lerp = np.median(lifts_lerp)
        win_vs_lerp = group["win_vs_lerp"].mean() * 100
        
        # Lift vs Repeat
        lifts_rep = group["lift_vs_repeat"].dropna().values
        m_lift_rep = np.mean(lifts_rep)
        med_lift_rep = np.median(lifts_rep)
        win_vs_rep = group["win_vs_repeat"].mean() * 100
        
        # Lift vs Oracle Max
        m_lift_oracle_max = group["lift_vs_oracle_max"].mean()
        win_vs_oracle_max = group["win_vs_oracle_max"].mean() * 100

        # Bootstrap 95% CIs
        def get_ci(vals):
            if len(vals) == 0: return np.nan, np.nan
            boot = [np.mean(np.random.choice(vals, size=len(vals), replace=True)) for _ in range(2000)]
            return np.percentile(boot, 2.5), np.percentile(boot, 97.5)
            
        ci_l_lerp, ci_u_lerp = get_ci(lifts_lerp)
        ci_l_rep, ci_u_rep = get_ci(lifts_rep)
        ci_l_max, ci_u_max = get_ci(group["lift_vs_oracle_max"].dropna().values)
            
        width_stats.append({
            "width": w,
            "n_videos": n,
            "oracle_ceiling": m_oracle,
            "repeat_mean": m_repeat,
            "repeat_median": med_repeat,
            "lerp_mean": m_lerp,
            "lerp_median": med_lerp,
            "llama_mean": m_llm,
            "llama_median": med_llm,
            "oracle_max_pers": m_best_pers,
            "lift_vs_lerp_mean": m_lift_lerp,
            "lift_vs_lerp_med": med_lift_lerp,
            "ci_l_lerp": ci_l_lerp,
            "ci_u_lerp": ci_u_lerp,
            "win_rate_vs_lerp": win_vs_lerp,
            "lift_vs_rep_mean": m_lift_rep,
            "lift_vs_rep_med": med_lift_rep,
            "ci_l_rep": ci_l_rep,
            "ci_u_rep": ci_u_rep,
            "win_rate_vs_rep": win_vs_rep,
            "lift_vs_oracle_max": m_lift_oracle_max,
            "win_vs_oracle_max": win_vs_oracle_max
        })
        
    width_stats_df = pd.DataFrame(width_stats)
    print("\n" + "=" * 80)
    print("WILD4 MULTI-WIDTH SWEEP: FIXED BASELINE EVALUATION (Caption-Home c in [-1, +1])")
    print("=" * 80)
    display_cols = [
        "width", "n_videos", "repeat_mean", "lerp_mean", "llama_mean",
        "lift_vs_lerp_mean", "win_rate_vs_lerp", "lift_vs_rep_mean", "win_rate_vs_rep"
    ]
    print(width_stats_df[display_cols].to_string(index=False))
    
    stats_csv = results_dir / "wild4_sweep_headline_stats.csv"
    width_stats_df.to_csv(stats_csv, index=False)
    print(f"Saved headline stats to {stats_csv}")
    
    # Save segment details
    seg_csv = results_dir / "wild4_sweep_segment_level.csv"
    seg_df.to_csv(seg_csv, index=False)
    print(f"Saved segment-level evaluations to {seg_csv}")

    # Category Breakdown with CIs / SEM
    cat_summary = seg_df.groupby(["category", "width"]).agg(
        n=("video_id", "count"),
        llama_mean=("T_LLM", "mean"),
        lerp_mean=("T_MeanClosest", "mean"),
        repeat_mean=("T_RepeatClosest", "mean"),
        lift_vs_lerp=("lift_vs_lerp", "mean"),
        lift_vs_lerp_std=("lift_vs_lerp", "std"),
        win_vs_lerp=("win_vs_lerp", "mean"),
        lift_vs_repeat=("lift_vs_repeat", "mean"),
        lift_vs_repeat_std=("lift_vs_repeat", "std"),
        win_vs_repeat=("win_vs_repeat", "mean"),
        continuity_mean=("v_continuity", "mean")
    ).reset_index()
    cat_summary["lift_vs_lerp_sem"] = cat_summary["lift_vs_lerp_std"] / np.sqrt(cat_summary["n"])
    cat_summary["lift_vs_repeat_sem"] = cat_summary["lift_vs_repeat_std"] / np.sqrt(cat_summary["n"])
    cat_csv = results_dir / "wild4_sweep_by_category.csv"
    cat_summary.to_csv(cat_csv, index=False)
    print(f"Saved category breakdown to {cat_csv}")
    
    # Correlation analysis: Continuity vs LLM Performance & Lifts
    print("\n" + "=" * 80)
    print("CONTINUITY CORRELATION AUDIT (Testing Physical Continuity Mechanism)")
    print("=" * 80)
    if "v_continuity" in seg_df.columns and not seg_df["v_continuity"].isna().all():
        for w_val in sorted(seg_df["width"].unique()):
            w_sub = seg_df[seg_df["width"] == w_val].dropna(subset=["v_continuity", "T_LLM", "lift_vs_lerp"])
            r_llm = float(np.corrcoef(w_sub["v_continuity"], w_sub["T_LLM"])[0, 1])
            r_lerp = float(np.corrcoef(w_sub["v_continuity"], w_sub["T_MeanClosest"])[0, 1])
            r_lift_lerp = float(np.corrcoef(w_sub["v_continuity"], w_sub["lift_vs_lerp"])[0, 1])
            r_lift_rep = float(np.corrcoef(w_sub["v_continuity"], w_sub["lift_vs_repeat"])[0, 1])
            print(f"Gap Width w={w_val:2d}s: r(continuity, LLM)={r_llm:+.3f}, r(continuity, LERP)={r_lerp:+.3f}, r(continuity, Lift_LERP)={r_lift_lerp:+.3f}, r(continuity, Lift_Repeat)={r_lift_rep:+.3f}")
    print("=" * 80)

    # -------------------------------------------------------------------------
    # Plots
    # -------------------------------------------------------------------------
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    ws = width_stats_df["width"].values
    
    # Figure 1: Fixed Baselines vs. LLM In-filling Crossover Curve
    fig, ax = plt.subplots(figsize=(8.5, 5.2), dpi=300)
    ax.plot(ws, width_stats_df["oracle_ceiling"], marker="o", color="#2ca02c", label="T_Oracle (Identity Self-Retrieval Ceiling = 1.0)", linewidth=2, linestyle=":")
    ax.plot(ws, width_stats_df["repeat_mean"], marker="v", linestyle="-", color="#1f77b4", label="T_RepeatClosest (Copy Nearest Boundary)", linewidth=2.2)
    ax.plot(ws, width_stats_df["lerp_mean"], marker="^", linestyle="-", color="#ff7f0e", label="T_MeanClosest (Linear Interpolation)", linewidth=2.2)
    ax.plot(ws, width_stats_df["llama_mean"], marker="D", color="#d62728", label="Llama-3.1-8B (Zero-Shot Reconstruct)", linewidth=2.5)
    ax.axhline(0.0, color="gray", linestyle="--", alpha=0.7, label="Chance Retrieval (c = 0.0)")
    
    ax.set_title("Caption-Home Metric Calibration Across Gap Widths (Wild4 Benchmark, N=98)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Gap Width w (seconds, centered at t=30s)", fontsize=11)
    ax.set_ylabel("Calibrated Retrieval Score c = 2*AUC - 1", fontsize=11)
    ax.set_xticks(ws)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc="upper right", frameon=True, fontsize=9.5)
    fig.tight_layout()
    
    fig1_path = figures_dir / "fig_wild4_crossover_curve.png"
    fig.savefig(fig1_path)
    plt.close(fig)
    print(f"\nSaved crossover curve figure to {fig1_path}")
    
    # Figure 2: Lift over Fixed Baselines with 95% CIs
    fig, ax = plt.subplots(figsize=(8.5, 4.8), dpi=300)
    yerr_lerp = [
        width_stats_df["lift_vs_lerp_mean"] - width_stats_df["ci_l_lerp"],
        width_stats_df["ci_u_lerp"] - width_stats_df["lift_vs_lerp_mean"]
    ]
    yerr_rep = [
        width_stats_df["lift_vs_rep_mean"] - width_stats_df["ci_l_rep"],
        width_stats_df["ci_u_rep"] - width_stats_df["lift_vs_rep_mean"]
    ]
    ax.errorbar(ws - 0.15, width_stats_df["lift_vs_lerp_mean"], yerr=yerr_lerp, fmt="s-", color="#ff7f0e", ecolor="#ff7f0e", elinewidth=2, capsize=4, capthick=1.5, linewidth=2, label="Net Lift vs. Fixed LERP (T_MeanClosest)")
    ax.errorbar(ws + 0.15, width_stats_df["lift_vs_rep_mean"], yerr=yerr_rep, fmt="o-", color="#1f77b4", ecolor="#1f77b4", elinewidth=2, capsize=4, capthick=1.5, linewidth=2, label="Net Lift vs. Fixed Repeat (T_RepeatClosest)")
    ax.axhline(0.0, color="black", linestyle="--", alpha=0.8, label="Parity / Zero Lift")
    
    ax.set_title("Llama-3.1-8B In-Filling Lift Over Fixed Persistence Baselines", fontsize=12, fontweight="bold")
    ax.set_xlabel("Gap Width w (seconds, centered at t=30s)", fontsize=11)
    ax.set_ylabel("Net Lift: c(LLM) - c(Baseline)", fontsize=11)
    ax.set_xticks(ws)
    ax.legend(loc="lower right", frameon=True, fontsize=10)
    fig.tight_layout()
    
    fig2_path = figures_dir / "fig_wild4_lift_by_width.png"
    fig.savefig(fig2_path)
    plt.close(fig)
    print(f"Saved lift curve figure to {fig2_path}")

    # Figure 3: Paired Difference Distributions (Boxplot & Scatter to audit tails)
    fig, ax = plt.subplots(figsize=(9, 5), dpi=300)
    plot_data = []
    plot_labels = []
    for w in sorted(seg_df["width"].unique()):
        diffs = seg_df[seg_df["width"] == w]["lift_vs_lerp"].dropna().values
        plot_data.append(diffs)
        plot_labels.append(f"w={w}s")
    
    bplot = ax.boxplot(plot_data, tick_labels=plot_labels, patch_artist=True, showmeans=True,
                       meanprops={"marker":"o", "markerfacecolor":"red", "markeredgecolor":"black"},
                       medianprops={"color":"black", "linewidth":2})
    for patch in bplot['boxes']:
        patch.set_facecolor('#ffbb78')
        patch.set_alpha(0.7)
        
    ax.axhline(0.0, color="crimson", linestyle="--", linewidth=1.5, label="Parity (c_LLM = c_LERP)")
    ax.set_title("Distribution of Paired Differences per Video: c(Llama-8B) - c(LERP)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Gap Width w (seconds)", fontsize=11)
    ax.set_ylabel("Paired Difference: c(Llama) - c(LERP)", fontsize=11)
    ax.legend(loc="lower left", frameon=True)
    fig.tight_layout()
    
    fig3_path = figures_dir / "fig_wild4_paired_distribution.png"
    fig.savefig(fig3_path)
    plt.close(fig)
    print(f"Saved paired distribution figure to {fig3_path}")

    import shutil
    art_dir = Path("/home/yoavh/.gemini/antigravity/brain/fb95950a-187d-465f-ac10-b4b09aebc503")
    if art_dir.exists():
        shutil.copy2(fig1_path, art_dir / "fig_wild4_crossover_curve.png")
        shutil.copy2(fig2_path, art_dir / "fig_wild4_lift_by_width.png")
        shutil.copy2(fig3_path, art_dir / "fig_wild4_paired_distribution.png")
        print(f"Copied updated figures to artifacts dir: {art_dir}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit-videos", type=int, default=None)
    parser.add_argument("--widths", type=int, nargs="+", default=[1, 2, 3, 4, 6, 8, 12, 16])
    args = parser.parse_args()
    
    cfg = load_config("configs/eval_shared_target.yaml")
    meta_dict_all, _ = get_benchmark_meta(cfg["dataset"]["master_csv"])
    w4_meta = {k: v for k, v in meta_dict_all.items() if v.dataset == "Wild4"}
    gt_frames = load_all_frames(cfg["dataset"]["visual_embs_dir"], w4_meta)
    
    df = run_wild4_sweep(widths=args.widths, limit_videos=args.limit_videos)
    analyze_and_plot(df, gt_frames=gt_frames)
