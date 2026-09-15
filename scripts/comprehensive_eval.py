#!/usr/bin/env python3
"""
scripts/comprehensive_eval.py

Comprehensive evaluation of dense caption reconstruction across multiple baselines and metrics.
Runs purely on CPU using precomputed embeddings (diskcache) and local texts.
"""

import os
import json
import glob
import re
import argparse
from collections import Counter
import numpy as np
import pandas as pd
from scipy import stats
import diskcache
from huggingface_hub import HfApi, hf_hub_download

# Simple tokenization & content word extractor (nouns, verbs, adjectives)
def extract_content_words(text: str) -> set[str]:
    STOP_WORDS = {
        'the', 'and', 'a', 'an', 'in', 'on', 'at', 'to', 'for', 'of', 'with',
        'is', 'are', 'was', 'were', 'it', 'its', 'they', 'their', 'this', 'that',
        'from', 'as', 'by', 'into', 'over', 'after', 'before', 'between', 'out',
        'up', 'down', 'about', 'then', 'there', 'here', 'where', 'when', 'how',
        'what', 'which', 'who', 'whom', 'whose', 'all', 'both', 'each', 'few',
        'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only',
        'own', 'same', 'so', 'than', 'too', 'very', 'can', 'will', 'just', 'should'
    }
    words = re.findall(r'[a-z]+', text.lower())
    return {w for w in words if len(w) >= 3 and w not in STOP_WORDS}

def content_f1(pred_text: str, gt_text: str) -> float:
    pred_words = extract_content_words(pred_text)
    gt_words = extract_content_words(gt_text)
    if not pred_words or not gt_words:
        return 0.0
    common = pred_words.intersection(gt_words)
    if not common:
        return 0.0
    precision = len(common) / len(pred_words)
    recall = len(common) / len(gt_words)
    return 2.0 * precision * recall / (precision + recall)

def bootstrap_ci(arr: np.ndarray, num_resamples: int = 2000, alpha: float = 0.05) -> tuple[float, float]:
    rng = np.random.default_rng(42)
    n = len(arr)
    boot_means = [np.mean(arr[rng.integers(0, n, size=n)]) for _ in range(num_resamples)]
    low = np.percentile(boot_means, 100 * (alpha / 2))
    high = np.percentile(boot_means, 100 * (1 - alpha / 2))
    return float(low), float(high)

def main():
    parser = argparse.ArgumentParser(description="Comprehensive evaluation across baselines and metrics")
    parser.add_argument("--cache_dir", default="disk_cache/local_all-mpnet-base-v2", help="Diskcache path for embeddings")
    parser.add_argument("--gt_dir", default="datasets/wildQA/captions__wild4", help="Directory of ground truth caption jsons")
    parser.add_argument("--output_csv", default="results/comprehensive_eval.csv", help="Output summary CSV")
    parser.add_argument("--per_video_csv", default="results/comprehensive_eval_per_video.csv", help="Per-video detail CSV")
    args = parser.parse_args()

    print(f"Loading embedding cache from {args.cache_dir}...")
    cache = diskcache.Cache(args.cache_dir)
    print(f"Cache size: {len(cache)} items")

    # Load Ground Truth
    gt_dict = {}
    for gf in glob.glob(os.path.join(args.gt_dir, "*.json")):
        if os.path.basename(gf) == "categories.json":
            continue
        with open(gf) as fp:
            d = json.load(fp)
        clips = d.get("captions") or d.get("clips")
        gt_dict[d["video_id"]] = [c["caption"] for c in clips]
    print(f"Loaded {len(gt_dict)} ground truth videos")

    # Pool of all captions across corpus for random baseline
    all_caps = [c for caps in gt_dict.values() for c in caps]
    all_vecs = np.array([cache[c] for c in all_caps if c in cache])
    print(f"Corpus pool: {len(all_caps)} captions, {len(all_vecs)} cached vectors")

    # Load Llama results from HF dataset
    api = HfApi()
    files = api.list_repo_files(repo_id="Y3/dense_video_captions", repo_type="dataset")
    llama_files = [f for f in files if "wild4_llama_w3_window_v3" in f and f.endswith(".json")]
    print(f"Found {len(llama_files)} Llama JSON files on HF")

    # Initialize embedder to automatically fill any missing cache entries on CPU
    from llm.local_embedder import LocalEmbedder
    embedder = LocalEmbedder(model_name="all-mpnet-base-v2")

    positions = {
        "start": ("fixed_fill(w=3, i=0)", 3),       # gap at 0..2, boundary after = clip 3
        "mid":   ("fixed_fill(w=3, i=29)", 28),     # gap at 29..31, boundary before = clip 28
        "end":   ("fixed_fill(w=3, i=59)", 58),     # gap at 59..61 (capped), boundary before = clip 58
    }

    rng = np.random.default_rng(2026)
    records = []

    for pos_name, (pattern, boundary_idx) in positions.items():
        sub_files = [f for f in llama_files if pattern in f]
        print(f"\nProcessing {pos_name} position ({len(sub_files)} files)...")

        for f in sub_files:
            lp = hf_hub_download(repo_id="Y3/dense_video_captions", filename=f, repo_type="dataset")
            with open(lp) as fp:
                d = json.load(fp)
            vid = d["video_id"]
            if vid not in gt_dict:
                continue
            
            gt_caps = gt_dict[vid]
            recon_caps = d.get("reconstructed_captions", {})
            if not recon_caps:
                continue

            sorted_keys = sorted(recon_caps.keys(), key=lambda x: int(x))
            gap_indices = [int(k) for k in sorted_keys]
            m = len(gap_indices)
            if m == 0:
                continue

            # Ground truth gap captions
            gt_gap_caps = [gt_caps[idx] for idx in gap_indices if idx < len(gt_caps)]
            if len(gt_gap_caps) != m:
                continue
            
            # Ground truth gap vectors (fill cache if missing)
            missing_gt = [c for c in gt_gap_caps if c not in cache]
            if missing_gt:
                embedder.get_embeddings(f"{vid}_gt", missing_gt)
            gt_gap_vecs = np.array([cache[c] for c in gt_gap_caps])

            # Model predicted vectors (fill cache if missing)
            pred_texts = [recon_caps[k] for k in sorted_keys]
            missing_pred = [t for t in pred_texts if t not in cache]
            if missing_pred:
                embedder.get_embeddings(f"{vid}_pred", missing_pred)
            pred_vecs = np.array([cache[t] for t in pred_texts])

            # Video GT vectors for K=60 retrieval
            missing_vid_gt = [c for c in gt_caps if c not in cache]
            if missing_vid_gt:
                embedder.get_embeddings(f"{vid}_all_gt", missing_vid_gt)
            vid_gt_vecs = np.array([cache[c] for c in gt_caps])
            has_full_video = (len(vid_gt_vecs) == len(gt_caps))

            # -------------------------------------------------------------
            # 1. LLAMA-3.1-8B METRICS
            # -------------------------------------------------------------
            llama_cos_sims = np.sum(pred_vecs * gt_gap_vecs, axis=1)
            llama_f1s = [content_f1(p, g) for p, g in zip(pred_texts, gt_gap_caps)]
            
            # Video retrieval
            llama_vid_mrrs = []
            llama_vid_r1s = []
            if has_full_video:
                sim_vid = np.dot(pred_vecs, vid_gt_vecs.T)
                for i, gt_idx in enumerate(gap_indices):
                    gt_score = sim_vid[i, gt_idx]
                    better = np.sum([sim_vid[i, j] > gt_score + 1e-6 for j in range(len(vid_gt_vecs)) if j != gt_idx])
                    rank = better + 1
                    llama_vid_mrrs.append(1.0 / rank)
                    llama_vid_r1s.append(1 if rank == 1 else 0)

            # -------------------------------------------------------------
            # 2. REPEAT-BOUNDARY BASELINE
            # -------------------------------------------------------------
            b_idx = min(boundary_idx, len(gt_caps) - 1)
            b_cap = gt_caps[b_idx]
            b_vec = cache.get(b_cap)
            
            rep_cos_sims = []
            rep_f1s = []
            rep_vid_mrrs = []
            rep_vid_r1s = []
            if b_vec is not None:
                rep_cos_sims = [float(np.dot(b_vec, gt_gap_vecs[i])) for i in range(m)]
                rep_f1s = [content_f1(b_cap, gt_gap_caps[i]) for i in range(m)]
                if has_full_video:
                    sim_b = np.dot(b_vec, vid_gt_vecs.T)
                    for gt_idx in gap_indices:
                        gt_score = sim_b[gt_idx]
                        better = np.sum([sim_b[j] > gt_score + 1e-6 for j in range(len(vid_gt_vecs)) if j != gt_idx])
                        rank = better + 1
                        rep_vid_mrrs.append(1.0 / rank)
                        rep_vid_r1s.append(1 if rank == 1 else 0)

            # -------------------------------------------------------------
            # 3. RANDOM CORPUS BASELINE
            # -------------------------------------------------------------
            rand_indices = rng.integers(0, len(all_vecs), size=m)
            rand_vecs = all_vecs[rand_indices]
            rand_texts = [all_caps[idx] for idx in rand_indices]
            rand_cos_sims = np.sum(rand_vecs * gt_gap_vecs, axis=1)
            rand_f1s = [content_f1(r, g) for r, g in zip(rand_texts, gt_gap_caps)]

            # -------------------------------------------------------------
            # 4. RANDOM WITHIN-VIDEO BASELINE
            # -------------------------------------------------------------
            non_gap_indices = [idx for idx in range(len(gt_caps)) if idx not in gap_indices]
            w_rand_indices = rng.choice(non_gap_indices, size=m, replace=True)
            w_rand_caps = [gt_caps[idx] for idx in w_rand_indices]
            w_rand_vecs = np.array([cache[c] for c in w_rand_caps if c in cache])
            if len(w_rand_vecs) == m:
                w_rand_cos_sims = np.sum(w_rand_vecs * gt_gap_vecs, axis=1)
            else:
                w_rand_cos_sims = rand_cos_sims
            w_rand_f1s = [content_f1(w, g) for w, g in zip(w_rand_caps, gt_gap_caps)]

            # Record per-video aggregated metrics
            records.append({
                "video_id": vid,
                "position": pos_name,
                # Cosine Similarity
                "llama_cos_sim": float(np.mean(llama_cos_sims)),
                "repeat_cos_sim": float(np.mean(rep_cos_sims)) if rep_cos_sims else np.nan,
                "rand_corpus_cos_sim": float(np.mean(rand_cos_sims)),
                "rand_within_cos_sim": float(np.mean(w_rand_cos_sims)),
                # Content F1
                "llama_content_f1": float(np.mean(llama_f1s)),
                "repeat_content_f1": float(np.mean(rep_f1s)) if rep_f1s else np.nan,
                "rand_corpus_content_f1": float(np.mean(rand_f1s)),
                "rand_within_content_f1": float(np.mean(w_rand_f1s)),
                # Video Retrieval MRR (K=60)
                "llama_mrr_k60": float(np.mean(llama_vid_mrrs)) if llama_vid_mrrs else np.nan,
                "repeat_mrr_k60": float(np.mean(rep_vid_mrrs)) if rep_vid_mrrs else np.nan,
                # Video Retrieval Recall@1 (K=60)
                "llama_r1_k60": float(np.mean(llama_vid_r1s)) if llama_vid_r1s else np.nan,
                "repeat_r1_k60": float(np.mean(rep_vid_r1s)) if rep_vid_r1s else np.nan,
            })

    df = pd.DataFrame(records)
    os.makedirs(os.path.dirname(args.per_video_csv), exist_ok=True)
    df.to_csv(args.per_video_csv, index=False)
    print(f"\nSaved per-video details ({len(df)} rows) to {args.per_video_csv}")

    # -------------------------------------------------------------
    # STATISTICAL COMPARISON ACROSS POSITIONS AND OVERALL
    # -------------------------------------------------------------
    summary_rows = []
    analysis_groups = [("OVERALL (All Positions)", df)] + [(f"Position: {p.upper()}", df[df["position"] == p]) for p in ["start", "mid", "end"]]

    print("\n" + "=" * 80)
    print("STATISTICAL BENCHMARK: LLAMA-3.1-8B VS BASELINES")
    print("=" * 80)

    for group_name, sub_df in analysis_groups:
        n = len(sub_df)
        print(f"\n### {group_name} (N = {n} videos)")
        print("-" * 70)

        for metric in ["cos_sim", "content_f1", "mrr_k60"]:
            llama_col = f"llama_{metric}"
            repeat_col = f"repeat_{metric}"
            rand_corp_col = f"rand_corpus_{metric}"
            rand_with_col = f"rand_within_{metric}"

            llama_vals = sub_df[llama_col].dropna().to_numpy()
            rep_vals = sub_df[repeat_col].dropna().to_numpy()
            rc_vals = sub_df[rand_corp_col].dropna().to_numpy() if rand_corp_col in sub_df else None
            rw_vals = sub_df[rand_with_col].dropna().to_numpy() if rand_with_col in sub_df else None

            l_mean, (l_ci_low, l_ci_high) = np.mean(llama_vals), bootstrap_ci(llama_vals)
            r_mean, (r_ci_low, r_ci_high) = np.mean(rep_vals), bootstrap_ci(rep_vals)

            # Paired comparison: Llama vs Repeat-Boundary
            paired_diff = llama_vals - rep_vals[:len(llama_vals)]
            d_repeat = np.mean(paired_diff) / (np.std(paired_diff) + 1e-9)
            t_res = stats.ttest_rel(llama_vals, rep_vals[:len(llama_vals)])
            w_res = stats.wilcoxon(llama_vals, rep_vals[:len(llama_vals)])
            llama_wins = np.sum(paired_diff > 0)
            win_pct = llama_wins / len(paired_diff) * 100

            print(f"  [{metric.upper()}]:")
            print(f"    Llama-3.1-8B:      {l_mean:.4f}  [95% CI: {l_ci_low:.4f} - {l_ci_high:.4f}]")
            print(f"    Repeat-Boundary:   {r_mean:.4f}  [95% CI: {r_ci_low:.4f} - {r_ci_high:.4f}]")
            if rc_vals is not None:
                rc_mean = np.mean(rc_vals)
                print(f"    Random-Corpus:     {rc_mean:.4f}")
            if rw_vals is not None:
                rw_mean = np.mean(rw_vals)
                print(f"    Random-Within-Vid: {rw_mean:.4f}")
            print(f"    Difference (L - R): {np.mean(paired_diff):+.4f} (Cohen's d = {d_repeat:.3f})")
            print(f"    Paired t-test:     t = {t_res.statistic:.3f}, p = {t_res.pvalue:.2e}")
            print(f"    Wilcoxon test:     W = {w_res.statistic:.1f}, p = {w_res.pvalue:.2e}")
            print(f"    Llama Win Rate:    {llama_wins}/{len(paired_diff)} ({win_pct:.1f}%)")

            summary_rows.append({
                "group": group_name,
                "metric": metric,
                "N": n,
                "llama_mean": l_mean,
                "llama_ci_low": l_ci_low,
                "llama_ci_high": l_ci_high,
                "repeat_mean": r_mean,
                "repeat_ci_low": r_ci_low,
                "repeat_ci_high": r_ci_high,
                "rand_corpus_mean": np.mean(rc_vals) if rc_vals is not None else np.nan,
                "rand_within_mean": np.mean(rw_vals) if rw_vals is not None else np.nan,
                "diff_llama_repeat": np.mean(paired_diff),
                "cohens_d": d_repeat,
                "t_stat": t_res.statistic,
                "t_pvalue": t_res.pvalue,
                "wilcoxon_stat": w_res.statistic,
                "wilcoxon_pvalue": w_res.pvalue,
                "llama_win_pct": win_pct
            })

    sum_df = pd.DataFrame(summary_rows)
    sum_df.to_csv(args.output_csv, index=False)
    print(f"\nSaved summary CSV to {args.output_csv}")

if __name__ == "__main__":
    main()
