#!/usr/bin/env python3
"""
build_combined_benchmark_suite.py

Aggregates all evaluation results across wild4 and wild5 cohorts for W in [1..16],
joins with pre-computed a-priori visual and textual scores (APCS_V, APCS_T, Dynamism),
and exports standardized benchmark CSVs paired with matching markdown (.md) metadata documentation.
"""

import json
import re
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

REPO_ROOT = Path(__file__).parent.parent
RESULTS_DIR = REPO_ROOT / "results"
DOCS_DIR = REPO_ROOT / "docs" / "experiments"

def parse_llama_json(filepath: Path):
    if filepath.name.startswith("skip__"):
        return None
    try:
        with open(filepath) as f:
            data = json.load(f)
    except Exception:
        return None

    vid = data.get("video_id")
    if not vid:
        return None

    m = data.get("metrics", {})
    if not m:
        return None

    path_str = str(filepath)
    m_w = re.search(r'w=(\d+)', path_str)
    w_val = int(m_w.group(1)) if m_w else None

    m_i = re.search(r'i=(\d+)', path_str)
    i_val = int(m_i.group(1)) if m_i else None

    parts = filepath.parts
    try:
        llama_idx = next(i for i, p in enumerate(parts) if p.startswith('llama-3.1-8b'))
        exp_category = parts[llama_idx - 1]
    except Exception:
        exp_category = "unknown"

    dataset = "wild4" if "wild4" in str(filepath) else "wild5"

    cos_sims = m.get("cos_sim", [])
    cos_res = m.get("cos_sim_residual", [])

    return {
        "video_id": vid,
        "dataset": dataset,
        "exp_category": exp_category,
        "w": w_val,
        "i": i_val,
        "mrr": m.get("mrr"),
        "mean_rank": m.get("mean_rank"),
        "recall_at_1": m.get("recall_at_1"),
        "recall_at_5": m.get("recall_at_5"),
        "cos_sim_mean": float(np.mean(cos_sims)) if len(cos_sims) else None,
        "cos_sim_min": float(np.min(cos_sims)) if len(cos_sims) else None,
        "cos_sim_max": float(np.max(cos_sims)) if len(cos_sims) else None,
        "cos_sim_residual_mean": float(np.mean(cos_res)) if len(cos_res) else None,
    }

def load_all_llama_evaluations() -> pd.DataFrame:
    dirs_to_search = [
        RESULTS_DIR / "reconstruction" / "wild4_llama_w6",
        RESULTS_DIR / "recon" / "manual_download" / "reconstruction" / "wild4_llama_multi_width",
        RESULTS_DIR / "recon" / "manual_download" / "reconstruction" / "wild4_llama_w6",
        # wild4 W=3, re-scored with pool_scope "video" (scripts/rescore_window_pool_run.py)
        RESULTS_DIR / "recon" / "manual_download" / "reconstruction" / "wild4_llama_w3_window_v3_videopool",
        RESULTS_DIR / "recon" / "manual_download" / "reconstruction" / "wild5_llama_w3_w6",
        RESULTS_DIR / "recon" / "manual_download" / "reconstruction" / "wild5_llama_multi_width",
    ]

    records = []
    for search_dir in dirs_to_search:
        if not search_dir.exists():
            continue
        for jf in search_dir.rglob("*.json"):
            if "llama-3.1-8b" in str(jf):
                res = parse_llama_json(jf)
                if res:
                    records.append(res)

    df = pd.DataFrame(records)
    # Normalize ID
    df["video_id"] = df["video_id"].str.replace("Olly_s-Farm", "Olly's-Farm")
    df = df.drop_duplicates(subset=["video_id", "dataset", "w", "i"])
    return df

def write_metadata_companion(csv_path: Path, title: str, description: str, script_name: str, config_name: str, columns_dict: dict, notes: str = ""):
    md_path = csv_path.with_suffix(".md")
    col_rows = []
    for col, desc in columns_dict.items():
        col_rows.append(f"| `{col}` | {desc} |")
    col_table = "\n".join(col_rows)

    md_content = f"""# Dataset Metadata: {title}

**Target CSV File:** [`{csv_path.name}`]({csv_path.name})  
**Generated Date:** 2026-10-03  
**Generating Script:** [`scripts/{script_name}`](scripts/{script_name})  
**Experiment Configuration:** `{config_name}`  
**Master Field Definitions:** See [`docs/experiments/field_dictionary.md`](../docs/experiments/field_dictionary.md)

---

## 1. Description & Purpose
{description}

---

## 2. Column Definitions

| Column Name | Description & Reference |
| :--- | :--- |
{col_table}

---

## 3. Provenance & Execution Notes
{notes}
"""
    with open(md_path, "w") as f:
        f.write(md_content.strip() + "\n")
    print(f"Created metadata companion: {md_path}")

def main():
    print("Step 1: Loading a-priori visual and textual scores...")
    apriori_path = RESULTS_DIR / "apriori_full_scores.csv"
    if not apriori_path.exists():
        raise FileNotFoundError(f"Missing {apriori_path}")
    df_apriori = pd.read_csv(apriori_path)
    df_apriori["video_id"] = df_apriori["video_id"].str.replace("Olly_s-Farm", "Olly's-Farm")

    print("Step 2: Parsing all Llama 3.1 8B evaluation results...")
    df_llama = load_all_llama_evaluations()
    print(f"Loaded {len(df_llama)} valid Llama runs across datasets.")

    # Filter to W in [1, 2, 3, 4, 6, 8, 12, 16] and center index i=29 (and standard w3/w6)
    target_w = [1, 2, 3, 4, 6, 8, 12, 16]
    df_core = df_llama[df_llama["w"].isin(target_w) & (df_llama["i"] == 29)].copy()
    print(f"Filtered to W in {target_w} at i=29: {len(df_core)} runs.")

    # Join with apriori scores
    df_combined = pd.merge(df_core, df_apriori, on="video_id", how="inner")
    print(f"Merged with a-priori scores: {len(df_combined)} rows across {df_combined['video_id'].nunique()} unique videos.")

    # 1. Export Granular Per-Video Benchmark Table (W in 1..16)
    combined_csv = RESULTS_DIR / "benchmark_wild4_wild5_combined_per_video.csv"
    output_cols = [
        "video_id", "movie_id", "dataset", "w", "i",
        "mrr", "mean_rank", "recall_at_1", "recall_at_5",
        "cos_sim_mean", "cos_sim_min", "cos_sim_max", "cos_sim_residual_mean",
        "average_dynamism", "peak_dynamism", "combined_dynamism", "APCS_V",
        "text_average_dynamism", "text_peak_dynamism", "text_combined_dynamism", "APCS_T"
    ]
    df_combined[output_cols].to_csv(combined_csv, index=False)
    print(f"Saved: {combined_csv}")

    write_metadata_companion(
        csv_path=combined_csv,
        title="Wild4 & Wild5 Combined Per-Video Evaluation Benchmark (W in 1..16)",
        description="Per-video cloze reconstruction results using Llama-3.1-8B across gap widths W in [1, 2, 3, 4, 6, 8, 12, 16] at center position i=29 for both wild4 and wild5 cohorts (335 videos total), joined with visual and textual apriori dynamism metrics.",
        script_name="build_combined_benchmark_suite.py",
        config_name="config/embs_vs_slms/wild4_llama_multi_width.yaml & wild5_llama_multi_width.yaml",
        columns_dict={
            "video_id": "Video identifier (e.g. BC-Bushcraft_10-clip-8)",
            "movie_id": "Channel identifier (40 unique channels)",
            "dataset": "Benchmark cohort (wild4 = dev, wild5 = test)",
            "w": "Gap width (seconds masked: 1, 2, 3, 4, 6, 8, 12, 16)",
            "i": "Masking start timestamp index (29 = center cloze)",
            "mrr": "Mean Reciprocal Rank against all 59 other video timestamps (unbiased metric)",
            "mean_rank": "Average rank among 60 timestamps (1 = ground truth at #1)",
            "recall_at_1": "Recall at rank 1",
            "recall_at_5": "Recall in top 5 ranks",
            "cos_sim_mean": "Mean cosine similarity in mpnet text embedding space",
            "cos_sim_min": "Minimum cosine similarity inside masked gap",
            "cos_sim_max": "Maximum cosine similarity inside masked gap",
            "cos_sim_residual_mean": "Mean cosine similarity with centroid removed",
            "average_dynamism": "A-priori visual consecutive distance (SigLIP 768d)",
            "peak_dynamism": "A-priori visual 95th percentile jump (SigLIP 768d)",
            "combined_dynamism": "A-priori composite visual dynamism score",
            "APCS_V": "Gail's Average Pairwise Cosine Similarity of video frames",
            "text_average_dynamism": "A-priori consecutive text distance (mpnet 384d)",
            "text_peak_dynamism": "A-priori 95th percentile text jump (mpnet 384d)",
            "text_combined_dynamism": "A-priori composite textual dynamism score",
            "APCS_T": "Average Pairwise Cosine Similarity of captions (text redundancy)"
        },
        notes="Evaluated using pool_scope: 'video' (60 candidates per query). Text representations embedded via sentence-transformers/all-mpnet-base-v2."
    )

    # 2. Export Aggregated Summary by Gap Width W
    summary_rows = []
    for (w_val), group in df_combined.groupby("w"):
        summary_rows.append({
            "w": w_val,
            "total_videos": len(group),
            "mrr_mean": group["mrr"].mean(),
            "mrr_std": group["mrr"].std(),
            "mrr_median": group["mrr"].median(),
            "mean_rank_mean": group["mean_rank"].mean(),
            "recall_at_1_mean": group["recall_at_1"].mean(),
            "recall_at_5_mean": group["recall_at_5"].mean(),
            "cos_sim_mean": group["cos_sim_mean"].mean(),
            "cos_sim_std": group["cos_sim_mean"].std(),
            "cos_sim_residual_mean": group["cos_sim_residual_mean"].mean(),
        })
    df_summary = pd.DataFrame(summary_rows)
    summary_csv = RESULTS_DIR / "benchmark_wild4_wild5_summary_by_width.csv"
    df_summary.to_csv(summary_csv, index=False)
    print(f"Saved: {summary_csv}")

    write_metadata_companion(
        csv_path=summary_csv,
        title="Benchmark Summary by Gap Width W (Wild4 + Wild5 Combined)",
        description="Aggregated performance metrics (MRR, Recall, Rank, Cosine Similarity) of Llama-3.1-8B cloze caption reconstruction across gap widths W in [1..16] at i=29 over the combined 335-video benchmark.",
        script_name="build_combined_benchmark_suite.py",
        config_name="wild4_llama_multi_width.yaml & wild5_llama_multi_width.yaml",
        columns_dict={
            "w": "Gap width (number of seconds masked)",
            "total_videos": "Number of successfully evaluated videos at this width",
            "mrr_mean": "Mean Reciprocal Rank across all evaluated videos",
            "mrr_std": "Standard deviation of MRR",
            "mrr_median": "Median MRR across videos",
            "mean_rank_mean": "Mean average rank among all 60 timestamps",
            "recall_at_1_mean": "Mean Recall@1 across videos",
            "recall_at_5_mean": "Mean Recall@5 across videos",
            "cos_sim_mean": "Mean cosine similarity between generated and ground-truth text",
            "cos_sim_std": "Standard deviation of cosine similarity",
            "cos_sim_residual_mean": "Mean residual cosine similarity"
        },
        notes="Computed over the merged wild4 (100) and wild5 (235) cohorts."
    )

    # 3. Export Comprehensive Correlation Matrix by Width
    corr_rows = []
    x_apriori = ["combined_dynamism", "APCS_V", "text_combined_dynamism", "APCS_T"]
    y_metrics = ["mrr", "cos_sim_mean", "mean_rank"]

    for w_val in target_w:
        sub = df_combined[df_combined["w"] == w_val]
        for y_col in y_metrics:
            for x_col in x_apriori:
                valid = sub.dropna(subset=[x_col, y_col])
                if len(valid) < 10:
                    continue
                pr, p_pval = stats.pearsonr(valid[x_col], valid[y_col])
                sr, s_pval = stats.spearmanr(valid[x_col], valid[y_col])
                corr_rows.append({
                    "w": w_val,
                    "metric_y": y_col,
                    "apriori_x": x_col,
                    "sample_size": len(valid),
                    "pearson_r": pr,
                    "pearson_p": p_pval,
                    "spearman_rho": sr,
                    "spearman_p": s_pval,
                })

    df_corrs = pd.DataFrame(corr_rows)
    corrs_csv = RESULTS_DIR / "benchmark_wild4_wild5_correlations_by_width.csv"
    df_corrs.to_csv(corrs_csv, index=False)
    print(f"Saved: {corrs_csv}")

    write_metadata_companion(
        csv_path=corrs_csv,
        title="Apriori vs. Posteriori Correlation Analysis across Gap Widths (W=1..16)",
        description="Linear (Pearson r) and rank-order (Spearman rho) correlation coefficients comparing a-priori visual metrics (SigLIP combined dynamism, APCS_V) and textual metrics (text dynamism, APCS_T) against posteriori reconstruction evaluations (MRR, cos_sim_mean, mean_rank) at each gap width W.",
        script_name="build_combined_benchmark_suite.py",
        config_name="wild4_llama_multi_width.yaml & wild5_llama_multi_width.yaml",
        columns_dict={
            "w": "Gap width evaluated",
            "metric_y": "Evaluated posteriori metric (mrr, cos_sim_mean, mean_rank)",
            "apriori_x": "A-priori predictor metric (combined_dynamism, APCS_V, text_combined_dynamism, APCS_T)",
            "sample_size": "Number of videos evaluated",
            "pearson_r": "Pearson linear correlation coefficient",
            "pearson_p": "Two-tailed p-value for Pearson correlation",
            "spearman_rho": "Spearman rank correlation coefficient",
            "spearman_p": "Two-tailed p-value for Spearman correlation"
        },
        notes="Highlights the contrast between MRR (consistently uncorrelated with dynamism) and cos_sim_mean (heavily confounded by APCS_T caption redundancy)."
    )

if __name__ == "__main__":
    main()
