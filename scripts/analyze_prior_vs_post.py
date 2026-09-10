#!/usr/bin/env python3
"""
Comprehensive Statistical Analysis of A Priori Video/Caption Metrics
versus Post-Reconstruction Performance (Phi-3, Llama-3.1-8B, Visual Interpolation).
"""

import json
import glob
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd
from scipy import stats
from huggingface_hub import HfApi, hf_hub_download

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
MASTER_OUTPUT_CSV = RESULTS_DIR / "prior_vs_post_reconstruction_master.csv"


def load_base_integration():
    csv_path = RESULTS_DIR / "phi_vs_video_integration_summary.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing {csv_path}")
    df = pd.read_csv(csv_path)
    # Rename for clarity
    rename_map = {
        "text_mrr": "phi_mrr",
        "text_temporal_ndcg": "phi_t_ndcg",
        "text_cos_sim_mean": "phi_cos_sim_mean",
        "text_cos_sim_min": "phi_cos_sim_min",
        "text_cos_sim_residual_mean": "phi_cos_sim_residual_mean",
        "text_cos_sim_residual_min": "phi_cos_sim_residual_min",
    }
    return df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})


def load_llama_v3():
    api = HfApi()
    files = api.list_repo_files("Y3/dense_video_captions", repo_type="dataset")
    v3_files = [f for f in files if f.startswith("reconstruction/wild4_llama_w3_window_v3/") and f.endswith(".json")]

    llama_data = defaultdict(lambda: {
        "mrr": [], "r1": [], "cos_sim": [], "cos_sim_min": [],
        "mrr_i0": None, "mrr_i29": None, "mrr_i59": None
    })

    for f in v3_files:
        p = hf_hub_download("Y3/dense_video_captions", f, repo_type="dataset")
        with open(p) as fp:
            d = json.load(fp)
        vid = d.get("video_id")
        metrics = d.get("metrics", {})
        
        mrr = metrics.get("mrr")
        r1 = metrics.get("recall_at_1")
        cs = metrics.get("cos_sim", [])

        if mrr is not None:
            llama_data[vid]["mrr"].append(mrr)
            if "i=0" in f: llama_data[vid]["mrr_i0"] = mrr
            elif "i=29" in f: llama_data[vid]["mrr_i29"] = mrr
            elif "i=59" in f: llama_data[vid]["mrr_i59"] = mrr
        if r1 is not None:
            llama_data[vid]["r1"].append(r1)
        if cs:
            llama_data[vid]["cos_sim"].extend(cs)
            llama_data[vid]["cos_sim_min"].append(min(cs))

    rows = []
    for vid, vals in llama_data.items():
        norm_id = vid.replace("'", "_")
        rows.append({
            "norm_id": norm_id,
            "llama_mrr": float(np.mean(vals["mrr"])) if vals["mrr"] else None,
            "llama_recall_at_1": float(np.mean(vals["r1"])) if vals["r1"] else None,
            "llama_cos_sim_mean": float(np.mean(vals["cos_sim"])) if vals["cos_sim"] else None,
            "llama_cos_sim_min": float(np.mean(vals["cos_sim_min"])) if vals["cos_sim_min"] else None,
            "llama_mrr_start": vals["mrr_i0"],
            "llama_mrr_middle": vals["mrr_i29"],
            "llama_mrr_end": vals["mrr_i59"],
        })
    return pd.DataFrame(rows)


def main():
    print("Loading data...")
    df_base = load_base_integration()
    df_base["norm_id"] = df_base["video_id"].str.replace("'", "_")
    
    print("Loading Llama 3.1 v3 data...")
    df_llama = load_llama_v3()
    
    merged = pd.merge(df_base, df_llama, on="norm_id", how="left")
    
    # Compute new comparative deltas
    merged["llama_mrr_delta"] = merged["llama_mrr"] - merged["video_mrr"]
    merged["llama_vs_phi_mrr"] = merged["llama_mrr"] - merged["phi_mrr"]
    
    # Save master CSV
    merged.to_csv(MASTER_OUTPUT_CSV, index=False)
    print(f"✓ Master dataset saved to: {MASTER_OUTPUT_CSV} (N={len(merged)} videos)")

    # Print Correlations
    prior_vars = [
        ("apcs_nll", "Caption Language Prior NLL (Surprisal)"),
        ("apcs_perplexity", "Caption Language Prior Perplexity"),
        ("video_surprisal_avg", "Visual Motion Distance (Avg)"),
        ("video_surprisal_var", "Visual Motion Variance"),
        ("video_surprisal_max", "Visual Motion Max Peak"),
    ]

    post_vars = [
        ("phi_mrr", "Phi-3 MRR"),
        ("llama_mrr", "Llama-3.1-8B MRR"),
        ("video_mrr", "Visual Vector MRR"),
        ("llama_mrr_delta", "Llama vs Visual Delta"),
        ("llama_vs_phi_mrr", "Llama vs Phi-3 Delta"),
        ("phi_cos_sim_mean", "Phi-3 Cos Sim Mean"),
        ("phi_cos_sim_min", "Phi-3 Cos Sim Min (Bottleneck)"),
        ("llama_cos_sim_mean", "Llama-3.1 Cos Sim Mean"),
        ("llama_cos_sim_min", "Llama-3.1 Cos Sim Min (Bottleneck)"),
        ("video_cos_sim_mean", "Visual Cos Sim Mean"),
        ("video_cos_sim_min", "Visual Cos Sim Min (Bottleneck)"),
    ]

    print("\n" + "=" * 90)
    print("=== STATISTICAL CORRELATION MATRIX: PRIOR PROPERTIES vs POST-RECONSTRUCTION ===")
    print("=" * 90)
    print(f"{'Prior Metric':<32} | {'Post-Recon Metric':<30} | {'Pearson r':<10} | {'p-val':<8} | {'Spearman ρ':<10} | {'p-val':<8} | {'Sig'}")
    print("-" * 115)

    significant_findings = []

    for p_col, p_name in prior_vars:
        if p_col not in merged.columns:
            continue
        for post_col, post_name in post_vars:
            if post_col not in merged.columns:
                continue
            sub = merged[[p_col, post_col]].dropna()
            if len(sub) < 10:
                continue
            r, p_pearson = stats.pearsonr(sub[p_col], sub[post_col])
            rho, p_spearman = stats.spearmanr(sub[p_col], sub[post_col])

            sig_mark = ""
            if p_pearson < 0.001 or p_spearman < 0.001: sig_mark = "***"
            elif p_pearson < 0.01 or p_spearman < 0.01: sig_mark = "**"
            elif p_pearson < 0.05 or p_spearman < 0.05: sig_mark = "*"

            print(f"{p_col:<32} | {post_col:<30} | {r:+7.4f}    | {p_pearson:7.4f} | {rho:+7.4f}    | {p_spearman:7.4f} | {sig_mark}")

            if p_pearson < 0.05 or p_spearman < 0.05:
                significant_findings.append({
                    "prior": p_col, "post": post_col,
                    "r": r, "p_r": p_pearson,
                    "rho": rho, "p_rho": p_spearman,
                    "prior_desc": p_name, "post_desc": post_name
                })

    print("-" * 115)
    print("Significance levels: * p < 0.05, ** p < 0.01, *** p < 0.001")

    # Category breakdowns
    print("\n" + "=" * 90)
    print("=== CATEGORY STRATIFICATION: PRIOR SURPRISAL vs MODEL ADVANTAGE ===")
    print("=" * 90)
    cat_summary = merged.groupby("category").agg(
        n=("video_id", "count"),
        apcs_nll_mean=("apcs_nll", "mean"),
        video_surp_mean=("video_surprisal_avg", "mean"),
        phi_mrr=("phi_mrr", "mean"),
        llama_mrr=("llama_mrr", "mean"),
        video_mrr=("video_mrr", "mean"),
        llama_adv=("llama_mrr_delta", "mean"),
        llama_over_phi=("llama_vs_phi_mrr", "mean"),
    ).reset_index()
    print(cat_summary.to_string(index=False))

    return merged, significant_findings

if __name__ == "__main__":
    main()
