#!/usr/bin/env python3
"""
Generate a consolidated per-video summary table integrating:
- Video Reconstruction Scores (MRR, Temporal NDCG, Cosine Sim, Residual Sim)
- Text Reconstruction Scores using Phi-3 (MRR, Temporal NDCG, Cosine Sim, Residual Sim)
- Comparative Deltas (MRR Delta, Temporal NDCG Delta)
- APCS (A Priori Caption Surprisal NLL & Perplexity)
- Video Surprisal (Computed on the exact 60-second evaluated window)
- Both evaluated video length (strictly 60s) and raw source file length

Completes in ~1 second.
"""

import glob
import json
from pathlib import Path
import numpy as np
import pandas as pd
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT / "src"))
from data.video_surprisal import VideoSurprisalScorer

RESULTS_DIR = PROJECT_ROOT / "results"
OUTPUT_CSV = RESULTS_DIR / "phi_vs_video_integration_summary.csv"


def load_categories():
    cat_file = RESULTS_DIR / "video_categories.json"
    if not cat_file.exists():
        return {}
    with open(cat_file, "r") as f:
        d = json.load(f)
    return {
        k.replace("'", "_"): (v.get("category", "Unknown") if isinstance(v, dict) else str(v))
        for k, v in d.items()
    }


def load_apcs_scores():
    prior_path = RESULTS_DIR / "scores" / "prior_scoring_7109bc27281d.json"
    if not prior_path.exists():
        return pd.DataFrame()
    with open(prior_path, "r") as f:
        data = json.load(f)
    rows = []
    for vid_id, res in data.get("scores", {}).items():
        wvs = res.get("whole_video_surprisal")
        if wvs:
            nll = wvs.get("avg_surprisal_nll")
            ppl = wvs.get("avg_perplexity")
            rows.append({
                "norm_id": vid_id.replace("'", "_"),
                "apcs_nll": nll,
                "apcs_perplexity": ppl,
                # Clear aliases for analysis
                "caption_surprisal_nll": nll,
                "caption_perplexity": ppl,
                "apcs_surprisal_nll": nll  # Retained for legacy compatibility
            })
    return pd.DataFrame(rows)


def load_phi_metrics():
    # 1. Temporal metrics (MRR, Temporal NDCG)
    temporal_file = RESULTS_DIR / "temporal_metrics_final.csv"
    if not temporal_file.exists():
        raise FileNotFoundError(f"Missing {temporal_file}")
    df_temp = pd.read_csv(temporal_file)
    df_temp["norm_id"] = df_temp["video_id"].str.replace("'", "_")
    phi_temp_agg = df_temp.groupby("norm_id")[["phi_mrr", "temporal_ndcg"]].mean().reset_index()
    phi_temp_agg = phi_temp_agg.rename(columns={
        "phi_mrr": "text_mrr",
        "temporal_ndcg": "text_temporal_ndcg"
    })

    # 2. Extract cosine similarity (mean, min, max, residuals) from raw JSON results
    json_pattern = str(RESULTS_DIR / "recon" / "manual_download" / "reconstruction" / "wild_dev_sim_text" / "*" / "*.json")
    json_files = glob.glob(json_pattern)
    cos_rows = []
    for jf in json_files:
        try:
            with open(jf) as fp:
                d = json.load(fp)
            vid = d.get("video_id")
            if not vid:
                continue
            metrics = d.get("metrics", {})
            cos_sims = metrics.get("cos_sim", [])
            cos_res = metrics.get("cos_sim_residual", [])
            cos_rows.append({
                "norm_id": vid.replace("'", "_"),
                "text_cos_sim_mean": float(np.mean(cos_sims)) if len(cos_sims) else None,
                "text_cos_sim_min": float(np.min(cos_sims)) if len(cos_sims) else None,
                "text_cos_sim_max": float(np.max(cos_sims)) if len(cos_sims) else None,
                "text_cos_sim_residual_mean": float(np.mean(cos_res)) if len(cos_res) else None,
                "text_cos_sim_residual_min": float(np.min(cos_res)) if len(cos_res) else None,
            })
        except Exception:
            pass

    if cos_rows:
        df_cos = pd.DataFrame(cos_rows).groupby("norm_id").mean().reset_index()
        # Aliases for backwards compatibility
        df_cos["text_cos_sim"] = df_cos["text_cos_sim_mean"]
        df_cos["text_cos_sim_residual"] = df_cos["text_cos_sim_residual_mean"]
        phi_agg = pd.merge(phi_temp_agg, df_cos, on="norm_id", how="left")
    else:
        phi_agg = phi_temp_agg

    return phi_agg


def load_video_metrics():
    # 1. Baseline full metrics (MRR, Temporal NDCG)
    base_file = RESULTS_DIR / "baseline_full_metrics.csv"
    if not base_file.exists():
        raise FileNotFoundError(f"Missing {base_file}")
    df_base = pd.read_csv(base_file)
    df_base["norm_id"] = df_base["video_id"].str.replace("'", "_")
    base_agg = df_base.groupby("norm_id")[["mrr", "temporal_ndcg"]].mean().reset_index()
    base_agg = base_agg.rename(columns={
        "mrr": "video_mrr",
        "temporal_ndcg": "video_temporal_ndcg"
    })

    # 2. Video cosine similarity (mean, min, max, residuals) from wild_dev_sim_vec_vid.csv
    vec_vid_file = RESULTS_DIR / "for_analysis" / "wild_dev_sim_vec_vid.csv"
    if vec_vid_file.exists():
        df_vid = pd.read_csv(vec_vid_file)
        df_vid["norm_id"] = df_vid["video_id"].str.replace("'", "_")
        vid_cols = ["cos_sim_mean", "cos_sim_min", "cos_sim_max", "cos_sim_residual_mean", "cos_sim_residual_min"]
        vid_cols_present = [c for c in vid_cols if c in df_vid.columns]
        vid_cos_agg = df_vid.groupby("norm_id")[vid_cols_present].mean().reset_index()
        vid_rename = {
            "cos_sim_mean": "video_cos_sim_mean",
            "cos_sim_min": "video_cos_sim_min",
            "cos_sim_max": "video_cos_sim_max",
            "cos_sim_residual_mean": "video_cos_sim_residual_mean",
            "cos_sim_residual_min": "video_cos_sim_residual_min",
        }
        vid_cos_agg = vid_cos_agg.rename(columns=vid_rename)
        # Aliases for backwards compatibility
        vid_cos_agg["video_cos_sim"] = vid_cos_agg["video_cos_sim_mean"]
        vid_cos_agg["video_cos_sim_residual"] = vid_cos_agg["video_cos_sim_residual_mean"]
        base_agg = pd.merge(base_agg, vid_cos_agg, on="norm_id", how="left")

    return base_agg


def load_video_surprisal_60s():
    """
    Computes video surprisal metrics strictly on the 60-second window
    evaluated in experiments, while recording the source video file length.
    """
    scorer = VideoSurprisalScorer()
    emb_dir = PROJECT_ROOT / "local" / "wild_videos_embs"
    rows = []

    if emb_dir.exists():
        for npy_path in emb_dir.glob("*.npy"):
            try:
                norm_id = npy_path.stem.replace("'", "_")
                embs = np.load(npy_path)
                raw_len = len(embs)
                embs_60 = embs[:60]
                res = scorer.calculate_surprisal(embs_60)
                rows.append({
                    "norm_id": norm_id,
                    "video_surprisal_avg": res.avg_cosine_distance,
                    "video_surprisal_max": res.max_cosine_distance,
                    "video_surprisal_var": res.variance_cosine_distance,
                    "video_length": len(embs_60),  # Strictly 60s
                    "raw_video_file_length": raw_len
                })
            except Exception as e:
                pass

    if rows:
        return pd.DataFrame(rows)

    # Fallback to existing CSV if local embs are not available
    surp_file = RESULTS_DIR / "video_surprisal_scores.csv"
    if surp_file.exists():
        df_surp = pd.read_csv(surp_file)
        df_surp["norm_id"] = df_surp["video_id"].str.replace("'", "_")
        df_surp["raw_video_file_length"] = df_surp["video_length"]
        df_surp["video_length"] = 60
        return df_surp[["norm_id", "video_avg_dist", "video_max_dist", "video_var_dist", "video_length", "raw_video_file_length"]].rename(
            columns={
                "video_avg_dist": "video_surprisal_avg",
                "video_max_dist": "video_surprisal_max",
                "video_var_dist": "video_surprisal_var"
            }
        )
    return pd.DataFrame()


def main():
    print("Consolidating per-video results for team integration...")
    cats = load_categories()

    # Load components
    df_phi = load_phi_metrics()
    df_video = load_video_metrics()
    df_apcs = load_apcs_scores()
    df_surprisal = load_video_surprisal_60s()

    # Merge on normalized ID
    merged = pd.merge(df_phi, df_video, on="norm_id", how="inner")
    if not df_apcs.empty:
        merged = pd.merge(merged, df_apcs, on="norm_id", how="left")
    if not df_surprisal.empty:
        merged = pd.merge(merged, df_surprisal, on="norm_id", how="left")

    # Add Category
    merged["category"] = merged["norm_id"].map(cats).fillna("Unknown")

    # Compute Comparative Deltas
    merged["mrr_delta"] = merged["text_mrr"] - merged["video_mrr"]
    merged["t_ndcg_delta"] = merged["text_temporal_ndcg"] - merged["video_temporal_ndcg"]

    # Re-order and standardize columns
    merged["video_id"] = merged["norm_id"]
    cols = [
        "video_id",
        "category",
        "video_length",
        "raw_video_file_length",
        "text_mrr",
        "video_mrr",
        "mrr_delta",
        "text_temporal_ndcg",
        "video_temporal_ndcg",
        "t_ndcg_delta",
        "text_cos_sim_mean",
        "text_cos_sim_min",
        "video_cos_sim_mean",
        "video_cos_sim_min",
        "text_cos_sim_residual_mean",
        "text_cos_sim_residual_min",
        "video_cos_sim_residual_mean",
        "video_cos_sim_residual_min",
        "apcs_nll",
        "apcs_perplexity",
        "video_surprisal_var",
        "video_surprisal_avg",
        "video_surprisal_max",
        # Legacy column names for backward compatibility
        "text_cos_sim",
        "video_cos_sim",
        "text_cos_sim_residual",
        "video_cos_sim_residual",
        "apcs_surprisal_nll",
        "caption_surprisal_nll",
        "caption_perplexity"
    ]

    final_cols = [c for c in cols if c in merged.columns]
    final_df = merged[final_cols].sort_values("video_id").reset_index(drop=True)

    # Save CSV
    final_df.to_csv(OUTPUT_CSV, index=False)
    print(f"Successfully generated: {OUTPUT_CSV}")
    print(f"Total videos processed: {len(final_df)}")
    print(f"Null values in dataset: {final_df.isnull().sum().sum()}")
    print("\n--- First 3 Sample Rows ---")
    print(final_df.head(3).to_string(index=False))


if __name__ == "__main__":
    main()
