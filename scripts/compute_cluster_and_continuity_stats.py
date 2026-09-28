#!/usr/bin/env python3
"""
Computes cluster-aggregated (video-level) and segment-level statistics for the paper:
1. Video and segment counts per split/domain
2. Normalized delta means, SDs, and cluster-bootstrap 95% CIs
3. Segment-level and video-level win rates with exact binomial tests
4. Cross-pole Mann-Whitney U and Cohen's d at both segment and video level
5. Adjacent caption vs. adjacent visual continuity across categories
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parent.parent

def load_data():
    df = pd.read_csv(REPO_ROOT / "results" / "unified_benchmark_master.csv")
    return df

def analyze_split(df, dataset_name, width_val):
    sub = df[(df["dataset"] == dataset_name) & (df["width"] == width_val)]
    llama = sub[sub["method"] == "Llama-3.1-8B"][["video_id", "index", "category", "cos_sim"]].rename(columns={"cos_sim": "cos_sim_llama"})
    vis = sub[sub["method"] == "Visual_SigLIP_MeanClosest"][["video_id", "index", "cos_sim"]].rename(columns={"cos_sim": "cos_sim_vis"})
    merged = pd.merge(llama, vis, on=["video_id", "index"])
    N = len(merged)
    merged["rank_text"] = merged["cos_sim_llama"].rank(ascending=False)
    merged["rank_vis"] = merged["cos_sim_vis"].rank(ascending=False)
    merged["delta_norm"] = (merged["rank_text"] - merged["rank_vis"]) / N
    merged["text_win"] = merged["delta_norm"] < 0

    categories = ["Military", "Natural Disaster", "Survival", "Action & Vehicle", "Farming", "Nature & Scenery"]
    results = []
    for cat in categories:
        c_df = merged[merged["category"] == cat]
        n_seg = len(c_df)
        n_vids = c_df["video_id"].nunique()
        seg_mean = c_df["delta_norm"].mean()
        seg_std = c_df["delta_norm"].std()
        seg_wins = c_df["text_win"].sum()
        seg_win_rate = seg_wins / n_seg if n_seg else 0
        seg_bin_p = stats.binomtest(seg_wins, n_seg, 0.5).pvalue if n_seg else 1.0

        # Video-level aggregation
        vid_agg = c_df.groupby("video_id").agg(
            vid_delta=("delta_norm", "mean"),
            vid_win_frac=("text_win", "mean")
        )
        vid_mean = vid_agg["vid_delta"].mean()
        vid_std = vid_agg["vid_delta"].std()
        v_wins = (vid_agg["vid_delta"] < 0).sum()
        v_win_rate = v_wins / n_vids if n_vids else 0
        v_bin_p = stats.binomtest(v_wins, n_vids, 0.5).pvalue if n_vids else 1.0

        # Cluster bootstrap 95% CI for delta
        rng = np.random.default_rng(42)
        vid_list = vid_agg["vid_delta"].values
        boot_means = [np.mean(rng.choice(vid_list, size=len(vid_list), replace=True)) for _ in range(3000)]
        ci_low, ci_high = np.percentile(boot_means, [2.5, 97.5])

        results.append({
            "category": cat,
            "n_segs": n_seg,
            "n_vids": n_vids,
            "seg_mean": seg_mean,
            "seg_std": seg_std,
            "seg_wins": seg_wins,
            "seg_win_rate": seg_win_rate,
            "seg_bin_p": seg_bin_p,
            "vid_mean": vid_mean,
            "vid_std": vid_std,
            "vid_wins": v_wins,
            "vid_win_rate": v_win_rate,
            "vid_bin_p": v_bin_p,
            "boot_ci_low": ci_low,
            "boot_ci_high": ci_high
        })

    # Cross-pole stats: Military vs Nature & Scenery
    mil_seg = merged[merged["category"] == "Military"]["delta_norm"].values
    nat_seg = merged[merged["category"] == "Nature & Scenery"]["delta_norm"].values
    u_seg, p_seg = stats.mannwhitneyu(mil_seg, nat_seg)
    pooled_sd_seg = np.sqrt(((len(mil_seg)-1)*np.var(mil_seg, ddof=1) + (len(nat_seg)-1)*np.var(nat_seg, ddof=1)) / (len(mil_seg) + len(nat_seg) - 2))
    d_seg = (np.mean(mil_seg) - np.mean(nat_seg)) / pooled_sd_seg

    vid_agg_all = merged.groupby(["category", "video_id"])["delta_norm"].mean().reset_index()
    mil_vid = vid_agg_all[vid_agg_all["category"] == "Military"]["delta_norm"].values
    nat_vid = vid_agg_all[vid_agg_all["category"] == "Nature & Scenery"]["delta_norm"].values
    u_vid, p_vid = stats.mannwhitneyu(mil_vid, nat_vid)
    pooled_sd_vid = np.sqrt(((len(mil_vid)-1)*np.var(mil_vid, ddof=1) + (len(nat_vid)-1)*np.var(nat_vid, ddof=1)) / (len(mil_vid) + len(nat_vid) - 2))
    d_vid = (np.mean(mil_vid) - np.mean(nat_vid)) / pooled_sd_vid

    cross_pole = {
        "u_seg": u_seg, "p_seg": p_seg, "d_seg": d_seg,
        "u_vid": u_vid, "p_vid": p_vid, "d_vid": d_vid
    }

    return pd.DataFrame(results), cross_pole

def main():
    df = load_data()
    print("=================================================================")
    print("CLUSTER-AWARE RECONSTRUCTION BENCHMARK STATISTICS")
    print("=================================================================\n")

    for ds, w in [("Wild4", 6), ("Wild5", 6), ("Wild5", 3)]:
        res_df, cp = analyze_split(df, ds, w)
        print(f"--- Split: {ds} (w={w}s) ---")
        for _, r in res_df.iterrows():
            print(f"{r['category']:18} | Segs: {r['n_segs']:3d} (Wins: {r['seg_wins']:2d}/{r['n_segs']:2d} = {r['seg_win_rate']:.1%}, p={r['seg_bin_p']:.4f}) | Vids: {r['n_vids']:2d} (Wins: {r['vid_wins']:2d}/{r['n_vids']:2d} = {r['vid_win_rate']:.1%}, p={r['vid_bin_p']:.4f}) | Δ/N: {r['seg_mean']:+.3f} ± {r['seg_std']:.3f} | Vid Δ/N: {r['vid_mean']:+.3f} [95% CI: {r['boot_ci_low']:+.3f}, {r['boot_ci_high']:+.3f}]")
        print(f"Cross-Pole (Military vs Nature & Scenery):")
        print(f"  Segment-level: U = {cp['u_seg']}, p = {cp['p_seg']:.4e}, Cohen's d = {cp['d_seg']:.3f}")
        print(f"  Video-level:   U = {cp['u_vid']}, p = {cp['p_vid']:.4e}, Cohen's d = {cp['d_vid']:.3f}\n")

if __name__ == "__main__":
    main()
