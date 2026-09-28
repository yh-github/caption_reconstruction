#!/usr/bin/env python3
"""
Runs regression tests for Hypotheses H1, H2, and H3 with cluster-robust standard errors:
- H1: Direct text-space comparison (LLM vs. Copy-Nearest vs. Text LERP)
- H2: Regress text lift (Sim_LLM - Sim_Repeat) on domain (Military vs. Nature) clustered by channel
- H3: Regress Delta/N on domain + visual frame continuity clustered by channel
"""

import json
import re
from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

REPO_ROOT = Path(__file__).resolve().parent.parent

def load_data():
    df = pd.read_csv(REPO_ROOT / "results" / "unified_benchmark_master.csv")
    with open(REPO_ROOT / "results" / "video_categories.json") as f:
        cat_map = json.load(f)

    def get_channel(vid):
        if vid in cat_map and "series" in cat_map[vid]:
            return cat_map[vid]["series"]
        m = re.match(r"^([A-Za-z0-9\-]+?)(?:_\d+.*|-clip-.*|$)", vid)
        return m.group(1) if m else vid

    df["channel"] = df["video_id"].apply(get_channel)
    return df

def get_visual_continuity():
    embs_dir = REPO_ROOT / "local" / "wild_videos_embs_siglip"
    vid_v_cont = {}
    for p in embs_dir.glob("*.npy"):
        vid = p.stem
        try:
            embs = np.load(p)
            if len(embs) > 10:
                embs = embs / np.linalg.norm(embs, axis=1, keepdims=True)
                v_adj = np.mean(np.sum(embs[:-1] * embs[1:], axis=1))
                vid_v_cont[vid] = float(v_adj)
        except Exception:
            pass
    return vid_v_cont

def main():
    df = load_data()
    v_cont = get_visual_continuity()

    # Build merged table for w=6
    sub = df[df["width"] == 6].copy()
    llama = sub[sub["method"] == "Llama-3.1-8B"][["dataset", "video_id", "channel", "index", "category", "cos_sim", "mean_rank"]].rename(columns={"cos_sim": "sim_llama", "mean_rank": "rank_llama"})
    vis = sub[sub["method"] == "Visual_SigLIP_MeanClosest"][["dataset", "video_id", "index", "cos_sim", "mean_rank"]].rename(columns={"cos_sim": "sim_vis", "mean_rank": "rank_vis"})
    c_rep = sub[sub["method"] == "Caption_RepeatClosest"][["dataset", "video_id", "index", "cos_sim", "mean_rank"]].rename(columns={"cos_sim": "sim_crep", "mean_rank": "rank_crep"})
    c_lerp = sub[sub["method"] == "Caption_MeanClosest"][["dataset", "video_id", "index", "cos_sim", "mean_rank"]].rename(columns={"cos_sim": "sim_clerp", "mean_rank": "rank_clerp"})

    merged = pd.merge(llama, vis, on=["dataset", "video_id", "index"])
    merged = pd.merge(merged, c_rep, on=["dataset", "video_id", "index"])
    merged = pd.merge(merged, c_lerp, on=["dataset", "video_id", "index"])

    merged["delta_norm"] = 0.0
    for ds in ["Wild4", "Wild5"]:
        idx = merged["dataset"] == ds
        N = idx.sum()
        merged.loc[idx, "rank_text"] = merged.loc[idx, "sim_llama"].rank(ascending=False)
        merged.loc[idx, "rank_vis"] = merged.loc[idx, "sim_vis"].rank(ascending=False)
        merged.loc[idx, "delta_norm"] = (merged.loc[idx, "rank_text"] - merged.loc[idx, "rank_vis"]) / N

    merged["v_continuity"] = merged["video_id"].map(v_cont)
    merged["lift_over_repeat"] = merged["sim_llama"] - merged["sim_crep"]
    merged["lift_over_clerp"] = merged["sim_llama"] - merged["sim_clerp"]

    print("=================================================================")
    print("HYPOTHESIS 1: DIRECT TEST OF LLM VS. PERSISTENCE IN TEXT SPACE")
    print("=================================================================")
    h1_summary = merged.groupby("category")[["sim_llama", "sim_crep", "sim_clerp", "rank_llama", "rank_crep", "rank_clerp"]].mean()
    print(h1_summary)

    poles = merged[merged["category"].isin(["Military", "Nature & Scenery"])].dropna(subset=["v_continuity"]).copy()
    poles["is_nature"] = (poles["category"] == "Nature & Scenery").astype(float)

    print("\n=================================================================")
    print("HYPOTHESIS 2: REGRESSION OF TEXT LIFT OVER REPEAT ON DOMAIN")
    print("Clusters: 15 unique YouTube channels across 105 videos")
    print("=================================================================")
    m_h2 = smf.ols("lift_over_repeat ~ is_nature", data=poles).fit(cov_type="cluster", cov_kwds={"groups": poles["channel"]})
    print(m_h2.summary().tables[1])

    print("\n=================================================================")
    print("HYPOTHESIS 3: REGRESSION OF DELTA/N ON DOMAIN + VISUAL CONTINUITY")
    print("Testing if physical scene dynamics mediate the domain effect")
    print("=================================================================")
    m_h3_unadj = smf.ols("delta_norm ~ is_nature", data=poles).fit(cov_type="cluster", cov_kwds={"groups": poles["channel"]})
    print("--- Model 3A (Unadjusted Domain Effect) ---")
    print(m_h3_unadj.summary().tables[1])

    m_h3_adj = smf.ols("delta_norm ~ is_nature + v_continuity", data=poles).fit(cov_type="cluster", cov_kwds={"groups": poles["channel"]})
    print("\n--- Model 3B (Adjusted for Physical Visual Continuity) ---")
    print(m_h3_adj.summary().tables[1])

if __name__ == "__main__":
    main()
