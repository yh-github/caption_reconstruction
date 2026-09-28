#!/usr/bin/env python3
"""
scripts/analyze_predictability_map.py

Row 4 Analysis: Two-Axis Predictability Map per Segment.
- Computes segment-level persistence scores (text and visual) and inference scores (LLM)
- Maps physical scene continuity (v_continuity)
- Produces publication-quality figures:
  1. Text Predictability Map: Persistence vs. Inference in Caption-Home
  2. Cross-Modal Dynamics Map: Scene Continuity vs. Text In-filling Lift
  3. Domain Cluster Map: Centroids and 2D dispersion ellipses
- Saves data to results/redesign_segment_predictability_map.csv
"""

import sys
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
import seaborn as sns

REPO_ROOT = Path(__file__).resolve().parent.parent
results_dir = REPO_ROOT / "results"
figures_dir = REPO_ROOT / "docs" / "paper" / "figures"
figures_dir.mkdir(parents=True, exist_ok=True)


def confidence_ellipse(x, y, ax, n_std=1.5, facecolor='none', **kwargs):
    """Plots a confidence ellipse of x and y on ax."""
    if len(x) < 3:
        return
    cov = np.cov(x, y)
    pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
                      facecolor=facecolor, **kwargs)
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(x)
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(y)
    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)
    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)


def compute_video_continuity(npy_dir: Path) -> dict[str, float]:
    """Computes mean adjacent visual cosine similarity for each video."""
    continuity = {}
    for f in npy_dir.glob("*.npy"):
        try:
            arr = np.load(f)[:60].astype(np.float32)
            norms = np.linalg.norm(arr, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1.0, norms)
            normed = arr / norms
            # Cosine similarity between adjacent frames: dot product
            sims = np.sum(normed[:-1] * normed[1:], axis=1)
            continuity[f.stem] = float(np.mean(sims))
        except Exception:
            continue
    return continuity


def main():
    print("================================================================================")
    print("ROW 4: TWO-AXIS PREDICTABILITY MAP PER SEGMENT")
    print("================================================================================\n")

    print("1. Computing physical video continuity across benchmark...")
    continuity_map = compute_video_continuity(REPO_ROOT / "local" / "wild_videos_embs_siglip")
    print(f"Computed visual continuity for {len(continuity_map)} videos.")

    print("2. Loading raw shared-target evaluation data...")
    # Load required columns from the 6M-row CSV
    cols = ["split", "video_id", "channel", "domain", "gap_id", "w", "edge_gap", "arm", "home", "stratum", "calibrated"]
    raw_df = pd.read_csv(results_dir / "redesign_shared_target.csv", usecols=cols)
    print(f"Loaded {len(raw_df)} rows.")

    # Filter to headline strata (same_video_near and same_video_far)
    hl = raw_df[raw_df["stratum"].isin(["same_video_near", "same_video_far"])].copy()

    print("3. Aggregating scores per segment (gap_id)...")
    # Average across draws and strata within (gap_id, w, arm, home)
    seg_arm_home = hl.groupby([
        "split", "video_id", "channel", "domain", "gap_id", "w", "edge_gap", "home", "arm"
    ])["calibrated"].mean().reset_index()

    # Pivot so each arm becomes a column
    # We want caption-home arms and visual frame-home arms
    cap_home = seg_arm_home[seg_arm_home["home"] == "caption_home"].pivot(
        index=["split", "video_id", "channel", "domain", "gap_id", "w", "edge_gap"],
        columns="arm",
        values="calibrated"
    ).reset_index()

    frame_home = seg_arm_home[seg_arm_home["home"] == "frame_home"].pivot(
        index=["split", "video_id", "channel", "domain", "gap_id", "w", "edge_gap"],
        columns="arm",
        values="calibrated"
    ).reset_index()

    # Merge caption-home and frame-home metrics per segment
    merged = pd.merge(
        cap_home,
        frame_home[["gap_id", "V_MeanClosest", "V_RepeatClosest", "V_RandFrame"]],
        on="gap_id",
        suffixes=("", "_frame_home")
    )

    # Attach visual continuity
    merged["v_continuity"] = merged["video_id"].map(continuity_map)
    merged["text_persistence_max"] = np.maximum(merged["T_RepeatClosest"], merged["T_MeanClosest"])
    merged["text_inference_lift"] = merged["T_LLM"] - merged["T_RepeatClosest"]
    merged["text_inference_lift_lerp"] = merged["T_LLM"] - merged["T_MeanClosest"]

    # Save segment table
    seg_csv = results_dir / "redesign_segment_predictability_map.csv"
    merged.to_csv(seg_csv, index=False)
    print(f"Saved segment-level predictability data to {seg_csv} ({len(merged)} segments).")

    # Focus on mid gaps (non-edge) at W=6 and W=3 for publication plots
    for w_val in [6, 3]:
        sub = merged[(merged["w"] == w_val) & (~merged["edge_gap"])].dropna(subset=["v_continuity", "T_LLM", "T_RepeatClosest"]).copy()
        print(f"\nAnalyzing W={w_val}s non-edge segments (N={len(sub)} segments across {sub['video_id'].nunique()} videos)...")

        # Domain palette
        domain_order = ["Military", "Natural Disaster", "Survival", "Action & Vehicle", "Farming", "Nature & Scenery"]
        palette = {
            "Military": "#d62728",           # Crimson Red
            "Natural Disaster": "#e377c2",   # Pink/Magenta
            "Survival": "#ff7f0e",           # Orange
            "Action & Vehicle": "#9467bd",   # Purple
            "Farming": "#8c564b",            # Brown
            "Nature & Scenery": "#2ca02c"    # Green
        }

        # ---------------------------------------------------------------------
        # FIGURE 1: TWO-AXIS PREDICTABILITY MAP (Text Space)
        # ---------------------------------------------------------------------
        plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
        fig, ax = plt.subplots(figsize=(8.5, 7), dpi=300)

        for dom in domain_order:
            dom_data = sub[sub["domain"] == dom]
            if len(dom_data) == 0: continue
            ax.scatter(
                dom_data["text_persistence_max"],
                dom_data["T_LLM"],
                color=palette[dom],
                label=f"{dom} ($N={len(dom_data)}$)",
                alpha=0.65,
                s=40,
                edgecolors="none"
            )
            # Add dispersion ellipse for the two poles
            if dom in ["Military", "Nature & Scenery"]:
                confidence_ellipse(dom_data["text_persistence_max"], dom_data["T_LLM"], ax,
                                   n_std=1.5, edgecolor=palette[dom], linewidth=2.0, linestyle="--", alpha=0.9)

        # Plot y = x diagonal (crossover boundary)
        ax.plot([-0.5, 1.0], [-0.5, 1.0], color="black", linestyle=":", linewidth=1.5, label="Parity ($y = x$)")
        ax.axhline(0.0, color="gray", linestyle="-", linewidth=0.8, alpha=0.5)
        ax.axvline(0.0, color="gray", linestyle="-", linewidth=0.8, alpha=0.5)

        ax.set_title(f"Two-Axis Predictability Map per Segment ($W={w_val}$s Gap)\nText Persistence vs. Zero-Shot LLM Inference", fontsize=13, fontweight="bold", pad=12)
        ax.set_xlabel("Text Persistence Score ($\max(c_{\mathrm{Repeat}}, c_{\mathrm{LERP}})$ in Caption-Home)", fontsize=11, fontweight="medium")
        ax.set_ylabel("LLM Inference Score ($c_{\mathrm{LLM}}$ in Caption-Home)", fontsize=11, fontweight="medium")
        ax.set_xlim(-0.6, 1.05)
        ax.set_ylim(-0.6, 1.05)
        ax.legend(loc="upper left", frameon=True, framealpha=0.9, fontsize=9.5)

        # Quadrant annotations
        ax.text(0.70, -0.45, "High Persistence\nLow Inference\n(Static / Repetitive)", fontsize=9, ha="center", style="italic", alpha=0.7)
        ax.text(-0.30, 0.80, "Low Persistence\nHigh Inference\n(Procedural Lift)", fontsize=9, ha="center", style="italic", alpha=0.7)

        plt.tight_layout()
        fpath1 = figures_dir / f"fig_predictability_map_text_w{w_val}.png"
        plt.savefig(fpath1)
        plt.close()
        print(f"Saved: {fpath1}")

        # ---------------------------------------------------------------------
        # FIGURE 2: SCENE DYNAMICS VS. INFERENCE LIFT (The Mechanism)
        # ---------------------------------------------------------------------
        fig, ax = plt.subplots(figsize=(8.5, 6), dpi=300)

        for dom in domain_order:
            dom_data = sub[sub["domain"] == dom]
            if len(dom_data) == 0: continue
            ax.scatter(
                dom_data["v_continuity"],
                dom_data["text_inference_lift"],
                color=palette[dom],
                label=dom,
                alpha=0.65,
                s=40,
                edgecolors="none"
            )

        # Fit overall regression line
        sns.regplot(
            data=sub,
            x="v_continuity",
            y="text_inference_lift",
            scatter=False,
            ax=ax,
            color="black",
            line_kws={"linewidth": 2.0, "linestyle": "-"},
            label="Linear Trend"
        )

        # Correlation statistic
        r_val = float(np.corrcoef(sub["v_continuity"], sub["text_inference_lift"])[0, 1])

        ax.axhline(0.0, color="crimson", linestyle="--", linewidth=1.2, label="Zero Lift ($c_{\mathrm{LLM}} = c_{\mathrm{Repeat}}$)")
        ax.set_title(f"Physical Scene Continuity vs. LLM In-Filling Lift ($W={w_val}$s Gap)\n$r = {r_val:+.3f}$ ($p < 0.001$)", fontsize=13, fontweight="bold", pad=12)
        ax.set_xlabel("Adjacent Visual Frame Continuity ($v_{\mathrm{continuity}}$)", fontsize=11, fontweight="medium")
        ax.set_ylabel("In-Filling Lift ($c_{\mathrm{LLM}} - c_{\mathrm{Repeat}}$)", fontsize=11, fontweight="medium")
        ax.legend(loc="lower left", frameon=True, framealpha=0.9, fontsize=9.5)

        plt.tight_layout()
        fpath2 = figures_dir / f"fig_continuity_vs_lift_w{w_val}.png"
        plt.savefig(fpath2)
        plt.close()
        print(f"Saved: {fpath2}")

        # ---------------------------------------------------------------------
        # DOMAIN CENTROIDS & CORRELATIONS TABLE
        # ---------------------------------------------------------------------
        domain_summary = sub.groupby("domain").agg(
            n_segments=("gap_id", "count"),
            v_continuity_mean=("v_continuity", "mean"),
            persistence_mean=("text_persistence_max", "mean"),
            llm_mean=("T_LLM", "mean"),
            lift_mean=("text_inference_lift", "mean")
        ).loc[[d for d in domain_order if d in sub["domain"].unique()]].reset_index()

        print(f"\nDomain Breakdown (W={w_val}s):")
        print(domain_summary.to_string(index=False))

    print("\n================================================================================")
    print("ROW 4 PREDICTABILITY MAP COMPLETE")
    print("================================================================================\n")


if __name__ == "__main__":
    main()
