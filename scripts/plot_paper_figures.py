#!/usr/bin/env python3
"""
Generate publication-quality figures for the 4-page paper:
Figure 1: The Predictability Spectrum across Video Categories (Wild4 vs Wild5 replication).
Figure 2: Breaking Boundary Inertia: Contrastive Margins (LLM vs Baselines).
"""

from __future__ import annotations
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 14,
    "figure.dpi": 300,
})

OUTPUT_DIR = "results/plots/paper_figures"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def plot_figure_1_predictability_spectrum():
    """Figure 1: Rank Delta (Delta) distribution across categories in Wild4 and Wild5."""
    df = pd.read_csv("results/unified_benchmark_master.csv")
    
    rows = []
    for ds in ["Wild4", "Wild5"]:
        sub = df[(df["dataset"] == ds) & (df["width"] == 6)]
        llama = sub[sub["method"] == "Llama-3.1-8B"][["video_id", "index", "category", "cos_sim"]].rename(columns={"cos_sim": "cos_sim_llama"})
        vis = sub[sub["method"] == "Visual_SigLIP_MeanClosest"][["video_id", "index", "cos_sim"]].rename(columns={"cos_sim": "cos_sim_vis"})
        merged = pd.merge(llama, vis, on=["video_id", "index"])
        merged["rank_text"] = merged["cos_sim_llama"].rank(ascending=False)
        merged["rank_vis"] = merged["cos_sim_vis"].rank(ascending=False)
        # Normalize delta by total N in split to make Wild4 and Wild5 directly comparable on [-1, +1] scale!
        N = len(merged)
        merged["normalized_delta"] = (merged["rank_text"] - merged["rank_vis"]) / N
        merged["split"] = "Dev (Wild4, N=294)" if ds == "Wild4" else "Test (Wild5, N=673)"
        rows.append(merged)
        
    all_data = pd.concat(rows, ignore_index=True)
    
    # Order categories from most Procedural (negative delta) to most Stochastic (positive delta)
    category_order = ["Military", "Natural Disaster", "Survival", "Action & Vehicle", "Farming", "Nature & Scenery"]
    
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    
    palette = {"Dev (Wild4, N=294)": "#4C72B0", "Test (Wild5, N=673)": "#DD8452"}
    
    sns.barplot(
        data=all_data,
        x="category",
        y="normalized_delta",
        hue="split",
        order=category_order,
        palette=palette,
        capsize=0.1,
        err_kws={"linewidth": 1.5},
        ax=ax,
    )
    
    ax.axhline(0, color="gray", linestyle="--", linewidth=1.0, alpha=0.7)
    
    # Annotation arrows
    ax.annotate(
        "← Semantic Dominance\n(LLM Superior)",
        xy=(0.15, -0.18),
        xycoords="data",
        fontsize=10,
        fontweight="bold",
        color="#2b5c8f",
        ha="center",
        bbox=dict(boxstyle="round,pad=0.3", fc="#e8f0fe", ec="#2b5c8f", lw=1)
    )
    
    ax.annotate(
        "Visual Necessity →\n(Vision Superior)",
        xy=(4.85, 0.18),
        xycoords="data",
        fontsize=10,
        fontweight="bold",
        color="#b35422",
        ha="center",
        bbox=dict(boxstyle="round,pad=0.3", fc="#fdf2e9", ec="#b35422", lw=1)
    )

    # Effect size annotation
    ax.text(
        0.50, 0.90,
        "Military vs. Nature: Cohen's d = -0.59 (Test), p = 4.96e-6\nDev: d = -0.84, p = 4.06e-4",
        transform=ax.transAxes,
        fontsize=9,
        ha="center",
        va="top",
        bbox=dict(boxstyle="round,pad=0.4", fc="#ffffff", ec="#cccccc", lw=1)
    )
    
    ax.set_ylabel("Normalized Rank Delta (Δ / N)")
    ax.set_xlabel("Video Domain / Category")
    ax.set_title("The Predictability Spectrum: Semantic Inference vs. Visual Continuity")
    ax.legend(title="Benchmark Split", frameon=True, loc="upper left")
    
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    
    out_file = os.path.join(OUTPUT_DIR, "fig1_predictability_spectrum.png")
    fig.savefig(out_file, dpi=300)
    plt.close(fig)
    print(f"Saved Figure 1 to {out_file}")


def plot_figure_2_boundary_inertia():
    """Figure 2: Contrastive Margin across gap positions comparing LLM vs LERP."""
    df = pd.read_csv("results/contrastive_temporal_retrieval/contrastive_metrics_per_second.csv")
    
    # Map gap relative index (0..5) to normalized temporal position (seconds 1 to 6)
    df["gap_second"] = df["gap_relative_idx"] + 1
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.2), sharey=False)
    
    # Panel A: Text Semantic Space
    mean_llm_c = df.groupby("gap_second")["margin_max_llm_c"].mean()
    sem_llm_c = df.groupby("gap_second")["margin_max_llm_c"].sem()
    mean_lerp_c = df.groupby("gap_second")["margin_max_clerp_c"].mean()
    sem_lerp_c = df.groupby("gap_second")["margin_max_clerp_c"].sem()
    
    seconds = mean_llm_c.index
    
    ax1.plot(seconds, mean_llm_c, marker="o", color="#2ca02c", linewidth=2.0, label="LLM Text Reconstruction")
    ax1.fill_between(seconds, mean_llm_c - sem_llm_c, mean_llm_c + sem_llm_c, color="#2ca02c", alpha=0.15)
    
    ax1.plot(seconds, mean_lerp_c, marker="s", color="#d62728", linestyle="--", linewidth=2.0, label="Text LERP Baseline")
    ax1.fill_between(seconds, mean_lerp_c - sem_lerp_c, mean_lerp_c + sem_lerp_c, color="#d62728", alpha=0.15)
    
    ax1.set_title("(A) Text Semantic Space")
    ax1.set_xlabel("Elapsed Time Inside Gap (seconds)")
    ax1.set_ylabel("Boundary Contrastive Margin (Sim(Tgt) - Sim(Bnd))")
    ax1.legend(loc="lower left", frameon=True)
    ax1.axhline(0, color="black", linestyle=":", linewidth=1.0, alpha=0.6)
    ax1.text(0.98, 0.05, "Higher = Less Boundary Lock\n(Paired Wilcoxon p = 8.3e-18)", transform=ax1.transAxes, ha="right", va="bottom", fontsize=8, color="#555555")
    
    # Panel B: Cross-Modal Video Space
    mean_llm_v = df.groupby("gap_second")["margin_max_llm_v"].mean()
    sem_llm_v = df.groupby("gap_second")["margin_max_llm_v"].sem()
    mean_lerp_v = df.groupby("gap_second")["margin_max_vlerp_v"].mean()
    sem_lerp_v = df.groupby("gap_second")["margin_max_vlerp_v"].sem()
    
    ax2.plot(seconds, mean_llm_v, marker="o", color="#1f77b4", linewidth=2.0, label="LLM Cross-Modal Recon")
    ax2.fill_between(seconds, mean_llm_v - sem_llm_v, mean_llm_v + sem_llm_v, color="#1f77b4", alpha=0.15)
    
    ax2.plot(seconds, mean_lerp_v, marker="s", color="#ff7f0e", linestyle="--", linewidth=2.0, label="Visual LERP Baseline")
    ax2.fill_between(seconds, mean_lerp_v - sem_lerp_v, mean_lerp_v + sem_lerp_v, color="#ff7f0e", alpha=0.15)
    
    ax2.set_title("(B) Cross-Modal Video Frame Space")
    ax2.set_xlabel("Elapsed Time Inside Gap (seconds)")
    ax2.set_ylabel("Boundary Contrastive Margin (Sim(Tgt) - Sim(Bnd))")
    ax2.legend(loc="lower left", frameon=True)
    ax2.axhline(0, color="black", linestyle=":", linewidth=1.0, alpha=0.6)
    ax2.text(0.98, 0.05, "Higher = Less Boundary Lock\n(Paired Wilcoxon p = 9.7e-18)", transform=ax2.transAxes, ha="right", va="bottom", fontsize=8, color="#555555")
    
    plt.suptitle("Mitigating Boundary Anchoring: Contrastive Discrimination Over Time", fontsize=13)
    plt.tight_layout()
    
    out_file = os.path.join(OUTPUT_DIR, "fig2_boundary_inertia.png")
    fig.savefig(out_file, dpi=300)
    plt.close(fig)
    print(f"Saved Figure 2 to {out_file}")


if __name__ == "__main__":
    plot_figure_1_predictability_spectrum()
    plot_figure_2_boundary_inertia()
