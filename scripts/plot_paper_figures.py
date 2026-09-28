#!/usr/bin/env python3
"""
Generate publication-quality figures for the 4-page paper:
Figure 1: The Predictability Spectrum and the Parity Plateau across Categories.
Figure 2: Gap Duration Scaling and Methodological Control (Target Sim vs. Boundary Margin).
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
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9.5,
    "figure.titlesize": 13,
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
        N = len(merged)
        merged["normalized_delta"] = (merged["rank_text"] - merged["rank_vis"]) / N
        merged["split"] = f"Dev (Wild4, N=294)" if ds == "Wild4" else f"Test (Wild5, N=673)"
        rows.append(merged)
        
    all_data = pd.concat(rows, ignore_index=True)
    
    category_order = ["Military", "Natural Disaster", "Survival", "Action & Vehicle", "Farming", "Nature & Scenery"]
    
    # Custom x-tick labels with sample sizes and significance
    xtick_labels = [
        "Military**\n(57 / 132)",
        "Disaster\n(42 / 98)",
        "Survival\n(84 / 183)",
        "Action†\n(15 / 15)",
        "Farming\n(66 / 150)",
        "Nature**\n(30 / 95)"
    ]
    
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
    
    ax.axhline(0, color="black", linestyle="--", linewidth=1.0, alpha=0.7)
    
    # Parity Plateau shading
    ax.axvspan(0.5, 4.5, color="#f0f0f0", alpha=0.5, zorder=0)
    ax.text(2.5, 0.12, "Parity Plateau (n.s., p > 0.20)\nVisual Inertia & Semantic Logic at Equilibrium", 
            ha="center", va="center", fontsize=9, color="#666666", style="italic",
            bbox=dict(boxstyle="round,pad=0.3", fc="#ffffff", ec="#dddddd", alpha=0.9))
    
    # Pole Annotation arrows
    ax.annotate(
        "← Semantic Dominance\n(LLM Superior, p = 0.0012)",
        xy=(0.0, -0.16),
        xycoords="data",
        fontsize=9,
        fontweight="bold",
        color="#2b5c8f",
        ha="center",
        bbox=dict(boxstyle="round,pad=0.3", fc="#e8f0fe", ec="#2b5c8f", lw=1)
    )
    
    ax.annotate(
        "Visual Necessity →\n(Vision Superior, p = 0.0073)",
        xy=(5.0, 0.16),
        xycoords="data",
        fontsize=9,
        fontweight="bold",
        color="#b35422",
        ha="center",
        bbox=dict(boxstyle="round,pad=0.3", fc="#fdf2e9", ec="#b35422", lw=1)
    )

    ax.text(
        0.02, 0.96,
        "Poles: Mann-Whitney U = 4138.0, p = 4.96e-6, Cohen's d = -0.59\n** Binomial departure from 50% parity (p < 0.01); † Small N pilot",
        transform=ax.transAxes,
        fontsize=8.5,
        ha="left",
        va="top",
        bbox=dict(boxstyle="round,pad=0.4", fc="#ffffff", ec="#cccccc", lw=1)
    )
    
    ax.set_ylabel("Normalized Rank Delta (Δ / N)")
    ax.set_xlabel("Video Domain / Category (Dev N / Test N)")
    ax.set_title("The Predictability Spectrum: Semantic Inference vs. Visual Continuity")
    ax.set_xticklabels(xtick_labels)
    ax.legend(title="Benchmark Split", frameon=True, loc="upper right")
    
    plt.tight_layout()
    out_file = os.path.join(OUTPUT_DIR, "fig1_predictability_spectrum.png")
    fig.savefig(out_file, dpi=300)
    plt.close(fig)
    print(f"Saved Figure 1 to {out_file}")


def plot_figure_2_gap_scaling_and_control():
    """Figure 2: Panel A = Gap Scaling (w=3 vs w=6); Panel B = Methodological Control."""
    df = pd.read_csv("results/unified_benchmark_master.csv")
    
    # Panel A data: Test split w=3 vs w=6
    rows = []
    for w in [3, 6]:
        sub = df[(df["dataset"] == "Wild5") & (df["width"] == w)]
        llama = sub[sub["method"] == "Llama-3.1-8B"][["video_id", "index", "category", "cos_sim"]].rename(columns={"cos_sim": "cos_sim_llama"})
        vis = sub[sub["method"] == "Visual_SigLIP_MeanClosest"][["video_id", "index", "cos_sim"]].rename(columns={"cos_sim": "cos_sim_vis"})
        merged = pd.merge(llama, vis, on=["video_id", "index"])
        merged["rank_text"] = merged["cos_sim_llama"].rank(ascending=False)
        merged["rank_vis"] = merged["cos_sim_vis"].rank(ascending=False)
        N = len(merged)
        merged["normalized_delta"] = (merged["rank_text"] - merged["rank_vis"]) / N
        merged["width_label"] = f"w = {w}s"
        rows.append(merged)
    scale_data = pd.concat(rows, ignore_index=True)
    category_order = ["Military", "Natural Disaster", "Survival", "Action & Vehicle", "Farming", "Nature & Scenery"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5), gridspec_kw={"width_ratios": [1.2, 0.8]})

    # Panel A: Gap Duration Scaling
    sns.barplot(
        data=scale_data,
        x="category",
        y="normalized_delta",
        hue="width_label",
        order=category_order,
        palette={"w = 3s": "#2ca02c", "w = 6s": "#9467bd"},
        capsize=0.1,
        err_kws={"linewidth": 1.5},
        ax=ax1,
    )
    ax1.axhline(0, color="black", linestyle="--", linewidth=0.8, alpha=0.7)
    ax1.set_title("(A) Temporal Gap Duration Scaling (Wild5 Test)")
    ax1.set_ylabel("Normalized Rank Delta (Δ / N)")
    ax1.set_xlabel("Video Category")
    ax1.set_xticklabels(["Military", "Disaster", "Survival", "Action", "Farming", "Nature"], rotation=15)
    ax1.legend(title="Gap Width", frameon=True, loc="upper right")

    # Panel B: Falsification Control (Target Similarity vs Boundary Margin)
    # Empirically measured values from our Random Distractor Control experiment
    conditions = ["LLM Recon", "Text LERP", "Random Control"]
    target_sims = [0.7094, 0.7841, 0.5516]
    target_sim_errs = [0.1165 / np.sqrt(100), 0.0819 / np.sqrt(100), 0.1213 / np.sqrt(100)]
    margins = [-0.0533, -0.1452, -0.0335]
    
    x = np.arange(len(conditions))
    width = 0.35

    color_sim = "#1f77b4"
    color_margin = "#d62728"

    ax2_twin = ax2.twinx()

    rects1 = ax2.bar(x - width/2, target_sims, width, yerr=target_sim_errs, capsize=4, label="Target Similarity (Higher=Better)", color=color_sim, alpha=0.85)
    rects2 = ax2_twin.bar(x + width/2, margins, width, label="Boundary Margin (Tautological)", color=color_margin, alpha=0.85)

    ax2.set_ylabel("Target Cosine Similarity", color=color_sim)
    ax2_twin.set_ylabel("Boundary Margin (Sim(Tgt) - Sim(Bnd))", color=color_margin)
    ax2.set_xticks(x)
    ax2.set_xticklabels(conditions, rotation=15)
    ax2.set_ylim(0.4, 0.9)
    ax2_twin.set_ylim(-0.20, 0.02)
    ax2_twin.axhline(0, color="gray", linestyle=":", linewidth=0.8)

    ax2.set_title("(B) Methodological Control: LERP Geometry")
    ax2.text(0.50, 0.15, "Random Control beats LERP on margin\ndue to distance from boundary line segment,\nbut LLM maintains high target similarity.",
             transform=ax2.transAxes, ha="center", va="center", fontsize=8, color="#333333",
             bbox=dict(boxstyle="round,pad=0.3", fc="#fff9e6", ec="#d4b106", lw=1))

    plt.tight_layout()
    out_file = os.path.join(OUTPUT_DIR, "fig2_boundary_inertia.png")
    fig.savefig(out_file, dpi=300)
    plt.close(fig)
    print(f"Saved Figure 2 to {out_file}")


if __name__ == "__main__":
    plot_figure_1_predictability_spectrum()
    plot_figure_2_gap_scaling_and_control()
