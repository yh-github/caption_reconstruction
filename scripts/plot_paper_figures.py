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
    
    # Custom x-tick labels with video counts and significance
    xtick_labels = [
        "Military**\n(19v / 44v)",
        "Disaster\n(14v / 33v)",
        "Survival\n(28v / 61v)",
        "Action†\n(5v / 5v)",
        "Farming\n(22v / 50v)",
        "Nature**\n(10v / 32v)"
    ]
    
    fig, ax = plt.subplots(figsize=(8.8, 4.6))
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
    
    # Intermediate Transition Zone shading
    ax.axvspan(0.5, 4.5, color="#f0f0f0", alpha=0.5, zorder=0)
    ax.text(2.5, 0.12, "Intermediate Transition Zone (n.s., p > 0.20)\nNeither modality significantly diverges from benchmark mean", 
            ha="center", va="center", fontsize=8.5, color="#666666", style="italic",
            bbox=dict(boxstyle="round,pad=0.3", fc="#ffffff", ec="#dddddd", alpha=0.9))
    
    # Pole Annotation arrows
    ax.annotate(
        "← Relative Semantic Sensitivity\n(Test Win: 64.4%, p = 0.0012)",
        xy=(0.0, -0.16),
        xycoords="data",
        fontsize=8.5,
        fontweight="bold",
        color="#2b5c8f",
        ha="center",
        bbox=dict(boxstyle="round,pad=0.3", fc="#e8f0fe", ec="#2b5c8f", lw=1)
    )
    
    ax.annotate(
        "Relative Visual Sensitivity →\n(Test Win: 35.8%, p = 0.0073)",
        xy=(5.0, 0.16),
        xycoords="data",
        fontsize=8.5,
        fontweight="bold",
        color="#b35422",
        ha="center",
        bbox=dict(boxstyle="round,pad=0.3", fc="#fdf2e9", ec="#b35422", lw=1)
    )

    ax.text(
        0.02, 0.96,
        "Poles (Test): Video-level U = 378.0, p = 6.16e-4, Cohen's d = -0.83 (Seg: U = 4112.5, p = 9.91e-6, d = -0.59)\n** Binomial departure from 50% chance parity (p < 0.01); † Insufficient sample size (5 videos)",
        transform=ax.transAxes,
        fontsize=8.0,
        ha="left",
        va="top",
        bbox=dict(boxstyle="round,pad=0.4", fc="#ffffff", ec="#cccccc", lw=1)
    )
    
    ax.set_ylabel("Normalized Rank Delta (Δ / N)")
    ax.set_xlabel("Video Domain / Category (Dev N_vids / Test N_vids)")
    ax.set_title("The Predictability Spectrum: Relative Modality Sensitivity across Domains")
    ax.set_xticks(range(len(xtick_labels)))
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

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.6), gridspec_kw={"width_ratios": [1.1, 0.9]})

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
    ax1.set_title("(A) Temporal Gap Scaling (Wild5 Test)")
    ax1.set_ylabel("Normalized Rank Delta (Δ / N)")
    ax1.set_xlabel("Video Category")
    ax1.set_xticks(range(6))
    ax1.set_xticklabels(["Military", "Disaster", "Survival", "Action†", "Farming", "Nature"], rotation=15)
    ax1.legend(title="Gap Width", frameon=True, loc="upper right")
    ax1.text(0.03, 0.05, "w=3s Pole Separation: Video U=283.0, p=9.69e-6, d=-1.18\n(Segment U=3614.5, p=3.14e-8, d=-0.77)",
             transform=ax1.transAxes, fontsize=7.8, color="#333333",
             bbox=dict(boxstyle="round,pad=0.25", fc="#f8f9fa", ec="#cccccc", lw=0.8))

    # Panel B: Falsification Control (Target Similarity across Baselines)
    conditions = ["LLM Recon", "Copy Near", "Text LERP", "Rand (Within)", "Rand (Cross)"]
    target_sims = [0.7094, 0.7250, 0.7841, 0.5257, 0.5187]
    target_sim_errs = [0.0116, 0.0120, 0.0082, 0.0105, 0.0112]
    
    x = np.arange(len(conditions))
    width = 0.55
    color_sim = "#1f77b4"

    bars = ax2.bar(x, target_sims, width, yerr=target_sim_errs, capsize=4, color=color_sim, alpha=0.85)
    bars[0].set_color("#2ca02c") # Highlight LLM
    bars[3].set_color("#d62728") # Highlight random
    bars[4].set_color("#d62728")

    ax2.set_ylabel("Target Cosine Similarity (Text Space)", color=color_sim)
    ax2.set_xticks(x)
    ax2.set_xticklabels(conditions, rotation=20, ha="right", fontsize=8.5)
    ax2.set_ylim(0.4, 0.88)
    ax2.axhline(0.52, color="gray", linestyle=":", linewidth=0.8, alpha=0.8)

    ax2.set_title("(B) Grounding Control: LLM vs. Random Baselines")
    ax2.text(0.50, 0.16, "LLM achieves significant grounding over\nWithin-Domain Random (0.526, p < 10^-20)\nand Cross-Domain Random (0.519, p < 10^-20).",
             transform=ax2.transAxes, ha="center", va="center", fontsize=8.0, color="#333333",
             bbox=dict(boxstyle="round,pad=0.3", fc="#fff9e6", ec="#d4b106", lw=1))

    plt.tight_layout()
    out_file = os.path.join(OUTPUT_DIR, "fig2_boundary_inertia.png")
    fig.savefig(out_file, dpi=300)
    plt.close(fig)
    print(f"Saved Figure 2 to {out_file}")


if __name__ == "__main__":
    plot_figure_1_predictability_spectrum()
    plot_figure_2_gap_scaling_and_control()
