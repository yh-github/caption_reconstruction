#!/usr/bin/env python
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).parent.parent
CSV_PATH = PROJECT_ROOT / "results" / "wild4_reconstruction_master.csv"
PLOTS_DIR = PROJECT_ROOT / "results" / "plots" / "wild4_comparison"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# Clean publication style with pure matplotlib
plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.titlesize": 14,
    "grid.color": "#e0e0e0",
    "grid.linestyle": "--",
    "grid.alpha": 0.7,
})

def main():
    df = pd.read_csv(CSV_PATH)
    print(f"Loaded {len(df)} rows from {CSV_PATH}")

    methods_to_plot = [
        ("phi-3_v2__t=1.0_rp=1.0", "Phi-3 SLM (Factual v2, t=1.0, rp=1.0)", "#1f77b4", "o", "-"),
        ("phi-3_v2__t=0.1_rp=1.2", "Phi-3 SLM (Greedy v2, t=0.1, rp=1.2)", "#aec7e8", "s", "--"),
        ("vec_cap_mean", "Caption Vector Interpolation (all-mpnet)", "#2ca02c", "^", "-"),
        ("vec_cap_repeat", "Caption Boundary Repeat", "#98df8a", "v", ":"),
        ("vec_vid_mean", "Video Vector Interpolation (ViT)", "#ff7f0e", "D", "-.")
    ]

    widths = [3, 6, 9, 12]

    # -------------------------------------------------------------
    # 1. MRR vs Mask Width (W)
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    ax.grid(True)

    for m_id, label, color, marker, ls in methods_to_plot:
        sub = df[df["method"] == m_id]
        mrr_vals = [sub[sub["width"] == w]["mrr"].mean() for w in widths]
        ax.plot(widths, mrr_vals, label=label, color=color, marker=marker, linestyle=ls, linewidth=2.2, markersize=7)

    ax.set_title("Retrieval MRR vs. Gap Width (W)", fontweight="bold", pad=12)
    ax.set_xlabel("Mask Window Width W (Seconds / Frames)", fontweight="bold")
    ax.set_ylabel("Mean Reciprocal Rank (MRR)", fontweight="bold")
    ax.set_xticks(widths)
    ax.set_ylim(0.18, 0.62)
    ax.legend(loc="upper right", frameon=True, facecolor="white", edgecolor="#cccccc")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "mrr_vs_width.png", dpi=300)
    plt.close()
    print("Saved: mrr_vs_width.png")

    # -------------------------------------------------------------
    # 2. Recall@1 vs Mask Width (Top-1 Accuracy)
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    ax.grid(True)

    for m_id, label, color, marker, ls in methods_to_plot:
        sub = df[df["method"] == m_id]
        r1_vals = [sub[sub["width"] == w]["recall_at_1"].mean() for w in widths]
        ax.plot(widths, r1_vals, label=label, color=color, marker=marker, linestyle=ls, linewidth=2.2, markersize=7)

    ax.set_title("Exact Top-1 Retrieval (Recall@1) vs. Gap Width (W)", fontweight="bold", pad=12)
    ax.set_xlabel("Mask Window Width W (Seconds / Frames)", fontweight="bold")
    ax.set_ylabel("Recall@1 (Top-1 Precision)", fontweight="bold")
    ax.set_xticks(widths)
    ax.set_ylim(0.02, 0.32)
    ax.legend(loc="upper right", frameon=True, facecolor="white", edgecolor="#cccccc")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "recall_at_1_vs_width.png", dpi=300)
    plt.close()
    print("Saved: recall_at_1_vs_width.png")

    # -------------------------------------------------------------
    # 3. Positional Sensitivity (i=0 Opening, i=29 Middle, i=59 Ending)
    # -------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5), sharey=True)
    pos_cases = [
        (0, "Opening Gap (i=0)", axes[0]),
        (29, "Middle Gap (i=29)", axes[1]),
        (59, "Ending Gap (i=59)", axes[2])
    ]

    selected_methods = [
        ("phi-3_v2__t=1.0_rp=1.0", "Phi-3 SLM", "#1f77b4"),
        ("vec_cap_mean", "Caption Vector Interp", "#2ca02c"),
        ("vec_vid_mean", "Video Vector Interp", "#ff7f0e")
    ]

    bar_width = 0.25
    x = np.arange(len(widths))

    for i_pos, title, ax in pos_cases:
        ax.grid(True, axis="y")
        for idx, (m_id, label, color) in enumerate(selected_methods):
            sub = df[(df["method"] == m_id) & (df["index"] == i_pos)]
            mrr_vals = [sub[sub["width"] == w]["mrr"].mean() for w in widths]
            ax.bar(x + idx * bar_width, mrr_vals, width=bar_width, label=label, color=color, alpha=0.9, edgecolor="black", linewidth=0.5)

        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("Mask Width W", fontweight="bold")
        ax.set_xticks(x + bar_width)
        ax.set_xticklabels([f"W={w}" for w in widths])
        if ax == axes[0]:
            ax.set_ylabel("MRR", fontweight="bold")
            ax.legend(loc="upper right", frameon=True, facecolor="white", edgecolor="#cccccc")

    fig.suptitle("MRR across Gap Positions (Opening vs. Middle vs. Ending)", fontweight="bold", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "mrr_by_position.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved: mrr_by_position.png")

    # -------------------------------------------------------------
    # 4. Category Delta: MRR Improvement of Phi-3 over Caption Vector Interpolation
    # -------------------------------------------------------------
    phi_sub = df[df["method"] == "phi-3_v2__t=1.0_rp=1.0"][["video_id", "category", "width", "index", "mrr"]].rename(columns={"mrr": "phi_mrr"})
    vec_sub = df[df["method"] == "vec_cap_mean"][["video_id", "width", "index", "mrr"]].rename(columns={"mrr": "vec_mrr"})
    merged = pd.merge(phi_sub, vec_sub, on=["video_id", "width", "index"])
    merged["mrr_delta"] = merged["phi_mrr"] - merged["vec_mrr"]

    cat_summary = merged.groupby("category")["mrr_delta"].mean().sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.grid(True, axis="y")
    colors = ["#2ca02c" if val >= 0 else "#d62728" for val in cat_summary.values]
    bars = ax.bar(cat_summary.index, cat_summary.values, color=colors, alpha=0.85, edgecolor="black", linewidth=0.7)
    ax.axhline(0, color="black", linestyle="-", linewidth=1.0)
    ax.set_title("Mean MRR Advantage (Phi-3 SLM vs. Vector Interpolation) by Category", fontweight="bold", pad=12)
    ax.set_xlabel("Video Domain / Category", fontweight="bold")
    ax.set_ylabel("Delta MRR (Phi-3 − Vector)", fontweight="bold")
    plt.xticks(rotation=20, ha="right")

    for bar in bars:
        val = bar.get_height()
        va_val = "bottom" if val >= 0 else "top"
        y_offset = 0.003 if val >= 0 else -0.006
        ax.annotate(f"{val:+.3f}", (bar.get_x() + bar.get_width() / 2., val + y_offset),
                    ha='center', va=va_val, fontweight='bold', fontsize=10)

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "category_mrr_delta.png", dpi=300)
    plt.close()
    print("Saved: category_mrr_delta.png")

    print(f"\nAll 4 comparison graphs generated in: {PLOTS_DIR}")

if __name__ == "__main__":
    main()
