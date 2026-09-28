import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
results_dir = REPO_ROOT / "results"
figures_dir = REPO_ROOT / "docs" / "paper" / "figures"
figures_dir.mkdir(parents=True, exist_ok=True)

# 1. Plot Headroom Curve
headroom_df = pd.read_csv(results_dir / "redesign_persistence_headroom.csv")

plt.figure(figsize=(7, 4.5), dpi=300)
w = headroom_df["width"]
mean_hr = headroom_df["mean_headroom"]
ci_l = headroom_df["ci_lower"]
ci_u = headroom_df["ci_upper"]

plt.plot(w, mean_hr, marker="o", color="#1f77b4", linewidth=2.5, label="Mean Text Headroom ($c_{Oracle} - \max(c_{Rep}, c_{Mean})$)")
plt.fill_between(w, ci_l, ci_u, color="#1f77b4", alpha=0.2, label="95% Stratified Bootstrap CI")
plt.axhline(0.10, color="crimson", linestyle="--", linewidth=1.5, label="Pre-registered Go/No-Go Gate (0.10)")

plt.title("Text Headroom Curve across Gap Widths ($w=1..30$s)", fontsize=13, fontweight="bold", pad=12)
plt.xlabel("Masked Gap Duration $w$ (seconds)", fontsize=11)
plt.ylabel("Available Headroom (Calibrated Units)", fontsize=11)
plt.ylim(0.0, 1.0)
plt.grid(True, linestyle=":", alpha=0.6)
plt.legend(loc="lower right", fontsize=9.5)
plt.tight_layout()
plt.savefig(figures_dir / "fig_redesign_headroom_curve.png")
plt.close()
print("Saved headroom curve plot to docs/paper/figures/fig_redesign_headroom_curve.png")

# 2. Plot Headline Calibrated Retrieval (W=3 and W=6 in Caption-Home)
summary_df = pd.read_csv(results_dir / "redesign_shared_target_summary.csv")
cap_home = summary_df[summary_df["home"] == "caption_home"].copy()

# Focus on key arms
arms_order = ["T_Oracle", "T_MeanClosest", "T_RepeatClosest", "T_LLM_gapmean", "T_LLM", "T_RandWithinDomain", "T_RandCorpus"]
labels = ["Oracle", "Text LERP", "Text Repeat", "LLM (GapMean)", "LLM (Per-Sec)", "Rand (Domain)", "Rand (Corpus)"]
colors = ["#2ca02c", "#1f77b4", "#aec7e8", "#ff7f0e", "#d62728", "#7f7f7f", "#c7c7c7"]

w3_scores = [cap_home[(cap_home["w"] == 3) & (cap_home["arm"] == a)]["mean_calibrated"].values[0] for a in arms_order]
w6_scores = [cap_home[(cap_home["w"] == 6) & (cap_home["arm"] == a)]["mean_calibrated"].values[0] for a in arms_order]

x = np.arange(len(arms_order))
width = 0.35

plt.figure(figsize=(9, 4.5), dpi=300)
plt.bar(x - width/2, w3_scores, width, label="Gap $W=3$s", color="#3470a3", edgecolor="black", linewidth=0.8)
plt.bar(x + width/2, w6_scores, width, label="Gap $W=6$s", color="#f28e2b", edgecolor="black", linewidth=0.8)

plt.axhline(0.0, color="black", linestyle="-", linewidth=0.8)
plt.ylabel("Headline Calibrated Score ($c = 2 \cdot \mathrm{AUC} - 1$)", fontsize=11)
plt.title("Shared-Target Retrieval Performance in Caption-Home (Benchmark $N=323$ Videos)", fontsize=12, fontweight="bold", pad=12)
plt.xticks(x, labels, rotation=25, ha="right", fontsize=10)
plt.ylim(-0.15, 1.05)
plt.grid(axis="y", linestyle=":", alpha=0.6)
plt.legend(loc="upper right", fontsize=10)
plt.tight_layout()
plt.savefig(figures_dir / "fig_redesign_caption_home_arms.png")
plt.close()
print("Saved arms plot to docs/paper/figures/fig_redesign_caption_home_arms.png")
