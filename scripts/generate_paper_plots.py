#!/usr/bin/env python3
"""
Generate publication-quality figures for the ICCV Workshop Short Paper.

Outputs saved to docs/paper/LaTeX/figures/:
1. fig1_surprisal_immunity.png: APCS Surprisal vs Phi-3 vs Llama-3.1 MRR (dual regression).
2. fig2_modality_decoupling.png: Visual Motion Variance vs Visual Continuity vs Llama MRR.
3. fig3_category_performance.png: Grouped MRR comparison across video domains.
4. fig4_positional_mrr.png: Positional MRR across Opening, Middle, and Ending segments.
"""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import seaborn as sns

# Publication style
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans', 'Helvetica', 'Arial'],
    'font.size': 9,
    'axes.labelsize': 9.5,
    'axes.titlesize': 10,
    'xtick.labelsize': 8.5,
    'ytick.labelsize': 8.5,
    'legend.fontsize': 8.0,
    'figure.titlesize': 11,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.edgecolor': '#333333',
    'axes.linewidth': 0.8,
    'grid.color': '#cccccc',
    'grid.linestyle': '--',
    'grid.linewidth': 0.5,
    'grid.alpha': 0.7,
})

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = REPO_ROOT / 'results' / 'prior_vs_post_reconstruction_master.csv'
OUT_DIR = REPO_ROOT / 'docs' / 'paper' / 'LaTeX' / 'figures'
OUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA_PATH)
print(f"Loaded {len(df)} records from {DATA_PATH}")

c_phi = '#d62728'    # Crimson
c_llama = '#1f77b4'  # Navy / Deep Blue
c_vis = '#2ca02c'    # Forest Green

# -------------------------------------------------------------
# Figure 1: Caption Surprisal Immunity (Phi-3 vs Llama-3.1)
# -------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.8, 2.7))

# Plot Phi-3
sns.regplot(
    data=df, x='apcs_nll', y='phi_mrr', ax=ax1,
    color=c_phi,
    scatter_kws={'alpha': 0.65, 's': 28, 'edgecolor': 'none'},
    line_kws={'linewidth': 1.8}
)
r_phi, p_phi = stats.pearsonr(df['apcs_nll'], df['phi_mrr'])
ax1.set_title(r"$\mathbf{Phi\text{-}3\text{-}mini\ (3.8B)}$: Surprisal Collapse", fontsize=9.5)
ax1.set_xlabel("A Priori Caption Surprisal (APCS NLL)")
ax1.set_ylabel("Reconstruction MRR")
ax1.grid(True)
ax1.text(
    0.05, 0.12,
    f"$r = {r_phi:.3f}$\n$p < 0.0001$",
    transform=ax1.transAxes,
    fontsize=8.5,
    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='#cccccc', alpha=0.9)
)

# Plot Llama-3.1
sns.regplot(
    data=df, x='apcs_nll', y='llama_mrr', ax=ax2,
    color=c_llama,
    scatter_kws={'alpha': 0.65, 's': 28, 'edgecolor': 'none'},
    line_kws={'linewidth': 1.8}
)
r_llama, p_llama = stats.pearsonr(df['apcs_nll'], df['llama_mrr'])
ax2.set_title(r"$\mathbf{Llama\text{-}3.1\ (8B)}$: Surprisal Immunity", fontsize=9.5)
ax2.set_xlabel("A Priori Caption Surprisal (APCS NLL)")
ax2.set_ylabel("Reconstruction MRR")
ax2.grid(True)
ax2.text(
    0.05, 0.12,
    f"$r = {r_llama:+.3f}$\n$p = {p_llama:.2f}$ (n.s.)",
    transform=ax2.transAxes,
    fontsize=8.5,
    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='#cccccc', alpha=0.9)
)

sns.despine(fig)
plt.tight_layout()
fig1_path = OUT_DIR / 'fig1_surprisal_immunity.png'
plt.savefig(fig1_path)
plt.close()
print(f"Saved: {fig1_path}")

# -------------------------------------------------------------
# Figure 2: The Modality Decoupling Law
# -------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.8, 2.7))

# Panel A: Visual Continuity vs Motion Variance
sns.regplot(
    data=df, x='video_surprisal_var', y='video_cos_sim_min', ax=ax1,
    color=c_vis,
    scatter_kws={'alpha': 0.65, 's': 28, 'edgecolor': 'none'},
    line_kws={'linewidth': 1.8}
)
r_v_vis, p_v_vis = stats.pearsonr(df['video_surprisal_var'], df['video_cos_sim_min'])
ax1.set_title(r"$\mathbf{Visual\ Vector\ Continuity}$ ($\cos\text{Sim}_{\min}$)", fontsize=9.5)
ax1.set_xlabel(r"Video Optical Motion Variance ($\sigma^2_{\Delta v}$)")
ax1.set_ylabel(r"Vector Cosine Similarity (Min)")
ax1.grid(True)
ax1.text(
    0.50, 0.78,
    f"$r = {r_v_vis:.3f}$\n$p < 0.0001$",
    transform=ax1.transAxes,
    fontsize=8.5,
    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='#cccccc', alpha=0.9)
)

# Panel B: LLM Logic vs Motion Variance
sns.regplot(
    data=df, x='video_surprisal_var', y='llama_mrr', ax=ax2,
    color=c_llama,
    scatter_kws={'alpha': 0.65, 's': 28, 'edgecolor': 'none'},
    line_kws={'linewidth': 1.8}
)
r_v_llm, p_v_llm = stats.pearsonr(df['video_surprisal_var'], df['llama_mrr'])
ax2.set_title(r"$\mathbf{LLM\ Causal\ Reasoning}$ (Llama-3.1 MRR)", fontsize=9.5)
ax2.set_xlabel(r"Video Optical Motion Variance ($\sigma^2_{\Delta v}$)")
ax2.set_ylabel("Reconstruction MRR")
ax2.grid(True)
ax2.text(
    0.50, 0.78,
    f"$r = {r_v_llm:.3f}$\n$p = {p_v_llm:.2f}$ (n.s.)",
    transform=ax2.transAxes,
    fontsize=8.5,
    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='#cccccc', alpha=0.9)
)

sns.despine(fig)
plt.tight_layout()
fig2_path = OUT_DIR / 'fig2_modality_decoupling.png'
plt.savefig(fig2_path)
plt.close()
print(f"Saved: {fig2_path}")

# -------------------------------------------------------------
# Figure 3: Performance Across Video Domains
# -------------------------------------------------------------
cat_order = ['Action/Vehicle', 'Scenery', 'Military', 'Survival', 'Farming', 'Nature/Doc']
counts = df['category'].value_counts().to_dict()
cat_labels = [f"{cat}\n(n={counts.get(cat, 0)})" for cat in cat_order]

cat_df = df.groupby('category').agg(
    llama_mrr=('llama_mrr', 'mean'),
    phi_mrr=('phi_mrr', 'mean'),
    video_mrr=('video_mrr', 'mean'),
    llama_sem=('llama_mrr', 'sem'),
    phi_sem=('phi_mrr', 'sem'),
    video_sem=('video_mrr', 'sem')
).reindex(cat_order).reset_index()

fig, ax = plt.subplots(figsize=(6.8, 3.0))
x = np.arange(len(cat_order))
bar_width = 0.25

rects1 = ax.bar(x - bar_width, cat_df['llama_mrr'], bar_width, yerr=cat_df['llama_sem'],
                label='Llama-3.1-8B (Whole Window)', color=c_llama, capsize=3, edgecolor='black', linewidth=0.5)
rects2 = ax.bar(x, cat_df['phi_mrr'], bar_width, yerr=cat_df['phi_sem'],
                label='Phi-3-mini-4k (Iterative)', color=c_phi, capsize=3, edgecolor='black', linewidth=0.5)
rects3 = ax.bar(x + bar_width, cat_df['video_mrr'], bar_width, yerr=cat_df['video_sem'],
                label='Visual Vector Continuity', color=c_vis, capsize=3, edgecolor='black', linewidth=0.5)

ax.set_ylabel('Mean Reciprocal Rank (MRR)')
ax.set_title('Reconstruction Performance Across Diverse Video Domains (W=3)', fontsize=10.5)
ax.set_xticks(x)
ax.set_xticklabels(cat_labels, fontsize=8.5)
ax.legend(loc='upper right', frameon=True, framealpha=0.9)
ax.grid(True, axis='y')
ax.set_ylim(0, 0.75)

sns.despine(fig)
plt.tight_layout()
fig3_path = OUT_DIR / 'fig3_category_performance.png'
plt.savefig(fig3_path)
plt.close()
print(f"Saved: {fig3_path}")

# -------------------------------------------------------------
# Figure 4: Positional Dynamics (Opening vs Middle vs Ending)
# -------------------------------------------------------------
fig, ax = plt.subplots(figsize=(3.4, 2.7))

pos_data = [
    df['llama_mrr_start'].dropna(),
    df['llama_mrr_middle'].dropna(),
    df['llama_mrr_end'].dropna()
]
pos_means = [d.mean() for d in pos_data]
pos_sems = [d.sem() for d in pos_data]
pos_labels = ['Opening\n(i=0)', 'Middle\n(i=mid)', 'Ending\n(i=end)']

bars = ax.bar(pos_labels, pos_means, yerr=pos_sems, color=['#1f77b4', '#4b97c9', '#7cbbe0'],
              capsize=4, edgecolor='black', linewidth=0.6, width=0.55)

for bar, mean_val in zip(bars, pos_means):
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2.0, yval + 0.025, f"{mean_val:.3f}",
            ha='center', va='bottom', fontsize=8, fontweight='bold')

ax.set_ylabel('Llama-3.1-8B MRR')
ax.set_title('Positional Gap Reconstruction', fontsize=9.5)
ax.set_ylim(0, 0.72)
ax.grid(True, axis='y')
sns.despine(fig)
plt.tight_layout()
fig4_path = OUT_DIR / 'fig4_positional_mrr.png'
plt.savefig(fig4_path)
plt.close()
print(f"Saved: {fig4_path}")

print("All figures successfully regenerated!")
