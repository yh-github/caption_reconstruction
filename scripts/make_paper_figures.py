#!/usr/bin/env python3
"""
Figures for docs/paper/captions_vs_frames.md, from existing outputs (no models run).

fig1_lag:            (a) caption-to-frame lag profile, (b) near-pool score vs. caption shift
                     <- results/caption_audit/lag_profiles.csv, lag_near_pool.csv (scripts/caption_lag_robustness.py)
fig2_topic_timing:   calibrated c by distractor pool, frame space
                     <- results/redesign_shared_target.csv (scripts/eval_shared_target.py)
fig3_forced_choice:  (a) copy baselines by width, (b) Llama vs. copy on the items it scored
                     <- results/forced_choice/forced_choice_merged.csv (scripts/forced_choice_gap.py report)
fig4_grounding:      per-video caption grounding vs. caption copy's deficit to frame copy
                     <- results/caption_audit/per_video.csv (scripts/caption_vs_siglip_audit.py)

All CIs are 95% channel-cluster bootstrap. Colors follow one fixed slot per arm across figures.

Usage (from repo root):
    .venv/bin/python scripts/make_paper_figures.py
Outputs: docs/paper/figures/fig{1..4}_*.{pdf,png}
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

OUT = Path("docs/paper/figures")
AUDIT = Path("results/caption_audit")
N_BOOT = 2000
rng = np.random.default_rng(7)

# Fixed slot per arm (reference categorical palette, light mode, slots 1-4). Validated with the dataviz
# validator; aqua and yellow are below 3:1 on white, so every line is directly labeled.
COLOR = {"oracle": "#2a78d6", "caption_copy": "#eb6834", "llama": "#1baf7a", "frame_copy": "#eda100"}
LABEL = {"oracle": "True caption", "caption_copy": "Caption copy", "llama": "Llama-3.1-8B", "frame_copy": "Frame copy"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "axes.grid.axis": "y",
    "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True, "lines.linewidth": 2,
    "savefig.dpi": 300, "savefig.bbox": "tight", "font.family": "DejaVu Sans",
})


def boot_ci(df, stat, cluster="chan"):
    """95% CI of stat(df) from resampling clusters with replacement."""
    by = [g for _, g in df.groupby(cluster)]
    vals = [stat(pd.concat([by[i] for i in rng.integers(0, len(by), len(by))])) for _ in range(N_BOOT)]
    return np.nanpercentile(vals, [2.5, 97.5])


def end_label(ax, x, y, arm, dy=0.0, ha="left"):
    ax.annotate(LABEL[arm], (x, y), xytext=(4 if ha == "left" else -4, dy), textcoords="offset points",
                ha=ha, va="center", fontsize=7.5, color=INK)


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{name}.{ext}")
    plt.close(fig)
    print(f"saved {OUT / name}.{{pdf,png}}")


def fig1():
    prof = pd.read_csv(AUDIT / "lag_profiles.csv")
    prof = prof[prof.segment == "all"]
    lags = [str(k) for k in range(-5, 6)]
    mean = prof[lags].mean()
    ci = np.array([boot_ci(prof, lambda g, k=k: g[k].mean()) for k in lags])
    x = np.arange(-5, 6)

    near = pd.read_csv(AUDIT / "lag_near_pool.csv")
    pv = near.groupby(["method", "shift", "vid", "chan"], as_index=False).c.mean()

    fig, (a, b) = plt.subplots(1, 2, figsize=(6.6, 2.4), gridspec_kw=dict(wspace=0.45))
    a.fill_between(x, ci[:, 0], ci[:, 1], color=COLOR["oracle"], alpha=0.18, linewidth=0)
    a.plot(x, mean.values, color=COLOR["oracle"], marker="o", markersize=4)
    a.axvline(0, color=INK2, linewidth=0.8, linestyle=":")
    a.annotate("caption t matches\nframe t+1 to t+2 best", (1.5, mean.max()), xytext=(0.6, 0.2),
               fontsize=7, color=INK2, arrowprops=dict(arrowstyle="-", color=INK2, linewidth=0.6))
    a.set_xticks(x)
    a.set_xlabel("frame offset k (s)")
    a.set_ylabel("sim(caption t, frame t+k), z")
    a.set_title("(a) Captions run 1–2 s ahead of the frames", loc="left")

    for arm in ["oracle", "caption_copy", "llama"]:
        s = pv[pv.method == arm].groupby("shift").c.mean()
        lo_hi = np.array([boot_ci(pv[(pv.method == arm) & (pv["shift"] == k)], lambda g: g.c.mean()) for k in s.index])
        b.errorbar(s.index, s.values, yerr=[s.values - lo_hi[:, 0], lo_hi[:, 1] - s.values], color=COLOR[arm],
                   marker="o", markersize=4, capsize=0, elinewidth=1)
        end_label(b, s.index[-1], s.values[-1], arm, dy={"oracle": 4, "caption_copy": -4, "llama": 0}[arm])
    fc = pv[(pv.method == "frame_copy") & (pv["shift"] == 0)].c.mean()
    b.axhline(fc, color=COLOR["frame_copy"], linewidth=2, linestyle="--")
    b.annotate(f"{LABEL['frame_copy']} ({fc:.2f})", (0, fc), xytext=(0, -9), textcoords="offset points",
               fontsize=7.5, color=INK)
    b.axhline(0, color=INK2, linewidth=0.6)
    b.set_xticks([0, 1, 2, 3])
    b.set_xticklabels(["0", "+1", "+2", "+3"])
    b.set_xlim(-0.3, 4.6)
    b.set_ylim(-0.05, 0.6)
    b.set_xlabel("caption shift s (s)")
    b.set_ylabel("calibrated c vs. frames within ±10 s")
    b.set_title("(b) Correcting the lead helps only true captions", loc="left")
    save(fig, "fig1_lag")


def fig2():
    df = pd.read_csv("results/redesign_shared_target.csv",
                     usecols=["video_id", "channel", "edge_gap", "arm", "home", "stratum", "calibrated"])
    df = df[(~df.edge_gap) & (df.home == "frame_home")]
    arms = {"T_Oracle": "oracle", "T_RepeatClosest": "caption_copy", "T_LLM": "llama", "V_RepeatClosest": "frame_copy"}
    pools = ["same_video_near", "same_video_far", "other_video_same_channel", "other_video_other_channel"]
    names = ["same video\n±10 s", "same video\nfar", "other video,\nsame channel", "other\nchannel"]
    df = df[df.arm.isin(arms) & df.stratum.isin(pools)]
    pv = df.groupby(["arm", "stratum", "video_id", "channel"], as_index=False).calibrated.mean()
    pv = pv.rename(columns={"channel": "chan"})

    fig, ax = plt.subplots(figsize=(4.2, 2.7))
    x = np.arange(len(pools))
    # True caption and caption copy nearly coincide; a small horizontal dodge keeps both visible.
    dodge = {"oracle": -0.06, "caption_copy": 0.06, "llama": 0.0, "frame_copy": 0.0}
    for arm_key, arm in arms.items():
        sub = pv[pv.arm == arm_key]
        m = sub.groupby("stratum").calibrated.mean()[pools].values
        ci = np.array([boot_ci(sub[sub.stratum == p], lambda g: g.calibrated.mean()) for p in pools])
        ax.errorbar(x + dodge[arm], m, yerr=[m - ci[:, 0], ci[:, 1] - m], color=COLOR[arm], marker="o",
                    markersize=4.5, capsize=0, elinewidth=1)
        end_label(ax, x[-1] + 0.06, m[-1], arm, dy={"oracle": -5, "caption_copy": 5, "llama": 0, "frame_copy": 0}[arm])
    ax.axhline(0, color=INK2, linewidth=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_xlim(-0.3, 4.1)
    ax.set_ylim(-0.05, 1.02)
    ax.set_ylabel("calibrated c in frame space (0 = chance)")
    ax.set_xlabel("distractor frames", labelpad=4)
    ax.set_title("Captions keep the topic but not the second", loc="left")
    save(fig, "fig2_topic_timing")


def fig3():
    d = pd.read_csv("results/forced_choice/forced_choice_merged.csv")
    widths = sorted(d.w.unique())
    fig, (a, b) = plt.subplots(1, 2, figsize=(6.6, 2.4), gridspec_kw=dict(wspace=0.45, width_ratios=[1.5, 1]))
    for arm, col in [("frame_copy", "vis_copy_ok"), ("caption_copy", "text_copy_ok")]:
        m = d.groupby("w")[col].mean()[widths].values
        ci = np.array([boot_ci(d[d.w == w], lambda g: g[col].mean()) for w in widths])
        a.errorbar(widths, m, yerr=[m - ci[:, 0], ci[:, 1] - m], color=COLOR[arm], marker="o", markersize=4,
                   capsize=0, elinewidth=1)
        end_label(a, widths[-1], m[-1], arm)
    a.axhline(0.25, color=INK2, linewidth=0.8, linestyle=":")
    a.annotate("chance", (widths[0], 0.25), xytext=(0, 3), textcoords="offset points", fontsize=7, color=INK2)
    a.set_xscale("log", base=2)
    a.set_xticks(widths)
    a.set_xticklabels([str(w) for w in widths])
    a.set_xlim(widths[0] * 0.85, widths[-1] * 2.6)
    a.set_ylim(0, 1)
    a.set_xlabel("gap width W (s)")
    a.set_ylabel("accuracy, 4-way choice")
    a.set_title(f"(a) Copy baselines, all {len(d)} items", loc="left")

    s = d[d["llama-3.1-8b_pmi_ok"].notna()]
    rows = [("llama", "llama-3.1-8b_pmi_ok"), ("caption_copy", "text_copy_ok"), ("frame_copy", "vis_copy_ok")]
    for i, (arm, col) in enumerate(rows):
        m = s[col].mean()
        lo, hi = boot_ci(s, lambda g: g[col].mean())
        b.plot([lo, hi], [i, i], color=COLOR[arm], linewidth=1.2)
        b.plot(m, i, "o", color=COLOR[arm], markersize=6)
        b.annotate(f"{m:.0%}", (hi, i), xytext=(4, 0), textcoords="offset points", va="center", fontsize=7.5)
    b.axvline(0.25, color=INK2, linewidth=0.8, linestyle=":")
    b.set_yticks(range(len(rows)))
    b.set_yticklabels([LABEL[a] for a, _ in rows])
    b.invert_yaxis()
    b.set_xlim(0, 1.05)
    b.grid(axis="x", color=GRID, linewidth=0.6)
    b.grid(axis="y", visible=False)
    b.set_xlabel("accuracy")
    b.set_title(f"(b) Same {len(s)} items, with Llama", loc="left")
    save(fig, "fig3_forced_choice")


def fig4():
    pv = pd.read_csv(AUDIT / "per_video.csv").dropna(subset=["q1_mrr", "rank_capcopy_minus_viscopy"])
    x, y = pv.q1_mrr, pv.rank_capcopy_minus_viscopy
    rho = spearmanr(x, y)[0]
    lo, hi = boot_ci(pv, lambda g: spearmanr(g.q1_mrr, g.rank_capcopy_minus_viscopy)[0])
    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    ax.scatter(x, y, s=12, color=COLOR["oracle"], alpha=0.55, linewidths=0)
    chance = sum(1 / r for r in range(1, 61)) / 60
    ax.axvline(chance, color=INK2, linewidth=0.8, linestyle=":")
    ax.annotate("chance", (chance, y.min()), xytext=(-3, 0), textcoords="offset points", fontsize=7,
                color=INK2, ha="right", va="bottom")
    ax.axhline(0, color=INK2, linewidth=0.6)
    ax.set_xlabel("caption grounding (MRR, caption t → frame t)")
    ax.set_ylabel("caption copy rank − frame copy rank")
    ax.set_title(f"Better-grounded captions trail frames less\nSpearman ρ = {rho:.2f} [{lo:.2f}, {hi:.2f}], "
                 f"{len(pv)} videos", loc="left")
    ax.grid(axis="x", color=GRID, linewidth=0.6)
    save(fig, "fig4_grounding")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    fig1()
    fig2()
    fig3()
    fig4()


if __name__ == "__main__":
    main()
