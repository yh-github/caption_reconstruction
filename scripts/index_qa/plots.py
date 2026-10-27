#!/usr/bin/env python3
"""Paper figures for the caption index (docs/paper/caption_index_draft.md).

    .venv/bin/python scripts/index_qa/plots.py cost    # Figure 1: accuracy vs. list-price $ per question
    .venv/bin/python scripts/index_qa/plots.py types   # Figure 2: accuracy by video type
    .venv/bin/python scripts/index_qa/plots.py readers # Figure 3: accuracy by reader (claim 2a)

Figure 1 has one panel per number of questions per video Q, with the one-time captioning cost spread over Q
questions (Q = inf: the per-question cost alone). Costs are computed as in reader.py cost (list prices from the
logged tokens; captioning from Gemini's call log or the Claude captioner's caption files). Writes PNG and PDF to
docs/paper/figures/.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from index_qa import reader as R  # noqa: E402
from index_qa import tempcompass as tc  # noqa: E402
from index_qa.caption import caption_windows  # noqa: E402

FIG = tc.ROOT / "docs/paper/figures"
# Categorical slots 1-2 of the dataviz skill's validated default palette (light mode), one per reader
READER_COLOR = {"haiku": "#2a78d6", "sonnet": "#eb6834"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
# Figs. 2-3: a hue per model:arm from the default palette's slots 3-7, so that no hue also means a reader (Fig. 1).
# Validated in this order as adjacent bars (worst adjacent CVD ΔE 9.1); Fig. 3 uses the two caption hues
ARM_COLOR = {"haiku:frames5": "#1baf7a", "haiku:selfcaptions": "#eda100", "haiku:sonnetcaptions": "#6250d6",
             "haiku:captions": "#008300", "sonnet:frames5": "#e87ba4"}
LABEL = {"blind": "question only", "frame1": "1 frame", "frames5": "5 frames", "frames5_448": "5 frames 448 px",
         "describe5": "describe, answer", "selfcaptions": "self-captions", "sonnetcaptions": "Sonnet captions",
         "frameclipcaptions": "Gemini frame-clip capt.", "framecaptions": "Gemini frame capt.",
         "captions": "Gemini video capt.", "sonnetcaptions448": "Sonnet captions 448 px",
         "sonnetwhole": "Sonnet whole-video capt.", "sonnetaspects": "Sonnet aspect-aware capt."}
# Caption variants left out of Fig. 1 (robustness checks, in Table 1): C9's 448 px, C10's whole-video and C11's
# aspect-aware captions
FIG1_SKIP = {"sonnetcaptions448", "sonnetwhole", "sonnetaspects"}
# Label offsets (points) in the Q = inf panel, where the caption arms crowd at one cost; Sonnet labels go left
OFFSET = {("haiku", "captions"): (5, 4), ("haiku", "frameclipcaptions"): (5, 0), ("haiku", "sonnetcaptions"): (5, -11),
          ("haiku", "selfcaptions"): (5, -6), ("sonnet", "frames5"): (-6, 4)}


def arm_costs():
    """Per model:arm on the confirmatory questions: accuracy, mean list $ per question, and the mean one-time
    captioning $ per video (0 for frame arms)."""
    import pandas as pd
    qs = tc.items("confirm")
    vid = {q["item"]: q["video_id"] for q in qs}
    rows = [r for m in R.PRICES for r in R.latest(m).values() if r["question_item"] in vid]
    df = pd.DataFrame(rows)
    df["arm"] = df.model + ":" + df.arm
    df["usd"] = [R.list_usd(r["model"], r["usage"]) for r in rows]
    out = []
    for arm, d in df.groupby("arm"):
        model, a = arm.split(":")
        index = 0.0
        if a in R.INDEX_SOURCE:
            per_win = R.index_usd(a)
            per_vid = []
            for v in {vid[q] for q in d.question_item}:
                w = [per_win.get((v, p)) for p in range(len(caption_windows(v, R.CAPTIONERS[a])))]
                if all(c is not None for c in w):
                    per_vid.append(sum(w))
            index = sum(per_vid) / len(per_vid) if per_vid else float("nan")
        out.append(dict(model=model, arm=a, n=len(d), acc=d.correct.mean(), usd=d.usd.mean(), index=index))
    return pd.DataFrame(out)


def style(ax):
    ax.set_facecolor("white")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def cmd_cost(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    t = arm_costs()
    t = t[(t.n >= 290) & ~t.arm.isin(FIG1_SKIP)]  # arms with (nearly) every confirmatory question
    print(t.round(6).to_string(index=False))
    qs = [1, 10, None]
    fig, axes = plt.subplots(1, 3, figsize=(11, 4.2), sharey=True, gridspec_kw={"width_ratios": [1, 1, 1.8]})
    for ax, Q in zip(axes, qs):
        style(ax)
        x = t.usd + (t["index"] / Q if Q else 0)
        for (_, r), xi in zip(t.iterrows(), x):
            cap = r.arm in R.CAPTIONERS
            ax.scatter(xi, r.acc, s=42, marker="s" if cap else "o", color=READER_COLOR[r.model],
                       edgecolor="white", linewidth=1.5, zorder=3)
            if Q is None:
                left = r.model == "sonnet"  # the right edge of the panel: label to the left
                dx, dy = OFFSET.get((r.model, r.arm), (-6, -9) if left else (5, -3))
                ax.annotate(("Sonnet, " if left else "") + LABEL.get(r.arm, r.arm), (xi, r.acc),
                            xytext=(dx, dy), textcoords="offset points", fontsize=6.5, color=INK,
                            ha="right" if left else "left")
        ax.set_xscale("log")
        ax.set_title(f"Q = {Q} question{'s' if Q > 1 else ''} per video" if Q else "per question only (Q → ∞)",
                     fontsize=9, color=INK)
        ax.set_xlabel("list-price $ per question", fontsize=8, color=MUTED)
    axes[0].set_ylabel("accuracy (297 questions)", fontsize=8, color=MUTED)
    handles = [Line2D([], [], marker="o", ls="", color=READER_COLOR["haiku"], label="Haiku reads"),
               Line2D([], [], marker="o", ls="", color=READER_COLOR["sonnet"], label="Sonnet reads"),
               Line2D([], [], marker="o", ls="", color=MUTED, label="frames (no index)"),
               Line2D([], [], marker="s", ls="", color=MUTED, label="captions (index)")]
    fig.legend(handles=handles, loc="upper center", ncol=4, fontsize=8, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    FIG.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"caption_index_fig1_cost.{ext}", dpi=200)
    print(f"-> {FIG / 'caption_index_fig1_cost.png'}")


def cmd_types(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    wide, vtype, clus = R.confirm_frame()
    arms = [a for a in ["haiku:frames5", "haiku:selfcaptions", "haiku:sonnetcaptions", "haiku:captions",
                        "sonnet:frames5"] if a in wide]
    acc = wide[arms].groupby(vtype).mean().reindex(tc.VIDEO_TYPES)
    n = vtype.value_counts()
    print(acc.round(2).to_string())
    colors = [ARM_COLOR[a] for a in arms]
    fig, ax = plt.subplots(figsize=(7, 3.2))
    style(ax)
    w = 0.8 / len(arms)
    for i, a in enumerate(arms):
        xs = [k + (i - (len(arms) - 1) / 2) * w for k in range(len(tc.VIDEO_TYPES))]
        model, arm = a.split(":")
        ax.bar(xs, acc[a], width=w - 0.02, color=colors[i], label=f"{model.capitalize()}, {LABEL[arm]}", zorder=3)
    ax.set_xticks(range(len(tc.VIDEO_TYPES)))
    ax.set_xticklabels([f"{t.replace('_', '-')} ({n.get(t, 0)})" for t in tc.VIDEO_TYPES], fontsize=8)
    ax.set_ylabel("accuracy", fontsize=8, color=MUTED)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=7, frameon=False, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"caption_index_fig2_types.{ext}", dpi=200)
    print(f"-> {FIG / 'caption_index_fig2_types.png'}")


def cmd_readers(args):
    """Figure 3 (D3): accuracy by reader for question only, frames and the two indexes both readers read, with 95% CIs
    over clip clusters (claim 2a)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    wide, vtype, clus = R.confirm_frame()
    # (arm, colour, line style); frames and question only are references in ink and muted grey
    lines = [("frames5", INK, "-"), ("captions", ARM_COLOR["haiku:captions"], "-"),
             ("sonnetcaptions", ARM_COLOR["haiku:sonnetcaptions"], "-"), ("blind", MUTED, ":")]
    names = {"frames5": "5 frames", "captions": "Gemini video captions", "sonnetcaptions": "Sonnet captions",
             "blind": "question only"}
    # readers in order of their accuracy on frames; Qwen (C12) only once its rows exist
    readers = [m for m in ("haiku", "qwen", "sonnet") if f"{m}:frames5" in wide]
    fig, ax = plt.subplots(figsize=(2.1 + 1.05 * len(readers), 3.6))
    style(ax)
    ax.grid(False, axis="x")
    for i, (arm, color, ls) in enumerate(lines):
        xs = [x + (i - 1.5) * 0.04 for x in range(len(readers))]  # dodge, so the CIs at one reader do not overlap
        pts = []
        for x, model in zip(xs, readers):
            col = wide[f"{model}:{arm}"].dropna()
            est, lo, hi = R.cluster_boot(col.tolist(), clus[col.index].tolist())
            pts.append(est)
            print(f"{model}:{arm}: {est:.3f} [{lo:.3f}, {hi:.3f}] (n={len(col)})")
            ax.errorbar(x, est, yerr=[[est - lo], [hi - est]], color=color, linewidth=1, capsize=3, zorder=2)
        ax.plot(xs, pts, color=color, linestyle=ls, linewidth=2, marker="o", markersize=8,
                markeredgecolor="white", markeredgewidth=1.5, zorder=3)
        dy = {"frames5": 7, "captions": -5}.get(arm, 0)  # Sonnet on frames and on Gemini captions end 0.01 apart
        last = len(readers) - 1
        ax.annotate(f"{names[arm]} {pts[-1]:.2f}", (last + 0.1, pts[-1]), xytext=(4, dy), textcoords="offset points",
                    fontsize=7.5, color=INK, va="center")
        ax.annotate(f"{pts[0]:.2f}", (-0.1, pts[0]), xytext=(-4, 0), textcoords="offset points", fontsize=7.5,
                    color=INK, va="center", ha="right")
    ax.set_xticks(range(len(readers)))
    ax.set_xticklabels([f"{m.capitalize()} reads" for m in readers], fontsize=8, color=INK)
    ax.set_xlim(-0.35, len(readers) + 0.2)
    ax.set_ylim(0.3, 0.9)
    ax.set_ylabel("accuracy (297 questions)", fontsize=8, color=MUTED)
    ax.spines["bottom"].set_visible(False)
    ax.tick_params(axis="x", length=0)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"caption_index_fig3_readers.{ext}", dpi=200)
    print(f"-> {FIG / 'caption_index_fig3_readers.png'}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("cmd", choices=["cost", "types", "readers"])
    args = ap.parse_args()
    {"cost": cmd_cost, "types": cmd_types, "readers": cmd_readers}[args.cmd](args)


if __name__ == "__main__":
    main()
