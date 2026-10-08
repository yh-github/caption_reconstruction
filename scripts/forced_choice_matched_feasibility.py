#!/usr/bin/env python3
"""
Feasibility of continuity-matched distractors for the forced-choice gap test (CPU, no LLM).

Same gaps and span constraints as scripts/forced_choice_gap.py, but distractors are chosen among spans
whose continuity with the gap's boundaries (cosine of the span mean to the gap's boundary mean) is
within k * sd of the true gap's, where sd is the spread of that score over the video's candidate spans.
Matching is done on captions (MPNet), on frames (SigLIP 2), or on both.

Pass criteria (decision section of the assessment doc, 2026-10-07): >= ~300 of 1,325 items get K-1
matched distractors, and both copy baselines fall to <= 35% on them.

Usage (from repo root):
    .venv/bin/python scripts/forced_choice_matched_feasibility.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from forced_choice_gap import K, MARGIN, SEED, WIDTHS, load_videos, unit  # noqa: E402
from llm.local_embedder import LocalEmbedder  # noqa: E402


def candidate_spans(w):
    s0 = 30 - w // 2
    L, R = s0 - 1, s0 + w
    gap = list(range(s0, s0 + w))
    starts = [s for s in range(1, 60 - w) if s + w - 1 < L - MARGIN + 1 or s > R + MARGIN - 1]
    return gap, [list(range(s, s + w)) for s in starts]


def continuity(X, gap, spans):
    gb = unit(X[[gap[0] - 1, gap[-1] + 1]].mean(axis=0))
    return np.array([float(unit(X[sp].mean(axis=0)) @ gb) for sp in [gap] + spans])


def pick(spans, ok, gap, rng):
    chosen = [gap]
    for i in rng.permutation(np.where(ok)[0]):
        if all(min(abs(a - b) for a in spans[i] for b in sp) >= 2 for sp in chosen):
            chosen.append(spans[i])
        if len(chosen) == K:
            return chosen
    return None


def main():
    emb = LocalEmbedder("all-mpnet-base-v2")
    rng = np.random.default_rng(SEED)
    rows = []
    for _, vid, caps in load_videos():
        vf = Path(f"local/wild_videos_embs_siglip/{vid.replace(chr(39), '_')}.npy")
        if not vf.exists():
            continue
        V = unit(np.load(vf)[:60])
        if len(V) < 60:
            continue
        X = {"text": unit(emb.get_embeddings(f"{vid}_dense_caps", caps)), "vis": V}
        for w in WIDTHS:
            gap, spans = candidate_spans(w)
            sc = {m: continuity(X[m], gap, spans) for m in X}
            for k in [0.1, 0.25, 0.5]:
                close = {m: np.abs(sc[m][1:] - sc[m][0]) <= k * sc[m][1:].std() for m in sc}
                for match, ok in [("text", close["text"]), ("vis", close["vis"]), ("both", close["text"] & close["vis"])]:
                    ch = pick(spans, ok, gap, rng)
                    row = dict(vid=vid, w=w, k=k, match=match, feasible=ch is not None)
                    if ch is not None:
                        idx = [0] + [spans.index(sp) + 1 for sp in ch[1:]]
                        for m in sc:
                            row[f"{m}_copy_ok"] = float(np.argmax(sc[m][idx]) == 0)
                    rows.append(row)
    d = pd.DataFrame(rows)
    out = Path("results/forced_choice")
    out.mkdir(parents=True, exist_ok=True)
    d.to_csv(out / "matched_feasibility.csv", index=False)
    s = d.groupby(["match", "k"]).agg(items=("feasible", "sum"), text_copy=("text_copy_ok", "mean"),
                                      vis_copy=("vis_copy_ok", "mean"))
    print(f"{d.vid.nunique()} videos x {len(WIDTHS)} widths = {d.groupby(['match', 'k']).size().iloc[0]} items "
          f"per setting; chance 0.25")
    print(s.round(3).to_string())
    print("\nfeasible items by width (k=0.25):")
    print(d[d.k == 0.25].pivot_table(index="match", columns="w", values="feasible", aggfunc="sum").to_string())


if __name__ == "__main__":
    main()
