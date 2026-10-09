#!/usr/bin/env python3
"""
Reconstruction (recon task) across Claude models on the same gaps, with channel-bootstrap 95% CIs.

Reads results/blind_llm/recon__<model>__scores.csv (written by `blind_llm_runner.py score --model <m>`), keeps the
(video, W) gaps that every listed model, caption copy and Llama filled, averages each paired difference per video,
then resamples channels. Ranks are of the true second among the video's 60 (1 = best, chance 30.5).

Usage (from repo root):
    .venv/bin/python scripts/blind_recon_compare.py sonnet [opus]
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

RES = Path("results/blind_llm")
KEY = ["vid", "chan", "w"]
METRICS = ["text_rank", "frame_rank"]


def main():
    models = sys.argv[1:] or ["sonnet"]
    parts = []
    for i, m in enumerate(models):
        d = pd.read_csv(RES / f"recon__{m}__scores.csv")
        d.loc[d.method == "claude", "method"] = m
        parts.append(d if i == 0 else d[d.method == m])  # baselines are the same in every file; keep one copy
    df = pd.concat(parts)
    arms = models + ["caption_copy", "llama"]
    wide = df.pivot_table(index=KEY, columns="method", values=METRICS).dropna(subset=[(x, a) for x in METRICS for a in arms])
    idx = wide.index
    print(f"{len(idx)} gaps, {idx.get_level_values('vid').nunique()} videos, "
          f"{idx.get_level_values('chan').nunique()} channels (W = {sorted(idx.get_level_values('w').unique())})")
    print("mean rank (1 = best, chance 30.5):")
    print(wide[[(x, a) for x in METRICS for a in arms]].mean().unstack(0).round(1).to_string())

    rng = np.random.default_rng(0)

    def boot(diff: pd.Series) -> str:  # per-video mean, then resample channels
        v = diff.groupby(level=["vid", "chan"]).mean()
        chans = v.index.get_level_values("chan")
        by = [v.to_numpy()[chans == c] for c in chans.unique()]
        bs = [np.concatenate([by[j] for j in rng.integers(0, len(by), len(by))]).mean() for _ in range(5000)]
        lo, hi = np.percentile(bs, [2.5, 97.5])
        return f"{v.mean():+.1f} [{lo:+.1f}, {hi:+.1f}]"

    pairs = [(m, b) for m in models for b in ["caption_copy", "llama"]]
    pairs += [(models[i], models[j]) for i in range(len(models)) for j in range(i + 1, len(models))]
    for a, b in pairs:
        print(f"{a} - {b}: " + "   ".join(f"{x} {boot(wide[(x, a)] - wide[(x, b)])}" for x in METRICS))
    print("by width, model - caption copy (mean over gaps):")
    w = idx.get_level_values("w")
    print(pd.DataFrame({f"{m} {x}": (wide[(x, m)] - wide[(x, "caption_copy")]).groupby(w).mean()
                        for m in models for x in METRICS}).round(1).assign(n=pd.Series(w).value_counts().sort_index().values).to_string())


if __name__ == "__main__":
    main()
