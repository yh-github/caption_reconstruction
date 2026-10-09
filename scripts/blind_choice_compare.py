#!/usr/bin/env python3
"""
Forced choice (no hint) across Claude models on the same items, with channel-bootstrap 95% CIs.

Items: Sonnet's main estimate (2026-10-08), i.e. its choice_nohint rows after the first 40 distinct items (those 40
were the earlier no-hint check on the Llama pool). Other models are restricted to those items; a model that lacks
some of them is reported on the overlap, and the count is printed. Fill gaps with
    blind_llm_runner.py run --task choice_nohint --model <m> --pool all --shuffle --match sonnet

Usage (from repo root):
    .venv/bin/python scripts/blind_choice_compare.py haiku sonnet [opus]
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

RES = Path("results/blind_llm")
N_CHECK = 40  # leading Sonnet items from the no-hint check, not part of the main estimate


def rows(model: str) -> dict[str, dict]:
    out = {}
    for line in open(RES / f"choice_nohint__{model}.jsonl"):
        r = json.loads(line)
        out[r["item"]] = r  # latest attempt wins
    return out


def sonnet_main_items() -> list[str]:
    seen = []
    for line in open(RES / "choice_nohint__sonnet.jsonl"):
        item = json.loads(line)["item"]
        if item not in seen:
            seen.append(item)
    return seen[N_CHECK:]


def main():
    models = sys.argv[1:] or ["haiku", "sonnet"]
    items = set(sonnet_main_items())
    acc = {m: {k: float(r["parsed"] == r["answer_key"]) for k, r in rows(m).items()
               if r.get("status", "ok") == "ok" and k in items} for m in models}
    common = sorted(set.intersection(items, *[set(a) for a in acc.values()]))
    s = pd.read_csv("results/forced_choice/forced_choice_merged.csv").set_index("item").loc[common]
    for m in models:
        s[m] = [acc[m][k] for k in common]
    print(f"{len(s)} of {len(items)} items ({s.vid.nunique()} videos, {s.chan.nunique()} channels; chance 25%)")

    # Channel bootstrap on per-channel sums: resample channels, ratio of summed scores to summed item counts.
    rng = np.random.default_rng(0)
    by_chan = s.groupby("chan")
    n_items = by_chan.size().to_numpy()
    draws = rng.integers(0, len(n_items), (5000, len(n_items)))

    def ci(x: pd.Series):
        sums = x.groupby(s.chan).sum().to_numpy()
        return np.percentile(sums[draws].sum(1) / n_items[draws].sum(1), [2.5, 97.5])

    cols = models + ["text_copy_ok", "text_copy_assign_ok", "vis_copy_ok"]
    for c in cols:
        lo, hi = ci(s[c])
        print(f"{c:22s} {100 * s[c].mean():5.1f} [{100 * lo:.1f}, {100 * hi:.1f}]")
    pairs = [(m, b) for m in models for b in ["text_copy_ok", "vis_copy_ok"]]
    pairs += [(models[i], models[j]) for i in range(len(models)) for j in range(i + 1, len(models))]
    for a, b in pairs:
        lo, hi = ci(s[a] - s[b])
        print(f"{a} - {b}: {100 * (s[a] - s[b]).mean():+.1f} [{100 * lo:+.1f}, {100 * hi:+.1f}]")
    print("by width:")
    print((100 * s.groupby("w")[cols].mean()).round(1).assign(n=s.groupby("w").size()).to_string())


if __name__ == "__main__":
    main()
