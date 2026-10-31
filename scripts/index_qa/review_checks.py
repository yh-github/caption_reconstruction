#!/usr/bin/env python3
"""D7 (research log): checks for claims the paper made without a computed basis. No model calls.

    1. Do question-level and clip-cluster CIs agree on every arm pair (confirmatory accuracy, minimal pairs)?
    2. Readers' accuracy on 5 frames: Qwen - Haiku, Sonnet - Qwen, Sonnet - Haiku.
    3. How much of A7's out-of-fold AUC is the (reader, captioner family) pair?
    4. Share of each audited reader arm's wrong answers (and omissions) that fall on built videos.

    .venv/bin/python scripts/index_qa/review_checks.py d7   # output to results/index_qa/reports/d7.txt
"""
from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

import numpy as np

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parents[2] / "src")]

from index_qa import audit as AU  # noqa: E402
from index_qa import decide as DE  # noqa: E402
from index_qa import reader as R  # noqa: E402
from index_qa import tempcompass as tc  # noqa: E402

BUILT = {"reversed", "stacked", "two_clip"}


def excl0(lo: float, hi: float) -> bool:
    return lo > 0 or hi < 0


def agreement(mat, clus, label: str) -> list[tuple]:
    """Every pair of columns of mat (rows = units, values = 0/1): both 95% CIs, and whether they agree on 0."""
    out = []
    cols = [c for c in mat.columns if mat[c].notna().all()]
    for a, b in itertools.combinations(sorted(cols), 2):
        d = (mat[a] - mat[b]).to_numpy()
        if not d.any():
            continue
        m, qlo, qhi = R.paired_boot(list(d))
        _, clo, chi = R.cluster_boot(list(d), list(clus.reindex(mat.index)))
        out.append((a, b, m, qlo, qhi, clo, chi, excl0(qlo, qhi) == excl0(clo, chi), (chi - clo) / (qhi - qlo)))
    n_dis = sum(not r[7] for r in out)
    ratio = np.median([r[8] for r in out])
    print(f"\n{label}: {len(out)} arm pairs with any difference; CIs disagree on 0 in {n_dis}; "
          f"median width ratio (cluster / question) {ratio:.3f} [IQR {np.percentile([r[8] for r in out], 25):.3f}, "
          f"{np.percentile([r[8] for r in out], 75):.3f}]")
    for a, b, m, qlo, qhi, clo, chi, ok, _ in out:
        if not ok:
            near = min(abs(x) for x in (qlo, qhi, clo, chi))
            print(f"  DISAGREE {a} - {b}: {m:+.3f} question [{qlo:+.3f}, {qhi:+.3f}], cluster [{clo:+.3f}, {chi:+.3f}]"
                  f"; nearest endpoint to 0: {near:.3f}")
    return out


def cmd_d7(args):
    import pandas as pd
    wide, vtype, clus = R.confirm_frame()
    print(f"D7. {len(wide)} confirmatory questions, {len(set(clus))} clip clusters, {wide.shape[1]} arms.")

    print("\n1. Question-level vs. clip-cluster 95% CIs (agree = same decision on whether 0 is excluded)")
    agreement(wide, clus, "Confirmatory accuracy, every pair of arms")
    both, _, _, _, _, pclus, _ = R.pair_frame()
    agreement(both, pclus, "Minimal-pair accuracy (193 pairs), every pair of Haiku and Sonnet arms")

    print("\n2. Readers on 5 frames (95% CIs; question-level, then clip clusters)")
    for a, b in (("qwen", "haiku"), ("sonnet", "qwen"), ("sonnet", "haiku")):
        d = list(wide[f"{a}:frames5"] - wide[f"{b}:frames5"])
        m, lo, hi = R.paired_boot(d)
        _, clo, chi = R.cluster_boot(d, list(clus))
        print(f"  {a} - {b}: {m:+.3f} [{lo:+.3f}, {hi:+.3f}], clustered [{clo:+.3f}, {chi:+.3f}]")

    print("\n3. A7: how much of the out-of-fold AUC is the pair (898 disagreeing rows, A7's folds; 95% CIs over clip"
          " clusters)")
    from sklearn.metrics import roc_auc_score
    df, _ = DE.rows()
    dis = df[df.index_ok != df.frames_ok].reset_index(drop=True)
    y, cl, pair = dis.won.to_numpy(), dis.cluster.to_numpy(), dis.pair.to_numpy()

    def auc(p):
        return lambda i: roc_auc_score(y[i], p[i]) if len(set(y[i])) > 1 else np.nan

    def within_pair_auc(p):
        """Stratified AUC: concordance over (won, lost) row pairs from the same (reader, index) pair."""
        def stat(i):
            num = den = 0.0
            for g in np.unique(pair[i]):
                j = i[pair[i] == g]
                pos, neg = p[j][y[j] == 1], p[j][y[j] == 0]
                if len(pos) and len(neg):
                    c = (pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()
                    num, den = num + c, den + len(pos) * len(neg)
            return num / den if den else np.nan
        return stat

    m1 = DE.oof(dis, DE.MODELS[DE.A8_M1], False)
    for name, feats in (("pair only (reader + family)", ["reader", "family"]), ("aspect only", ["dim"]),
                        ("M1 aspect + pair (A7, restated)", DE.MODELS[DE.A8_M1])):
        p = DE.oof(dis, feats, False)
        e, lo, hi = DE.cluster_ci(auc(p), cl)
        print(f"  {name:34s} AUC {e:.3f} [{lo:.3f}, {hi:.3f}]")
    e, lo, hi = DE.cluster_ci(within_pair_auc(m1), cl)
    print(f"  {'M1, within pair (stratified)':34s} AUC {e:.3f} [{lo:.3f}, {hi:.3f}]")

    print("\n4. Audited reader arms: share of wrong answers, and of omission-labelled ones, on built videos"
          " (reversed, stacked, two-clip); built videos are "
          f"{np.mean([v in BUILT for v in vtype]):.2f} of the questions")
    lab = AU.labels()
    qs = {q["item"]: q for q in tc.items("confirm")}
    for ma in AU.READER_ARMS:
        model, arm = ma.split(":")
        wrong = [r["question_item"] for r in R.latest(model).values()
                 if r["arm"] == arm and r["question_item"] in qs and not r["correct"]]
        built = [tc.video_type(qs[i]["video_id"]) in BUILT for i in wrong]
        om = [b for i, b in zip(wrong, built) if lab.get(AU.key(arm, i), {}).get("label") == "silent"]
        print(f"  {ma:26s} wrong {len(wrong):3d}, on built {np.mean(built):.2f}; omissions {len(om):3d}, "
              f"on built {np.mean(om):.2f}")


LIST_USD = {"haiku": (0.10, 0.50), "sonnet": (2.0, 10.0)}  # $ per 1M input / output (thinking billed as output)
UUID = __import__("re").compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-")


def cmd_compute(args):
    """Claude calls and list-price dollars per study, from every logged call (one row per call, retries included):
    reads (claude/*.jsonl), captions (self_captions.jsonl) and the audit judge (audit/sonnet.jsonl). EgoSchema rows
    have items starting "eg" (reads) or UUID video IDs (captions)."""
    import collections
    import json
    n, usd = collections.Counter(), collections.Counter()
    files = [(tc.OUT / "claude" / "haiku.jsonl", "read"), (tc.OUT / "claude" / "sonnet.jsonl", "read"),
             (tc.OUT / "self_captions.jsonl", "caption"), (tc.OUT / "audit" / "sonnet.jsonl", "audit")]
    for path, kind in files:
        for line in path.read_text().splitlines():
            r = json.loads(line)
            fam = "haiku" if "haiku" in str(r.get("model")) else "sonnet"
            item = str(r.get("question_item") or r.get("video_id") or "")
            study = "egoschema" if item.startswith("eg") or UUID.match(item) else "tempcompass"
            u = r.get("usage") or {}
            tin = sum(u.get(k) or 0 for k in ("input_tokens", "cache_creation_input_tokens", "cache_read_input_tokens"))
            pin, pout = LIST_USD[fam]
            n[study, kind] += 1
            usd[study, kind] += (tin * pin + (u.get("output_tokens") or 0) * pout) / 1e6
    for study in ("tempcompass", "egoschema"):
        ks = sorted(k for k in n if k[0] == study)
        print(f"{study}: {sum(n[k] for k in ks)} Claude calls, ${sum(usd[k] for k in ks):.2f} at list price; "
              + ", ".join(f"{k[1]} {n[k]} (${usd[k]:.2f})" for k in ks))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("d7").set_defaults(fn=cmd_d7)
    sub.add_parser("compute").set_defaults(fn=cmd_compute)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
