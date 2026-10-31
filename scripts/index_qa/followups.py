#!/usr/bin/env python3
"""Follow-ups from the second paper review (research log C13, C14, A9, A10). Each command prints its entry's terms
exactly as listed in the log; run each once.

    c13  question-aware captions read without the frames (Haiku and Sonnet caption each window with the question)
    c14  Sonnet reads Haiku's self-captions and Gemini's frame and frame-clip captions; Haiku re-reads its
         self-captions (drift and reader noise between the October 20 runs and today)
    a9   D5's long-video cost projection against E1's measured costs at W = 36 (EgoSchema development set only)
    a10  A8's direction shift on Haiku's frames, adjusted for the number of answer options

    .venv/bin/python scripts/index_qa/followups.py c13 > results/index_qa/reports/c13.txt
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parents[2] / "src")]

from index_qa import reader as R  # noqa: E402
from index_qa import tempcompass as tc  # noqa: E402
from index_qa.caption import caption_windows  # noqa: E402

# C13: (name, arm a, arm b, prediction); a - b per question
C13_TERMS = [
    ("C13a Haiku: question-aware - blind self-captions", "haiku:selfqcaptions", "haiku:selfcaptions", +0.04),
    ("C13b Sonnet: question-aware - blind own captions", "sonnet:sonnetqcaptions", "sonnet:sonnetcaptions", +0.05),
    ("C13c Sonnet: question-aware own captions - frames", "sonnet:sonnetqcaptions", "sonnet:frames5", -0.04),
    ("C13d Haiku on Sonnet's question-aware captions - Sonnet on frames", "haiku:sonnetqcaptions", "sonnet:frames5",
     -0.05),
    ("C13e Haiku: question-aware self-captions - describe, then answer", "haiku:selfqcaptions", "haiku:describe5",
     0.00),
    ("C13f Haiku: Sonnet's question-aware - blind captions", "haiku:sonnetqcaptions", "haiku:sonnetcaptions", +0.04),
]
C13_BY_TYPE_PRED = {"C13c": {"stacked": -0.11, "reversed": -0.09, "plain": 0.00, "two_clip": 0.00}}
# C14
C14_TERMS = [
    ("C14a Sonnet - Haiku on Haiku's self-captions", "sonnet:selfcaptions", "haiku:selfcaptions", +0.03),
    ("C14b Sonnet - Haiku on Gemini frame captions", "sonnet:framecaptions", "haiku:framecaptions", +0.05),
    ("C14c Sonnet - Haiku on Gemini frame-clip captions", "sonnet:frameclipcaptions", "haiku:frameclipcaptions", +0.05),
    ("C14e Sonnet: Sonnet's - Haiku's captions (captioner effect for Sonnet)", "sonnet:sonnetcaptions",
     "sonnet:selfcaptions", +0.04),
    ("C14g Sonnet: Gemini frame captions - frames", "sonnet:framecaptions", "sonnet:frames5", +0.04),
    ("C14f Haiku: re-read - first read of its self-captions", "haiku:selfcaptions_rerun", "haiku:selfcaptions", 0.00),
]


def wide_frame():
    wide, vtype, clus = R.confirm_frame()
    return wide, vtype, clus


def term(wide, clus, name, a, b, pred):
    d = (wide[a] - wide[b]).dropna()
    e, lo, hi = R.paired_boot(d.tolist())
    _, clo, chi = R.cluster_boot(d.tolist(), clus[d.index].tolist())
    print(f"  {name}: {a} - {b} = {e:+.3f} [{lo:+.3f}, {hi:+.3f}], clustered [{clo:+.3f}, {chi:+.3f}] "
          f"(n={len(d)}; prediction {pred:+.2f}; {'inside' if lo <= pred <= hi else 'OUTSIDE'})")


def accuracies(wide, vtype, arms):
    t = wide[arms].groupby(vtype).mean().reindex(tc.VIDEO_TYPES).T
    t["all"] = wide[arms].mean()
    print(t.round(3).to_string())


def retention(wide, clus, model, arms):
    for arm in arms:
        a = f"{model}:{arm}"
        x = wide[[a, f"{model}:frames5", f"{model}:blind"]].dropna()
        e, lo, hi = R.retention_boot(x[a], x[f"{model}:frames5"], x[f"{model}:blind"], clusters=clus[x.index].tolist())
        print(f"  {a}: retention {e:.2f}, clustered [{lo:.2f}, {hi:.2f}]")


def need(wide, arms):
    missing = [a for a in arms if a not in wide or wide[a].isna().any()]
    if missing:
        sys.exit(f"not every question answered yet in: {missing}")


def caption_stats(captioner: str):
    """Median words per window, and the share of windows whose caption contains an option letter as a token
    ("A." or "(A)" or "option A"), for a captioner over the confirmatory videos."""
    import re
    words, letter = [], []
    for q in tc.items("confirm"):
        caps = R.captions(q["video_id"], captioner) or []
        for c in caps:
            words.append(len(c.split()))
            letter.append(bool(re.search(r"\b(option|answer)\s+[A-E]\b|\([A-E]\)|\b[A-E]\.\s", c)))
    return float(np.median(words)), float(np.mean(letter)), len(words)


def cmd_c13(args):
    wide, vtype, clus = wide_frame()
    arms = ["haiku:selfcaptions", "haiku:selfqcaptions", "haiku:describe5", "haiku:sonnetcaptions",
            "haiku:sonnetqcaptions", "sonnet:frames5", "sonnet:sonnetcaptions", "sonnet:sonnetqcaptions"]
    need(wide, arms)
    print(f"C13. {len(wide)} confirmatory questions. Accuracy by video type:")
    accuracies(wide, vtype, arms)
    for cap in ("claude-haiku-5-5-self", "claude-haiku-5-5-self-question", "claude-sonnet-5-5-frames",
                "claude-sonnet-5-5-frames-question"):
        m, share, n = caption_stats(cap)
        print(f"  captions {cap}: median {m:.0f} words over {n} windows; share with an option letter {share:.3f}")
    print("\nTerms (95% CIs over questions; clustered over the 156 clip clusters):")
    for t in C13_TERMS:
        term(wide, clus, *t)
    print("\nBy video type (95% CIs over clip clusters; descriptive):")
    R.type_effects(wide, C13_TERMS, vtype, clus)
    print(f"  predicted by type for C13c: {C13_BY_TYPE_PRED['C13c']}")
    print("\nRetention, Sonnet (clustered):")
    retention(wide, clus, "sonnet", ["sonnetcaptions", "sonnetqcaptions"])


def cmd_c14(args):
    wide, vtype, clus = wide_frame()
    arms = ["sonnet:selfcaptions", "sonnet:framecaptions", "sonnet:frameclipcaptions", "haiku:selfcaptions_rerun",
            "haiku:selfcaptions", "haiku:framecaptions", "haiku:frameclipcaptions", "haiku:sonnetcaptions",
            "sonnet:sonnetcaptions", "sonnet:frames5"]
    need(wide, arms)
    print(f"C14. {len(wide)} confirmatory questions. Accuracy by video type:")
    accuracies(wide, vtype, arms)
    print("\nTerms (95% CIs over questions; clustered over the 156 clip clusters):")
    for t in C14_TERMS:
        term(wide, clus, *t)
    # C14d: own-words interaction, (S - H on Sonnet's captions) - (S - H on Haiku's captions), per question
    d = ((wide["sonnet:sonnetcaptions"] - wide["haiku:sonnetcaptions"])
         - (wide["sonnet:selfcaptions"] - wide["haiku:selfcaptions"])).dropna()
    e, lo, hi = R.paired_boot(d.tolist())
    _, clo, chi = R.cluster_boot(d.tolist(), clus[d.index].tolist())
    print(f"  C14d own-words interaction, (S - H on Sonnet's captions) - (S - H on Haiku's captions) = {e:+.3f} "
          f"[{lo:+.3f}, {hi:+.3f}], clustered [{clo:+.3f}, {chi:+.3f}] (n={len(d)}; prediction -0.03; "
          f"{'inside' if lo <= -0.03 <= hi else 'OUTSIDE'})")
    rows = R.latest("haiku")
    keep = {q["item"] for q in tc.items("confirm")}
    a = {r["question_item"]: r["letter"] for r in rows.values() if r["arm"] == "selfcaptions" and r["question_item"] in keep}
    b = {r["question_item"]: r["letter"] for r in rows.values()
         if r["arm"] == "selfcaptions_rerun" and r["question_item"] in keep}
    common = sorted(set(a) & set(b))
    print(f"  C14f answer-change rate, Haiku re-read vs. first read of the same captions: "
          f"{np.mean([a[i] != b[i] for i in common]):.3f} (n={len(common)}; prediction 0.08)")
    print("\nBy video type (95% CIs over clip clusters; descriptive):")
    R.type_effects(wide, C14_TERMS, vtype, clus)
    print("\nRetention, Sonnet (clustered):")
    retention(wide, clus, "sonnet", ["selfcaptions", "sonnetcaptions", "framecaptions", "frameclipcaptions",
                                     "captions"])


# A9: E1's pairs with a per-window caption index (the whole-video index has no D5 projection)
A9_PAIRS = [("haiku:sonnetcaptions", "sonnet:frames5"), ("haiku:selfcaptions", "sonnet:frames5"),
            ("sonnet:sonnetcaptions", "sonnet:frames5"), ("haiku:sonnetcaptions", "haiku:frames5"),
            ("haiku:selfcaptions", "haiku:frames5")]
EGO_W = 36
TC_FRAME_TOK, EGO_FRAME_TOK = 40, 48  # Claude image tokens per frame: 224x126 (TempCompass), 224x168 (EgoSchema)


def cmd_a9(args):
    """D5's projection (reader.qstar_by_length) evaluated at W = 36 from TempCompass's confirmatory measurements,
    against E1's measured costs. Variant 'as published' uses TempCompass's tokens per window; variant 'frame size'
    adds the known extra image tokens of EgoSchema's frames (8 per frame) to frame arms and Claude caption calls.
    E1 (ego_dev) only: E2 was stopped and its rows are not read."""
    import pandas as pd
    qs = tc.items("confirm")
    vid = {q["item"]: q["video_id"] for q in qs}
    tcf = R.call_frame("confirm")
    wins = {v: len(caption_windows(v, R.CAPTIONERS["captions"])) for v in set(vid.values())}
    tcf = tcf.assign(W=[wins[vid[i]] for i in tcf.question_item])
    w_bar = tcf.W.mean()
    ego = R.call_frame("ego_dev")
    ego_vid = {q["item"]: q["video_id"] for q in tc.items("ego_dev")}
    tc_videos, ego_videos = set(vid.values()), set(ego_vid.values())
    extra_img = EGO_FRAME_TOK - TC_FRAME_TOK
    frames_per_window = {"frames5": 5, "frame1": 1}
    print(f"A9. Projection from TempCompass (mean W = {w_bar:.2f}) to W = {EGO_W}; measured on E1 "
          f"({len(ego_videos)} EgoSchema videos, {EGO_W} windows each).")

    def projected(arm, adjust):
        """The arm's TempCompass mean cost, with its input moved from w_bar windows to EGO_W windows: input =
        mean input - slope * w_bar + (slope + extra) * EGO_W, where extra is the frame-size adjustment (every window
        of a frame arm, including the first w_bar, carries the larger frames). Output and thinking stay at their
        TempCompass mean, as in D5."""
        d = tcf[tcf.arm == arm]
        model, kind = arm.split(":")
        slope = np.polyfit(d.W, d.in_tok, 1)[0]
        extra = extra_img * frames_per_window.get(kind, 0) if adjust else 0
        return d.usd.mean() + R.PRICES[model][0] / 1e6 * ((slope + extra) * EGO_W - slope * w_bar)

    def index_projected(cap, adjust):
        per = [c for (v, _), c in R.index_usd(cap).items() if v in tc_videos]
        k = float(np.mean(per))
        if adjust and R.INDEX_SOURCE[cap][0] == "claude":
            k += R.PRICES[R.INDEX_SOURCE[cap][1]][0] / 1e6 * extra_img * 5
        return k * EGO_W

    print("\nPer-question cost ($), projected (as published / frame size) and measured; ratio measured / projected:")
    arms = sorted({x for p in A9_PAIRS for x in p})
    meas = {a: ego[ego.arm == a].usd.mean() for a in arms}
    for a in arms:
        p0, p1 = projected(a, False), projected(a, True)
        print(f"  {a:24s} {p0:.5f} / {p1:.5f}; measured {meas[a]:.5f}; ratio {meas[a] / p0:.2f} / {meas[a] / p1:.2f}")
    print("\nIndex cost per video ($):")
    caps = sorted({a.split(":")[1] for a, _ in A9_PAIRS})
    imeas = {}
    for cap in caps:
        per_video = R.index_per_video(cap, ego_videos)
        imeas[cap] = float(np.mean([x for x in per_video.values() if x is not None]))
        q0, q1 = index_projected(cap, False), index_projected(cap, True)
        print(f"  {cap:16s} {q0:.4f} / {q1:.4f}; measured {imeas[cap]:.4f}; ratio {imeas[cap] / q0:.2f} / "
              f"{imeas[cap] / q1:.2f}")
    print("\nQ_BE at W = 36: projected (as published / frame size), measured (point estimate from the means); "
          "ratio measured / projected:")
    for a, b in A9_PAIRS:
        cap = a.split(":")[1]
        row = []
        for adj in (False, True):
            save = projected(b, adj) - projected(a, adj)
            row.append(index_projected(cap, adj) / save if save > 0 else float("inf"))
        m = imeas[cap] / (meas[b] - meas[a]) if meas[b] > meas[a] else float("inf")
        print(f"  {a} vs. {b}: {row[0]:.1f} / {row[1]:.1f}; measured {m:.1f}; ratio {m / row[0]:.2f} / "
              f"{m / row[1]:.2f}")
    print("\nWhere the error comes from, per arm: input tokens and output tokens (incl. thinking) per read, "
          "projected (frame size) vs. measured:")
    for a in arms:
        d = tcf[tcf.arm == a]
        kind = a.split(":")[1]
        slope = np.polyfit(d.W, d.in_tok, 1)[0]
        pin = d.in_tok.mean() + (slope + extra_img * frames_per_window.get(kind, 0)) * EGO_W - slope * w_bar
        e = ego[ego.arm == a]
        print(f"  {a:24s} input {pin:.0f} vs. {e.in_tok.mean():.0f}; output {d.out_tok.mean():.0f} vs. "
              f"{e.out_tok.mean():.0f}")


def cmd_a10(args):
    """Haiku on 5 frames, A8's questions (set rest) against C1's (confirm), with the number of options."""
    import pandas as pd
    rows = []
    for set_ in ("confirm", "rest"):
        keep = {q["item"]: q for q in tc.items(set_)}
        for r in R.latest("haiku").values():
            if r["arm"] == "frames5" and r["question_item"] in keep:
                q = keep[r["question_item"]]
                rows.append(dict(set=set_, item=r["question_item"], video_id=q["video_id"], dim=q["dim"],
                                 k=len(q["options"]), correct=float(r["correct"])))
    df = pd.DataFrame(rows)
    vids = sorted(set(df.video_id))
    cl = dict(zip(vids, R.clip_clusters(vids)))
    df["cluster"] = df.video_id.map(cl)
    df["cc"] = (df.correct - 1 / df.k) / (1 - 1 / df.k)  # chance-corrected
    print(f"A10. Haiku on 5 frames: {len(df[df.set == 'confirm'])} C1 and {len(df[df.set == 'rest'])} A8 questions, "
          f"{df.cluster.nunique()} clip clusters (resampled jointly for both sets).")
    print("\nOptions per question, by aspect and set (count of questions with k options):")
    print(df.groupby(["dim", "set", "k"]).size().unstack("k", fill_value=0).to_string())
    print("\nAccuracy by aspect, set and k:")
    print(df.groupby(["dim", "k", "set"]).correct.agg(["mean", "size"]).unstack("set").round(3).to_string())

    keys = np.array(sorted(df.cluster.unique()))
    at = {k: np.flatnonzero(df.cluster.to_numpy() == k) for k in keys}
    rng = np.random.default_rng(0)
    boots = [np.concatenate([at[k] for k in rng.choice(keys, len(keys))]) for _ in range(5000)]

    def ci(stat):
        e = stat(df)
        bs = np.array([stat(df.iloc[i]) for i in boots])
        lo, hi = np.nanpercentile(bs, [2.5, 97.5])
        return f"{e:+.3f} [{lo:+.3f}, {hi:+.3f}]"

    def raw(d, dims):
        d = d[d.dim.isin(dims)]
        return d[d.set == "rest"].correct.mean() - d[d.set == "confirm"].correct.mean()

    def standardized(d, dims, col="correct"):
        """Direct standardization: the set difference within each (aspect, k) stratum, weighted by the stratum's
        share of both sets' questions (fixed from the full sample)."""
        full = df[df.dim.isin(dims)]
        w = full.groupby(["dim", "k"]).size() / len(full)
        d = d[d.dim.isin(dims)]
        m = d.groupby(["dim", "k", "set"])[col].mean().unstack("set")
        m = m.reindex(w.index)
        diff = (m.get("rest") - m.get("confirm"))
        ok = diff.notna()
        return float((diff[ok] * w[ok]).sum() / w[ok].sum())

    def chance_corrected(d, dims):
        d = d[d.dim.isin(dims)]
        return d[d.set == "rest"].cc.mean() - d[d.set == "confirm"].cc.mean()

    print("\nTerms (A8 - C1, Haiku on 5 frames; 95% CIs over clip clusters):")
    print(f"  A10a direction, raw: {ci(lambda d: raw(d, ['direction']))} (known: 0.552 - 0.283)")
    print(f"  A10b direction, standardized by k: {ci(lambda d: standardized(d, ['direction']))} (prediction +0.18)")
    print(f"  A10c direction, chance-corrected: {ci(lambda d: chance_corrected(d, ['direction']))} "
          "(prediction +0.25)")
    # A10d: same video, direction asked in both sets; pairs with equal k
    conf = df[(df.set == "confirm") & (df.dim == "direction")].set_index("video_id")
    rest = df[(df.set == "rest") & (df.dim == "direction") & df.video_id.isin(conf.index)].copy()
    rest["conf_correct"] = rest.video_id.map(conf.correct)
    rest["conf_k"] = rest.video_id.map(conf.k)
    for label, sub, pred in (("all same-video pairs", rest, None), ("pairs with equal k", rest[rest.k == rest.conf_k],
                                                                     0.15)):
        d = (sub.correct - sub.conf_correct).to_numpy()
        e, lo, hi = R.cluster_boot(list(d), list(sub.cluster))
        print(f"  A10d direction, {label}: A8 {sub.correct.mean():.3f} vs. C1 {sub.conf_correct.mean():.3f}, "
              f"difference {e:+.3f} [{lo:+.3f}, {hi:+.3f}] (n={len(sub)} A8 questions on "
              f"{sub.video_id.nunique()} videos" + (f"; prediction +{pred:.2f})" if pred else ")"))
    alld = list(tc.DIMS)
    print(f"  A10e all aspects, raw: {ci(lambda d: raw(d, alld))}; standardized by (aspect, k): "
          f"{ci(lambda d: standardized(d, alld))} (prediction +0.06)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, fn in (("c13", cmd_c13), ("c14", cmd_c14), ("a9", cmd_a9), ("a10", cmd_a10)):
        sub.add_parser(name).set_defaults(fn=fn)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
