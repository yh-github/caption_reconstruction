#!/usr/bin/env python3
"""A7 (research log): a decision formula for when a reader should read the caption index instead of its frames.

One row per (reader, index, question) where the reader's index answer and frames answer disagree; the target is
"the index won". Features are known before the question is read (no model calls):
    video     motion (priors.py), cuts (log1p of consecutive-frame SigLIP 2 distances above CUT), log windows
    captions  mention: share of the index's windows whose caption uses the wording of the question's aspect
              (MENTION; 0 for action); coverage (priors.py, for this index's captions)
    task      the question's aspect
    pair      the reader and the captioner family
Models are L2 logistic regressions (C = 1, standardized numbers, one-hot categories), scored out of fold: 10 folds
grouped by clip cluster, and the same crossed with leaving one (reader, index) pair out.

    .venv/bin/python scripts/index_qa/decide.py a7   # output to results/index_qa/reports/a7.txt
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parents[2] / "src")]

from index_qa import priors as PR  # noqa: E402
from index_qa import reader as R  # noqa: E402
from index_qa import tempcompass as tc  # noqa: E402
from index_qa.caption import captions  # noqa: E402

# (reader, index arm): every per-window question-blind index a reader read on the confirmatory set
PAIRS = [("haiku", a) for a in ("selfcaptions", "sonnetcaptions", "sonnetcaptions448", "captions", "framecaptions",
                                "frameclipcaptions")] \
    + [("sonnet", a) for a in ("sonnetcaptions", "sonnetcaptions448", "captions")] \
    + [("qwen", a) for a in ("qwencaptions", "selfcaptions", "sonnetcaptions", "captions", "framecaptions",
                             "frameclipcaptions")]
FAMILY = {"selfcaptions": "haiku", "sonnetcaptions": "sonnet", "sonnetcaptions448": "sonnet", "captions": "gemini",
          "framecaptions": "gemini", "frameclipcaptions": "gemini", "qwencaptions": "qwen"}
CUT = 0.20  # a consecutive-frame SigLIP 2 cosine distance above this counts as a cut; sensitivity CUT_ALT
CUT_ALT = (0.15, 0.30)
# Wording of each temporal aspect, fixed in A7 before any caption was searched (whole words, case-insensitive)
MENTION = {
    "speed": r"fast|faster|fastest|slow|slower|slowest|slowly|quick|quickly|rapid|rapidly|swift|swiftly|speed\w*|"
             r"pace|gradual|gradually|sudden|suddenly|steady|steadily|accelerat\w*|decelerat\w*|brisk\w*|leisurely",
    "direction": r"left|leftward|right|rightward|up|upward|upwards|down|downward|downwards|toward|towards|away|"
                 r"forward|forwards|backward|backwards|clockwise|counterclockwise|anticlockwise|ascend\w*|"
                 r"descend\w*|rise|rises|rising|fall|falls|falling|approach\w*|recede\w*|retreat\w*|revers\w*",
    "order": r"first|then|after|afterward|afterwards|before|next|later|finally|followed|follows|begins|ends|"
             r"earlier|subsequently|until|once|cuts|switches|transitions",
    "attribute_change": r"becomes|become|becoming|turns|turning|changes|changing|changed|grows|growing|shrinks|"
                        r"shrinking|fades|fading|brightens|darkens|increases|decreases|transforms|melts|melting|"
                        r"expands|expanding|swells|dissolves|fills|empties|emerges",
}
MENTION_RE = {k: re.compile(rf"\b(?:{v})\b", re.I) for k, v in MENTION.items()}
NUMERIC = ["motion", "cuts", "log_windows", "mention", "coverage"]
MODELS = {"M1 task + pair": ["dim", "reader", "family"],
          "M2 M1 + video": ["dim", "reader", "family", "motion", "cuts", "log_windows"],
          "M3 M1 + captions": ["dim", "reader", "family", "mention", "coverage"],
          "M4 M1 + video + captions": ["dim", "reader", "family", "motion", "cuts", "log_windows", "mention",
                                       "coverage"],
          "M5 M4 + aspect x motion, aspect x cuts": ["dim", "reader", "family", "motion", "cuts", "log_windows",
                                                     "mention", "coverage", "dim*motion", "dim*cuts"],
          "MT M1 + video type (TempCompass's label; reference)": ["dim", "reader", "family", "type"]}
SIZE = {k: i for i, k in enumerate(MODELS)}  # the selection rule's order, smaller first; MT is never selected


def video_scores(vids: list[str]) -> dict[str, dict]:
    out = {}
    for v in vids:
        flat = np.concatenate(PR.frame_embs(v))
        dist = 1 - np.sum(flat[1:] * flat[:-1], axis=1)
        out[v] = {"motion": float(dist.mean()), "log_windows": float(np.log(len(tc.windows(tc.duration_s(v))))),
                  **{f"cuts@{c}": float(np.log1p((dist > c).sum())) for c in (CUT, *CUT_ALT)}}
    return out


def index_scores(vids: list[str]) -> dict[tuple[str, str], dict]:
    """Per (index arm, video): coverage and, per temporal aspect, the share of windows that mention it."""
    from llm.local_embedder import SiglipTextEmbedder
    text = SiglipTextEmbedder("google/siglip2-base-patch16-224")
    out = {}
    for arm in sorted(FAMILY):
        for v in vids:
            caps = captions(v, R.CAPTIONERS[arm])
            if caps is None:
                continue
            wins = PR.frame_embs(v)
            c = PR.unit(text.get_embeddings(v, caps))
            out[arm, v] = {"coverage": float(np.mean([c[p] @ PR.unit(wins[p].mean(0)) for p in range(len(c))])),
                           **{f"mention_{k}": float(np.mean([bool(r.search(x)) for x in caps]))
                              for k, r in MENTION_RE.items()}}
    return out


def rows(cut: float = CUT):
    """All (reader, index, question) rows with both answers: index right, frames right, and the features."""
    import pandas as pd
    wide, vtype, clus = R.confirm_frame()
    q = {x["item"]: x for x in tc.items("confirm")}
    vids = sorted({q[i]["video_id"] for i in wide.index})
    vs, ix = video_scores(vids), index_scores(vids)
    out = []
    for reader, arm in PAIRS:
        a, f = f"{reader}:{arm}", f"{reader}:frames5"
        for i in wide.index:
            v = q[i]["video_id"]
            if np.isnan(wide.at[i, a]) or np.isnan(wide.at[i, f]) or (arm, v) not in ix:
                continue
            dim = q[i]["dim"]
            out.append({"pair": a, "reader": reader, "family": FAMILY[arm], "item": i, "video_id": v, "dim": dim,
                        "type": vtype[i], "cluster": clus[i], "index_ok": int(wide.at[i, a]),
                        "frames_ok": int(wide.at[i, f]), "motion": vs[v]["motion"], "cuts": vs[v][f"cuts@{cut}"],
                        "log_windows": vs[v]["log_windows"], "coverage": ix[arm, v]["coverage"],
                        "mention": ix[arm, v][f"mention_{dim}"] if dim in MENTION else 0.0})
    df = pd.DataFrame(out)
    df["won"] = df.index_ok
    return df, vs


def design(d, features: list[str], ref=None):
    """Feature matrix: one-hot categories (drop first level), numbers, and dim*x interactions."""
    import pandas as pd
    cats = [f for f in features if f in ("dim", "reader", "family", "type")]
    X = pd.get_dummies(d[cats].astype(str), drop_first=True).astype(float)
    for f in features:
        if f in NUMERIC:
            X[f] = d[f].to_numpy()
        elif "*" in f:
            _, x = f.split("*")
            for k in tc.DIMS[1:]:
                X[f"{k}*{x}"] = (d.dim == k).to_numpy() * d[x].to_numpy()
    if ref is not None:
        X = X.reindex(columns=ref, fill_value=0.0)
    return X


def fit(Xtr, y):
    from sklearn.linear_model import LogisticRegression
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
    m = LogisticRegression(C=1.0, max_iter=2000).fit((Xtr - mu) / sd, y)
    return lambda X: m.predict_proba((X - mu) / sd)[:, 1], m, mu, sd


def oof(dis, features: list[str], leave_pair: bool) -> np.ndarray:
    """Out-of-fold P(index won) on the disagreeing rows: folds grouped by clip cluster, and, with leave_pair, the
    training rows of each fold also exclude the test row's (reader, index) pair."""
    from sklearn.model_selection import GroupKFold
    Xv, y = design(dis, features).to_numpy(), dis.won.to_numpy()
    p = np.full(len(dis), np.nan)
    for tr, te in GroupKFold(10).split(Xv, y, dis.cluster):
        if not leave_pair:
            p[te] = fit(Xv[tr], y[tr])[0](Xv[te])
            continue
        for pair in dis.pair.iloc[te].unique():
            t = te[dis.pair.to_numpy()[te] == pair]
            r = tr[dis.pair.to_numpy()[tr] != pair]
            p[t] = fit(Xv[r], y[r])[0](Xv[t])
    return p


def cluster_ci(stat, clusters, n_boot: int = 2000, seed: int = 0):
    """Percentile 95% CI of stat(rows) over clip-cluster resamples; stat takes an index array."""
    clusters = np.asarray(clusters)
    keys = np.unique(clusters)
    at = {k: np.flatnonzero(clusters == k) for k in keys}
    rng = np.random.default_rng(seed)
    bs = [stat(np.concatenate([at[k] for k in rng.choice(keys, len(keys))])) for _ in range(n_boot)]
    return stat(np.arange(len(clusters))), *np.nanpercentile(bs, [2.5, 97.5])


def cmd_a7(args):
    df, vs = rows()
    dis = df[df.index_ok != df.frames_ok].reset_index(drop=True)
    print(f"Rows: {len(df)} (reader, index, question) over {df.pair.nunique()} pairs; {len(dis)} disagree, "
          f"index won {dis.won.sum()}.")
    print("Per pair (index won / frames won):")
    for p, g in dis.groupby("pair"):
        print(f"  {p:28s} {g.won.sum():3d} / {(1 - g.won).sum():3d}")
    import pandas as pd
    vdf = pd.DataFrame(vs).T.assign(type=[tc.video_type(v) for v in vs])
    print("\nVideo scores by type (median; share of videos with a cut at each threshold):")
    print(vdf.groupby("type").agg(motion=("motion", "median"),
                                  **{f"cut@{c}": (f"cuts@{c}", lambda s: (s > 0).mean()) for c in (CUT, *CUT_ALT)}
                                  ).round(3).to_string())
    print("\nMention by aspect and captioner family (mean share of windows; rows of that aspect):")
    print(df[df.dim.isin(MENTION)].groupby(["dim", "family"]).mention.mean().unstack().round(3).to_string())

    from sklearn.metrics import roc_auc_score
    y, cl = dis.won.to_numpy(), dis.cluster.to_numpy()
    print("\n1. Out-of-fold AUC on the disagreeing rows, 95% CIs over clip clusters:")
    preds = {}
    for name, feats in MODELS.items():
        for lp in (False, True):
            p = oof(dis, feats, lp)
            preds[name, lp] = p
            e, lo, hi = cluster_ci(lambda i: roc_auc_score(y[i], p[i]) if len(set(y[i])) > 1 else np.nan, cl)
            print(f"  {name:52s} {'pair held out' if lp else 'clusters only':13s} {e:.3f} [{lo:.3f}, {hi:.3f}]")

    auc = {k: roc_auc_score(y, preds[k, False]) for k in MODELS}
    best = max((k for k in MODELS if not k.startswith("MT")), key=auc.get)
    chosen = min((k for k in MODELS if not k.startswith("MT") and auc[k] >= auc[best] - 0.01), key=SIZE.get)
    print(f"\n   Selection rule: highest clusters-only AUC {best} ({auc[best]:.3f}); smallest within 0.01: {chosen}")

    print("\n2. Decision accuracy over all rows (the index is chosen when P(index won) > 0.5; agreeing rows score"
          " the same either way), out of fold, 95% CIs over clip clusters:")
    allcl = df.cluster.to_numpy()
    key = dict(zip(zip(dis.pair, dis.item), range(len(dis))))
    pos = np.array([key.get((p, i), -1) for p, i in zip(df.pair, df.item)])
    fr, ix = df.frames_ok.to_numpy(), df.index_ok.to_numpy()
    oracle = np.maximum(fr, ix)
    pair_best = df.groupby("pair")[["index_ok", "frames_ok"]].mean().pipe(lambda t: t.index_ok > t.frames_ok)
    fixed = np.where(df.pair.map(pair_best).to_numpy(), ix, fr)
    for lp in (False, True):
        p = preds[chosen, lp]
        use = np.where(pos >= 0, p[np.maximum(pos, 0)] > 0.5, False)
        pol = np.where(use, ix, fr)
        tag = "pair held out" if lp else "clusters only"
        for name, base in (("always frames", fr), ("always index", ix), ("best fixed per pair (in sample)", fixed)):
            e, lo, hi = cluster_ci(lambda i: pol[i].mean() - base[i].mean(), allcl)
            print(f"  {chosen} ({tag}) - {name}: {e:+.4f} [{lo:+.4f}, {hi:+.4f}]")
        head = (oracle.mean() - fixed.mean())
        print(f"    accuracy: policy {pol.mean():.4f}, frames {fr.mean():.4f}, index {ix.mean():.4f}, best fixed "
              f"{fixed.mean():.4f}, oracle {oracle.mean():.4f}; headroom captured "
              f"{(pol.mean() - fixed.mean()) / head:.2f}")
        for p_ in sorted(df.pair.unique()):
            m = df.pair.to_numpy() == p_
            print(f"      {p_:28s} policy {pol[m].mean():.3f}  frames {fr[m].mean():.3f}  index {ix[m].mean():.3f}"
                  f"  oracle {oracle[m].mean():.3f}  index chosen {use[m].mean():.2f}")

    print(f"\n3. The formula: {chosen}, fit on all disagreeing rows (standardized coefficients; 95% CIs over clip "
          "clusters, 1,000 resamples):")
    X = design(dis, MODELS[chosen])
    Xv = X.to_numpy()
    _, m, mu, sd = fit(Xv, y)
    rng = np.random.default_rng(0)
    keys = np.unique(cl)
    at = {k: np.flatnonzero(cl == k) for k in keys}
    bs = []
    for _ in range(1000):
        i = np.concatenate([at[k] for k in rng.choice(keys, len(keys))])
        if len(set(y[i])) > 1:
            bs.append(fit(Xv[i], y[i])[1].coef_[0])
    lo, hi = np.percentile(bs, [2.5, 97.5], axis=0)
    print(f"  intercept {m.intercept_[0]:+.3f}")
    for c, b, l, h, a, s in zip(X.columns, m.coef_[0], lo, hi, mu, sd):
        print(f"  {c:28s} {b:+.3f} [{l:+.3f}, {h:+.3f}]   (mean {a:.4f}, sd {s:.4f})")

    print(f"\n4. Cut threshold sensitivity ({chosen} if it uses cuts, else M2), clusters-only AUC:")
    feats = MODELS[chosen] if "cuts" in MODELS[chosen] else MODELS["M2 M1 + video"]
    for c in CUT_ALT:
        d2 = rows(c)[0]
        d2 = d2[d2.index_ok != d2.frames_ok].reset_index(drop=True)
        print(f"  cut {c}: {roc_auc_score(d2.won, oof(d2, feats, False)):.3f}")

    print("\n5. By aspect and by video type, clusters-only out-of-fold AUC of the chosen model (descriptive):")
    p = preds[chosen, False]
    for col in ("dim", "type"):
        for k, g in dis.groupby(col):
            if g.won.nunique() > 1:
                print(f"  {col} {k:18s} {roc_auc_score(g.won, p[g.index]):.3f} (n={len(g)}, index won {g.won.sum()})")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("a7")
    args = ap.parse_args()
    {"a7": cmd_a7}[args.cmd](args)


if __name__ == "__main__":
    main()
