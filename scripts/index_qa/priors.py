#!/usr/bin/env python3
"""A6 (research log): can a question-blind score of a video predict where its captions lose the frames' answer?

Scores (per confirmatory video, from what an index builder has before any question; no model calls):
    motion    mean SigLIP 2 cosine distance between consecutive frames (1 s apart) over the whole video
    coverage  per captioner: mean over windows of the SigLIP 2 cosine between the window's caption and the mean of
              its frames (text tower google/siglip2-base-patch16-224, in the frames' space; 64-token truncation)
Frames are embedded with the SigLIP 2 image tower (timm vit_base_patch16_siglip_224.v2_webli) and cached in
local/tempcompass/siglip2_frames/ (about 4,000 frames; minutes on CPU).

    .venv/bin/python scripts/index_qa/priors.py scores   # the scores and the coverage gate; reads no reader row
    .venv/bin/python scripts/index_qa/priors.py a6       # A6's statistics; output to results/index_qa/reports/a6.txt
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parents[2] / "src")]

from index_qa import reader as R  # noqa: E402
from index_qa import tempcompass as tc  # noqa: E402

CACHE = tc.TC / "siglip2_frames"
SCORES = tc.OUT / "reports/a6_scores.csv"
# reader: (its self-captions arm, its frames arm, its captions' directory under results/index_qa/captions)
READERS = {"haiku": ("haiku:selfcaptions", "haiku:frames5", "claude-haiku-5-5-self"),
           "sonnet": ("sonnet:sonnetcaptions", "sonnet:frames5", "claude-sonnet-5-5-frames")}
# A6's primary scores: (column, sign), sign +1 if a higher score is predicted to mean more losses
PRIMARY = [("motion", +1), ("coverage", -1)]
GATE = 0.85  # coverage is tested only if a caption beats a random other video's window this often
LEVEL_PRIMARY = 1 - 0.05 / len(PRIMARY)


def unit(x):
    x = np.asarray(x, dtype=np.float32)
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-12)


def frame_embs(video_id: str) -> list[np.ndarray]:
    """Per window, its frames' unit SigLIP 2 image embeddings (frames x 768), cached per video."""
    wins = tc.window_frames(video_id)
    cache = CACHE / f"{video_id}.npy"
    if cache.exists():
        flat = np.load(cache)
    else:
        import timm
        import torch
        from PIL import Image
        global _M, _TF
        if "_M" not in globals():
            _M = timm.create_model("vit_base_patch16_siglip_224.v2_webli", pretrained=True, num_classes=0).eval()
            _TF = timm.data.create_transform(**timm.data.resolve_model_data_config(_M), is_training=False)
        x = torch.stack([_TF(Image.open(p).convert("RGB")) for w in wins for p in w])
        with torch.inference_mode():
            flat = unit(_M(x).numpy())
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache, flat)
    cuts = np.cumsum([len(w) for w in wins])[:-1]
    return np.split(flat, cuts)


def window_captions(directory: str, video_id: str, n: int) -> list[str]:
    return [json.loads((tc.OUT / "captions" / directory / video_id / f"{p:02d}.json").read_text())["caption"]
            for p in range(n)]


def compute_scores():
    """One row per confirmatory video: motion, coverage per reader's captioner, windows, video type, clip cluster;
    and the coverage gate per captioner (own window vs. a random window of another video, seed 0)."""
    import pandas as pd
    from llm.local_embedder import SiglipTextEmbedder
    text = SiglipTextEmbedder("google/siglip2-base-patch16-224")
    vids = sorted({q["video_id"] for q in tc.items("confirm")})
    clus = dict(zip(vids, R.clip_clusters(vids)))
    rows, wins, caps = [], {}, {}
    for v in vids:
        wins[v] = frame_embs(v)
        flat = np.concatenate(wins[v])
        row = {"video_id": v, "type": tc.video_type(v), "cluster": clus[v], "windows": len(wins[v]),
               "motion": float(np.mean(1 - np.sum(flat[1:] * flat[:-1], axis=1)))}
        for reader, (_, _, directory) in READERS.items():
            c = unit(text.get_embeddings(v, window_captions(directory, v, len(wins[v]))))
            caps[reader, v] = c
            row[f"coverage_{reader}"] = float(np.mean([c[p] @ unit(wins[v][p].mean(0)) for p in range(len(c))]))
        rows.append(row)
    df = pd.DataFrame(rows).set_index("video_id")
    rng = np.random.default_rng(0)
    keys = [(v, p) for v in vids for p in range(len(wins[v]))]
    gate = {}
    for reader in READERS:
        hits = []
        for v, p in keys:
            while (o := keys[rng.integers(len(keys))])[0] == v:
                pass
            own, other = unit(wins[v][p].mean(0)), unit(wins[o[0]][o[1]].mean(0))
            hits.append(caps[reader, v][p] @ own > caps[reader, v][p] @ other)
        gate[reader] = float(np.mean(hits))
    long_ = {r: float(np.mean([len(text.tokenizer(c.lower())["input_ids"]) > 64
                               for v in vids for c in window_captions(d, v, len(wins[v]))]))
             for r, (_, _, d) in READERS.items()}
    return df, gate, long_


def cmd_scores(args):
    df, gate, long_ = compute_scores()
    SCORES.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(SCORES)
    print(f"{len(df)} videos; scores in {SCORES}")
    print(df.groupby("type")[["windows", "motion", "coverage_haiku", "coverage_sonnet"]].median().round(3))
    for r in READERS:
        print(f"coverage gate, {r}'s captions: own window beats a random other video's window {gate[r]:.3f} "
              f"(gate {GATE}); captions over 64 tokens {long_[r]:.3f}")


def strat_auc(score, label, strata) -> float:
    """AUC over (loss, kept) pairs in the same stratum: P(score of the loss > score of the kept), ties count 1/2."""
    num = den = 0.0
    for s in np.unique(strata):
        m = strata == s
        pos, neg = score[m & (label == 1)], score[m & (label == 0)]
        if len(pos) and len(neg):
            num += (pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()
            den += len(pos) * len(neg)
    return num / den if den else float("nan")


def auc_boot(score, label, strata, clusters, n_boot: int = 4000, seed: int = 0, level: float = 0.95):
    """Stratified AUC and a percentile CI from resampling clip clusters (all rows of a cluster together)."""
    score, label, strata, clusters = map(np.asarray, (score, label, strata, clusters))
    keys = np.unique(clusters)
    rows = {k: np.flatnonzero(clusters == k) for k in keys}
    rng = np.random.default_rng(seed)
    bs = []
    for _ in range(n_boot):
        i = np.concatenate([rows[k] for k in rng.choice(keys, len(keys))])
        bs.append(strat_auc(score[i], label[i], strata[i]))
    lo, hi = np.nanpercentile(bs, [50 * (1 - level), 50 * (1 + level)])
    return strat_auc(score, label, strata), float(lo), float(hi)


def loss_rows(scores):
    """One row per (reader, question) where the reader's frames answer is right: lost = its self-captions answer is
    wrong. Also the deployment term (Sonnet frames right, Haiku on Sonnet captions wrong) as reader 'index'."""
    import pandas as pd
    wide, vtype, clus = R.confirm_frame()
    q = {x["item"]: x for x in tc.items("confirm")}
    terms = {r: (cap, frames, f"coverage_{r}") for r, (cap, frames, _) in READERS.items()}
    terms["index"] = ("haiku:sonnetcaptions", "sonnet:frames5", "coverage_sonnet")
    out = []
    for reader, (cap, frames, cov) in terms.items():
        for i in wide.index:
            if wide.at[i, frames] == 1 and not np.isnan(wide.at[i, cap]):
                v = q[i]["video_id"]
                out.append({"reader": reader, "item": i, "video_id": v, "type": vtype[i], "cluster": clus[i],
                            "dim": q[i]["dim"], "lost": int(wide.at[i, cap] == 0), "motion": scores.at[v, "motion"],
                            "coverage": scores.at[v, cov], "windows": scores.at[v, "windows"]})
    return pd.DataFrame(out), wide, vtype, clus


def show(name, d, col, sign, strata, level=0.95, pred=None):
    est, lo, hi = auc_boot(sign * d[col].to_numpy(), d.lost.to_numpy(), strata, d.cluster.to_numpy(), level=level)
    p = f"; prediction {pred:.2f}" if pred is not None else ""
    print(f"  {name}: AUC {est:.3f} [{lo:.3f}, {hi:.3f}] at {level:.4f} (losses {d.lost.sum()} of {len(d)}{p})")


def oof_auc(d, features: list[str], folds: int = 10) -> np.ndarray:
    """Out-of-fold P(lost) from an L2 logistic regression (C = 1, standardized numeric features, one-hot
    categoricals), folds grouped by clip cluster."""
    import pandas as pd
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    num = [f for f in features if f in ("motion", "coverage", "windows")]
    X = pd.get_dummies(d[[f for f in features if f not in num]].astype(str), drop_first=True).astype(float)
    for f in num:
        X[f] = d[f].to_numpy()
    X = X.to_numpy()
    p = np.zeros(len(d))
    for tr, te in GroupKFold(folds).split(X, d.lost, d.cluster):
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9
        m = LogisticRegression(C=1.0, max_iter=1000).fit((X[tr] - mu) / sd, d.lost.to_numpy()[tr])
        p[te] = m.predict_proba((X[te] - mu) / sd)[:, 1]
    return p


def cmd_a6(args):
    import pandas as pd
    if not SCORES.exists():
        sys.exit(f"run `priors.py scores` first ({SCORES})")
    scores = pd.read_csv(SCORES).set_index("video_id")
    df, wide, vtype, clus = loss_rows(scores)
    own = df[df.reader.isin(READERS)]
    strata = (own.type + ":" + own.reader).to_numpy()
    print(f"Rows: {len(own)} (reader, question) pairs with the frames answer right; {own.lost.sum()} lost "
          f"(haiku {own[own.reader == 'haiku'].lost.sum()}, sonnet {own[own.reader == 'sonnet'].lost.sum()}).")
    print(f"\n1. Primary: stratified AUC (video type x reader), {LEVEL_PRIMARY:.4f} CIs over clip clusters:")
    for (col, sign), pred in zip(PRIMARY, (0.60, 0.58)):
        show(f"{col} ({'higher' if sign > 0 else 'lower'} = more losses)", own, col, sign, strata,
             LEVEL_PRIMARY, pred)
    print("\n2. By reader (stratified by video type), 95%:")
    for r in READERS:
        d = own[own.reader == r]
        for col, sign in PRIMARY:
            show(f"{r}, {col}", d, col, sign, d.type.to_numpy())
    d = df[df.reader == "index"]
    print("\n3. Deployment term: Haiku on Sonnet captions loses Sonnet's frames answer (stratified by type), 95%:")
    for col, sign in PRIMARY:
        show(col, d, col, sign, d.type.to_numpy())
    plain = own[own.type == "plain"]
    print("\n4. Plain videos only (stratified by reader), 95%:")
    for col, sign in PRIMARY:
        show(col, plain, col, sign, plain.reader.to_numpy())
    print("\n5. By aspect (stratified by aspect x reader within each aspect; descriptive), 95%:")
    for dim in tc.DIMS:
        d = own[own.dim == dim]
        for col, sign in PRIMARY:
            show(f"{dim}, {col}", d, col, sign, (d.type + ":" + d.reader).to_numpy())
    print("\n6. Gains (frames wrong, self-captions right), stratified by type x reader; descriptive, 95%:")
    gains = []
    q = {x["item"]: x for x in tc.items("confirm")}
    for r, (cap, frames, _) in READERS.items():
        for i in wide.index:
            if wide.at[i, frames] == 0 and not np.isnan(wide.at[i, cap]):
                v = q[i]["video_id"]
                gains.append({"type": vtype[i], "reader": r, "cluster": clus[i], "lost": int(wide.at[i, cap] == 1),
                              "motion": scores.at[v, "motion"], "coverage": scores.at[v, f"coverage_{r}"]})
    g = pd.DataFrame(gains)
    for col, sign in PRIMARY:
        show(f"{col} (label = gained; sign as for losses)", g, col, sign, (g.type + ":" + g.reader).to_numpy())
    print("\n7. The classifier: out-of-fold AUC (10 folds grouped by clip cluster), 95% CIs over clip clusters:")
    sets = {"A aspect + reader": ["dim", "reader"],
            "B A + motion + coverage": ["dim", "reader", "motion", "coverage"],
            "C A + video type (TempCompass's label)": ["dim", "reader", "type"],
            "D C + motion + coverage": ["dim", "reader", "type", "motion", "coverage"]}
    oof = {k: oof_auc(own.reset_index(drop=True), f) for k, f in sets.items()}
    one = np.zeros(len(own), dtype=object)
    cl = own.cluster.to_numpy()
    for k, p in oof.items():
        e, lo, hi = auc_boot(p, own.lost.to_numpy(), one, cl)
        print(f"  {k}: {e:.3f} [{lo:.3f}, {hi:.3f}]")
    keys = np.unique(cl)
    rows = {k: np.flatnonzero(cl == k) for k in keys}
    rng = np.random.default_rng(0)
    y = own.lost.to_numpy()
    for a, b in [("B", "A"), ("D", "C")]:
        pa, pb = next(v for k, v in oof.items() if k[0] == a), next(v for k, v in oof.items() if k[0] == b)
        bs = []
        for _ in range(4000):
            i = np.concatenate([rows[k] for k in rng.choice(keys, len(keys))])
            bs.append(strat_auc(pa[i], y[i], one[i]) - strat_auc(pb[i], y[i], one[i]))
        lo, hi = np.percentile(bs, [2.5, 97.5])
        print(f"  {a} - {b}: {strat_auc(pa, y, one) - strat_auc(pb, y, one):+.3f} [{lo:+.3f}, {hi:+.3f}]")
    print("\n8. Loss rate by score tertile (tertiles over the 297 videos), by video type:")
    for col, _ in PRIMARY:
        edges = np.quantile(scores["motion"] if col == "motion" else
                            pd.concat([scores.coverage_haiku, scores.coverage_sonnet]), [1 / 3, 2 / 3])
        tab = own.assign(tertile=np.digitize(own[col], edges)).pivot_table(
            index="type", columns="tertile", values="lost", aggfunc=lambda s: f"{s.mean():.2f} ({len(s)})")
        print(f"  {col} (edges {edges[0]:.3f}, {edges[1]:.3f}; columns low / mid / high):\n{tab.to_string()}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("scores")
    sub.add_parser("a6")
    args = ap.parse_args()
    {"scores": cmd_scores, "a6": cmd_a6}[args.cmd](args)


if __name__ == "__main__":
    main()
