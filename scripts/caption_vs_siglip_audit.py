#!/usr/bin/env python3
"""
What do the dense captions keep and lose relative to the SigLIP frame embeddings? No masking.

Q1 grounding: does caption t retrieve frame t among the video's 60 frames (SigLIP 2 text vs image)?
   Also the lag profile: which frame offset does caption t match best?
Q2 shared information: ridge maps captions -> frames and frames -> captions on within-video variation
   (each video's mean removed), held out by channel. Reports R^2 per direction and per video.
Q3 change structure: per-second change in captions (MPNet) vs frames (SigLIP). Correlation per video,
   and how often the largest visual changes coincide with large caption changes (and the reverse).

Per-video metrics are then correlated with reconstruction outcomes (forced-choice and per-second rank),
to test whether caption/frame disagreement explains where text-based reconstruction falls behind.
Criteria for GO/NO-GO are in the assessment doc (2026-10-07 decision section).

Usage (from repo root):
    .venv/bin/python scripts/caption_vs_siglip_audit.py
Outputs: results/caption_audit/{per_video,per_second}.csv and printed summary.
"""
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from llm.local_embedder import LocalEmbedder, SiglipTextEmbedder

T = 60
OUT = Path("results/caption_audit")
VIS_DIR = Path("local/wild_videos_embs_siglip")
CAP_DIRS = [Path("datasets/wildQA/captions__wild4"), Path("datasets/wildQA/captions__wild5")]
LAGS = range(-10, 11)
rng = np.random.default_rng(2025)


def unit(x):
    x = np.asarray(x, dtype=np.float32)
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-12)


def channel(vid):
    return re.match(r"^(.*?)_\d+", vid.replace("'", "_")).group(1)


def load_videos():
    vids = sorted(pd.read_json("results/forced_choice/baselines.jsonl", lines=True).vid.unique())
    # The stored frames are SigLIP 2 (timm v2_webli), so captions must use the SigLIP 2 text tower.
    mp, sg = LocalEmbedder("all-mpnet-base-v2"), SiglipTextEmbedder("google/siglip2-base-patch16-224")
    data = {}
    for vid in vids:
        cf = next((d / f"{vid}.json" for d in CAP_DIRS if (d / f"{vid}.json").exists()), None)
        vf = VIS_DIR / f"{vid.replace(chr(39), '_')}.npy"
        if cf is None or not vf.exists():
            continue
        caps = [c["caption"] for c in json.load(open(cf))["captions"][:T]]
        V = np.load(vf)[:T]
        if len(caps) != T or len(V) != T:
            continue
        data[vid] = dict(caps=caps, V=unit(V),
                         M=unit(mp.get_embeddings(f"{vid}_dense_caps", caps)),
                         S=unit(sg.get_embeddings(f"{vid}_dense_caps", caps)))
    return data


def q1(d):
    """Caption t -> frame t retrieval among the 60 frames, and the lag profile of z-scored similarity."""
    sim = d["S"] @ d["V"].T  # rows: captions, cols: frames
    z = (sim - sim.mean(1, keepdims=True)) / (sim.std(1, keepdims=True) + 1e-9)
    ranks = 1 + (sim > np.diag(sim)[:, None]).sum(1)
    lag = {k: np.mean([z[t, t + k] for t in range(T) if 0 <= t + k < T]) for k in LAGS}
    # Chance-corrected: same retrieval with frames circularly shifted by a random offset >= 10 s.
    sh = int(rng.integers(10, T - 10))
    sim_sh = d["S"] @ np.roll(d["V"], sh, axis=0).T
    ranks_sh = 1 + (sim_sh > np.diag(sim_sh)[:, None]).sum(1)
    return dict(q1_mrr=np.mean(1 / ranks), q1_r_pm2=np.mean([np.abs(np.argmax(sim[t]) - t) <= 2 for t in range(T)]),
                q1_mrr_shift=np.mean(1 / ranks_sh), q1_best_lag=max(lag, key=lag.get)), lag, ranks


def changes(X):
    return 1 - np.sum(X[1:] * X[:-1], axis=1)  # cosine distance between consecutive seconds


def q2(data):
    vids = list(data)
    groups = np.repeat([channel(v) for v in vids], T)
    vidx = np.repeat(np.arange(len(vids)), T)
    center = lambda k: np.concatenate([data[v][k] - data[v][k].mean(0) for v in vids])
    X = {"mpnet": center("M"), "siglip_text": center("S"), "frames": center("V")}
    res, per_vid = {}, {}
    for src, dst in [("mpnet", "frames"), ("siglip_text", "frames"), ("frames", "mpnet"), ("frames", "siglip_text")]:
        pred = np.zeros_like(X[dst])
        for tr, te in GroupKFold(5).split(X[src], groups=groups):
            pred[te] = Ridge(alpha=10.0).fit(X[src][tr], X[dst][tr]).predict(X[src][te])
        resid, tot = ((X[dst] - pred) ** 2).sum(1), (X[dst] ** 2).sum(1)
        res[f"{src}->{dst}"] = 1 - resid.sum() / tot.sum()
        per_vid[f"q2_{src}_to_{dst}"] = [1 - resid[vidx == i].sum() / tot[vidx == i].sum() for i in range(len(vids))]
    return res, pd.DataFrame(per_vid, index=vids)


def q3(d):
    dc, dv = changes(d["M"]), changes(d["V"])
    return dict(q3_spearman=spearmanr(dc, dv)[0], cap_change_mean=dc.mean(), vis_change_mean=dv.mean(),
                cap_identical_frac=np.mean(dc < 1e-4)), dc, dv


def coincidence(per_sec, q=0.9, tol=1):
    """Of the top-10% visual changes, how many have a top-10% caption change within +-tol s (and the reverse)."""
    tv, tc = per_sec.dv.quantile(q), per_sec.dc.quantile(q)
    out = {}
    for name, a, b in [("vis_events_with_caption_change", "dv", "dc"), ("cap_events_with_visual_change", "dc", "dv")]:
        thr_a, thr_b = (tv, tc) if a == "dv" else (tc, tv)
        hits = []
        for vid, g in per_sec.groupby("vid"):
            A, B = g[a].to_numpy(), g[b].to_numpy()
            for t in np.where(A >= thr_a)[0]:
                hits.append(any(B[max(0, t - tol):t + tol + 1] >= thr_b))
        out[name] = np.mean(hits)
    # Chance: same rate with caption changes shuffled within each video.
    sh = per_sec.copy()
    sh["dc"] = sh.groupby("vid").dc.transform(lambda s: rng.permutation(s.to_numpy()))
    hits = []
    for vid, g in sh.groupby("vid"):
        A, B = g.dv.to_numpy(), g.dc.to_numpy()
        hits += [any(B[max(0, t - tol):t + tol + 1] >= tc) for t in np.where(A >= tv)[0]]
    out["chance"] = np.mean(hits)
    return out


def outcomes():
    """Per-video reconstruction outcomes: forced-choice accuracy and per-second rank gaps (lower rank = better)."""
    fc = pd.read_json("results/forced_choice/baselines.jsonl", lines=True)
    for m in ["text_copy", "vis_copy"]:
        fc[m + "_ok"] = [float(np.argmax(s) == 0) for s in fc[m]]
    o = fc.groupby("vid")[["text_copy_ok", "vis_copy_ok"]].mean()
    o["fc_vis_minus_text"] = o.vis_copy_ok - o.text_copy_ok
    m = pd.read_csv("results/unified_benchmark_master.csv")
    m = m[m.width <= 16].assign(video_id=lambda x: x.video_id.str.replace("'", "_"))
    r = m.pivot_table(index="video_id", columns="method", values="mean_rank", aggfunc="mean")
    o.index = o.index.str.replace("'", "_")
    o["rank_capcopy_minus_viscopy"] = r["Caption_RepeatClosest"] - r["Visual_SigLIP_RepeatClosest"]
    o["rank_llama_minus_capcopy"] = r["Llama-3.1-8B"] - r["Caption_RepeatClosest"]
    return o


def boot_spearman(df, x, y, n=2000):
    df = df[[x, y, "chan"]].dropna()
    chans = df.chan.unique()
    by = {c: g for c, g in df.groupby("chan")}
    vals = [spearmanr(*pd.concat([by[c] for c in rng.choice(chans, len(chans))])[[x, y]].T.to_numpy())[0]
            for _ in range(n)]
    return spearmanr(df[x], df[y])[0], *np.nanpercentile(vals, [2.5, 97.5])


def partial_spearman(df, x, y, z):
    """Spearman correlation of x and y after regressing the ranks of the controls z out of both."""
    R = lambda s: rankdata(s)
    Z = np.column_stack([np.ones(len(df))] + [R(df[c]) for c in z])
    res = lambda v: R(v) - Z @ np.linalg.lstsq(Z, R(v), rcond=None)[0]
    return np.corrcoef(res(df[x]), res(df[y]))[0, 1]


def boot_partial(df, x, y, z, n=2000):
    df = df[[x, y, "chan"] + z].dropna()
    by = {c: g for c, g in df.groupby("chan")}
    chans = list(by)
    vals = [partial_spearman(pd.concat([by[c] for c in rng.choice(chans, len(chans))]), x, y, z) for _ in range(n)]
    return partial_spearman(df, x, y, z), *np.percentile(vals, [2.5, 97.5])


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    data = load_videos()
    print(f"{len(data)} videos, {len({channel(v) for v in data})} channels")
    rows, lags, secs = [], [], []
    for vid, d in data.items():
        r1, lag, ranks = q1(d)
        r3, dc, dv = q3(d)
        rows.append(dict(vid=vid, chan=channel(vid), **r1, **r3))
        lags.append(lag)
        secs += [dict(vid=vid, t=t, q1_rank=ranks[t], dc=dc[t] if t < T - 1 else np.nan,
                      dv=dv[t] if t < T - 1 else np.nan) for t in range(T)]
    pv = pd.DataFrame(rows).set_index("vid")
    q2_tot, q2_pv = q2(data)
    pv = pv.join(q2_pv)
    per_sec = pd.DataFrame(secs)
    per_sec.to_csv(OUT / "per_second.csv", index=False)

    print("\n=== Q1 grounding (caption t -> frame t among 60; chance MRR 0.079) ===")
    print(f"MRR {pv.q1_mrr.mean():.3f} (shifted-frames control {pv.q1_mrr_shift.mean():.3f}); "
          f"best frame within +-2 s: {pv.q1_r_pm2.mean():.3f} (chance ~{5 / 60:.3f})")
    lag = pd.DataFrame(lags).mean()
    print("lag profile (mean z of sim(caption t, frame t+k)):",
          " ".join(f"{k:+d}:{v:.2f}" for k, v in lag.items() if k % 2 == 0))
    print("per-video best lag distribution:", pv.q1_best_lag.value_counts().sort_index().to_dict())
    print("Q1 MRR by channel (lowest 5):", pv.groupby("chan").q1_mrr.mean().nsmallest(5).round(3).to_dict())

    print("\n=== Q2 shared within-video variation (held-out channels, R^2) ===")
    for k, v in q2_tot.items():
        print(f"  {k}: {v:.3f}")

    print("\n=== Q3 change structure ===")
    print(f"per-video Spearman(caption change, frame change): mean {pv.q3_spearman.mean():.3f}, "
          f"median {pv.q3_spearman.median():.3f}")
    print(f"identical consecutive captions: {pv.cap_identical_frac.mean():.3f} of transitions")
    print("top-10% event coincidence (+-1 s):", {k: round(v, 3) for k, v in coincidence(per_sec.dropna()).items()})

    o = outcomes()
    pv = pv.join(o, how="left")
    pv.to_csv(OUT / "per_video.csv")
    print(f"\n=== Link to reconstruction outcomes ({pv.fc_vis_minus_text.notna().sum()} videos; "
          f"Spearman, channel-bootstrap 95% CI) ===")
    for x in ["q1_mrr", "q2_mpnet_to_frames", "q2_frames_to_mpnet", "q3_spearman", "vis_change_mean"]:
        for y in ["fc_vis_minus_text", "rank_capcopy_minus_viscopy", "rank_llama_minus_capcopy"]:
            r, lo, hi = boot_spearman(pv, x, y)
            print(f"  {x:22s} vs {y:28s} rho {r:+.3f} [{lo:+.3f}, {hi:+.3f}]")

    # Is it caption/frame agreement, or just how busy the video is? Control for both change rates.
    ctrl = ["vis_change_mean", "cap_change_mean"]
    print(f"\n=== Same links, partial Spearman controlling for {ctrl} ===")
    for x in ["q1_mrr", "q2_mpnet_to_frames", "q2_frames_to_mpnet", "q3_spearman"]:
        for y in ["fc_vis_minus_text", "rank_capcopy_minus_viscopy", "rank_llama_minus_capcopy"]:
            r, lo, hi = boot_partial(pv, x, y, ctrl)
            print(f"  {x:22s} vs {y:28s} rho {r:+.3f} [{lo:+.3f}, {hi:+.3f}]")


if __name__ == "__main__":
    main()
