#!/usr/bin/env python3
"""
Robustness of the caption-to-frame lag found by scripts/caption_vs_siglip_audit.py (caption t best
matches frame t+1..t+2). No masking for parts 1-2; part 3 re-scores existing gap fills.

1. Lag profile by channel, dataset and caption length: mean z-scored sim(caption t, frame t+k) per lag k,
   with a channel-bootstrap CI on the peak lag and on z(+2) - z(0).
2. Drift vs constant offset: the same profile within thirds of the video (t restricted to [5, 55) so
   every lag in -5..+5 is defined for every t), and a per-video slope of the per-second best lag on t.
3. Re-alignment: score each gap fill (oracle caption, caption copy, Llama-3.1-8B) for slot t against frame
   t+s with SigLIP 2 text, s in -2..+4. Rank of frame t+s among the video's 60 frames (lower is better).
   Text-to-text (MPNet) scores are unchanged by a shift, so frames are the only place a shift can show.
4. Q3 with the lag: per-video Spearman(caption change at t, frame change at t+s), and top-10% event
   coincidence (+-1 s) with caption changes shifted by s.
5. The shared-target near pool (scripts/eval_shared_target.py, same video +-10 s) with the gap shifted by s:
   how much of the true caption's weak near-pool score (c ~ 0.18) the lead explains.

Usage (from repo root):
    .venv/bin/python scripts/caption_lag_robustness.py
Outputs: results/caption_audit/lag_{profiles,shift_ranks,q3,near_pool}.csv and a printed summary.
"""
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from caption_vs_siglip_audit import OUT, T, channel, coincidence, load_videos, unit
from llm.local_embedder import SiglipTextEmbedder
from shared_target.metrics import compute_calibrated_score, compute_mid_rank
from shared_target.pools import build_same_video_near_candidates

LAGS = list(range(-5, 6))
SHIFTS = list(range(-2, 5))
WIDTHS = [1, 2, 3, 4, 6, 8, 12, 16]
RECON = Path("results/recon/manual_download/reconstruction")
# Same i=29 sources as scripts/rebuild_unified_master.py (W=3 for wild4 from the video-pool re-score).
LLAMA_RUNS = ["wild4_llama_multi_width", "wild4_llama_w3_window_v3_videopool", "wild4_llama_w6",
              "wild5_llama_multi_width", "wild5_llama_w3_w6"]
N_BOOT = 2000
rng = np.random.default_rng(2026)


def zsim(d):
    sim = d["S"] @ d["V"].T
    return (sim - sim.mean(1, keepdims=True)) / (sim.std(1, keepdims=True) + 1e-9)


def profile(z, ts):
    return {k: np.mean([z[t, t + k] for t in ts if 0 <= t + k < T]) for k in LAGS}


def boot_chan(df, stat, n=N_BOOT):
    """stat(df) on the full data and a 95% CI from resampling channels with replacement."""
    by = {c: g for c, g in df.groupby("chan")}
    chans = list(by)
    vals = [stat(pd.concat([by[c] for c in rng.choice(chans, len(chans))])) for _ in range(n)]
    return stat(df), *np.nanpercentile(vals, [2.5, 97.5])


def peak(prof_df):
    m = prof_df[LAGS].mean()
    return int(m.idxmax())


def lead(prof_df):
    m = prof_df[LAGS].mean()
    return m[2] - m[0]


def part1_2(data):
    rows, slopes = [], []
    mid = range(5, T - 5)
    thirds = {"early": range(5, 22), "middle": range(22, 39), "late": range(39, 55)}
    for vid, d in data.items():
        z = zsim(d)
        base = dict(vid=vid, chan=channel(vid), dataset="wild4" if vid in WILD4 else "wild5",
                    cap_len=np.mean([len(c.split()) for c in d["caps"]]))
        rows.append({**base, "segment": "all", **profile(z, range(T))})
        for name, ts in thirds.items():
            rows.append({**base, "segment": name, **profile(z, ts)})
        best = [LAGS[int(np.argmax([z[t, t + k] for k in LAGS]))] for t in mid]
        slopes.append(dict(vid=vid, chan=base["chan"], slope_per_min=np.polyfit(list(mid), best, 1)[0] * 60))
    prof = pd.DataFrame(rows)
    prof["cap_len_tertile"] = pd.qcut(prof.cap_len, 3, labels=["short", "medium", "long"]).astype(str)
    prof.to_csv(OUT / "lag_profiles.csv", index=False)
    allp = prof[prof.segment == "all"]

    print("=== 1. Lag profile (mean z of sim(caption t, frame t+k); channel-bootstrap 95% CI) ===")
    print("overall:", " ".join(f"{k:+d}:{v:.2f}" for k, v in allp[LAGS].mean().items()))
    print("peak lag %d [%d, %d]; z(+2) - z(0) %.3f [%.3f, %.3f]" % (*boot_chan(allp, peak), *boot_chan(allp, lead)))
    for col in ["dataset", "cap_len_tertile"]:
        for key, g in allp.groupby(col):
            print(f"  {col}={key:7s} n={len(g):3d} peak {peak(g):+d}  z(+2)-z(0) {lead(g):+.3f}  "
                  f"z(0) {g[0].mean():.2f} z(+1) {g[1].mean():.2f} z(+2) {g[2].mean():.2f}")
    per_chan = allp.groupby("chan").apply(lambda g: pd.Series(dict(n=len(g), peak=peak(g), lead=lead(g))),
                                          include_groups=False)
    print(f"per channel ({len(per_chan)}): peak lag counts {per_chan.peak.value_counts().sort_index().to_dict()}; "
          f"z(+2) > z(0) in {(per_chan.lead > 0).sum()} channels")
    print("  channels with peak <= 0:", per_chan[per_chan.peak <= 0].round(2).to_dict("index"))

    print("\n=== 2. Drift vs constant offset (t in [5, 55)) ===")
    for seg in ["early", "middle", "late"]:
        g = prof[prof.segment == seg]
        r = boot_chan(g, lead)
        print(f"  {seg:6s} peak {peak(g):+d}  z(+2)-z(0) {r[0]:.3f} [{r[1]:.3f}, {r[2]:.3f}]  "
              + " ".join(f"{k:+d}:{v:.2f}" for k, v in g[LAGS].mean().items()))
    sl = pd.DataFrame(slopes)
    r = boot_chan(sl, lambda g: g.slope_per_min.mean())
    print(f"  per-video slope of per-second best lag: {r[0]:+.2f} s per minute [{r[1]:+.2f}, {r[2]:+.2f}]")


def gap_fills(data):
    """(vid, W) -> {slot t: {method: text}} for oracle, caption copy (nearest boundary, ties left), Llama."""
    fills = {}
    for run in LLAMA_RUNS:
        for sub in (RECON / run).glob("*fixed_fill(w=*, i=29)"):
            w = int(re.search(r"w=(\d+)", sub.name).group(1))
            if w not in WIDTHS:
                continue
            for jf in sub.glob("*.json"):
                if jf.name.startswith("skip__") or jf.name.endswith("metadata.json"):
                    continue
                vid = next((v for v in (jf.stem, jf.stem.replace("Olly_s", "Olly's")) if v in data), None)
                if vid is None:
                    continue
                rec = {int(k): v for k, v in json.load(open(jf))["reconstructed_captions"].items()}
                fills[(vid, w)] = rec
    out = {}
    for (vid, w), rec in fills.items():
        caps = data[vid]["caps"]
        gap = sorted(rec)
        lo, hi = gap[0] - 1, gap[-1] + 1
        out[(vid, w)] = {t: dict(oracle=caps[t], llama=rec[t],
                                 caption_copy=caps[lo if (t - lo <= hi - t or hi >= T) else hi]) for t in gap}
    return out


def part3(data):
    fills = gap_fills(data)
    sg = SiglipTextEmbedder("google/siglip2-base-patch16-224")
    rows = []
    for (vid, w), slots in fills.items():
        V = data[vid]["V"]
        ts = sorted(slots)
        for m in ["oracle", "caption_copy", "llama"]:
            texts = [slots[t][m] for t in ts]
            E = data[vid]["S"][ts] if m == "oracle" else unit(sg.get_embeddings(f"{vid}_lag_{m}_w{w}", texts))
            sim = E @ V.T
            for j, t in enumerate(ts):
                for s in SHIFTS:
                    if 0 <= t + s < T:
                        rank = 1 + (sim[j] > sim[j, t + s]).sum()
                        rows.append(dict(vid=vid, chan=channel(vid), W=w, t=t, method=m, shift=s, rank=rank))
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "lag_shift_ranks.csv", index=False)
    print(f"\n=== 3. Re-alignment: rank of frame t+s for the fill at slot t (of 60; lower is better) ===")
    print(f"{df.groupby('method').vid.nunique().to_dict()} videos, {len(fills)} (video, W) gaps")
    pv = df.groupby(["method", "shift", "vid", "chan"], as_index=False)["rank"].mean()
    print(pv.pivot_table(index="method", columns="shift", values="rank").round(1).to_string())
    print("paired gain, rank(s=0) - rank(s): per-video means, channel-bootstrap 95% CI")
    for m in ["oracle", "caption_copy", "llama"]:
        wide = pv[pv.method == m].pivot_table(index=["vid", "chan"], columns="shift", values="rank").reset_index()
        for s in [1, 2]:
            r = boot_chan(wide, lambda g: (g[0] - g[s]).mean())
            print(f"  {m:12s} s=+{s}: {r[0]:+.2f} [{r[1]:+.2f}, {r[2]:+.2f}]")
    by_w = df[df["shift"].isin([0, 2])].pivot_table(index=["method", "W"], columns="shift", values="rank")
    by_w["gain"] = by_w[0] - by_w[2]
    print("by width, rank(s=0) - rank(s=+2):")
    print(by_w["gain"].unstack("W").round(1).to_string())


def part5(data):
    """Shared-target near pool (same video, +-10 s, gap and boundaries excluded) with the gap shifted by s:
    the fill at slot t is scored against frame t+s, and the pool is built around the shifted gap."""
    fills = {k: v for k, v in gap_fills(data).items() if k[1] in (3, 6)}
    sg = SiglipTextEmbedder("google/siglip2-base-patch16-224")
    rows = []
    for (vid, w), slots in fills.items():
        V, ts = data[vid]["V"], sorted(slots)
        embs = {m: data[vid]["S"][ts] if m == "oracle" else
                unit(sg.get_embeddings(f"{vid}_lag_{m}_w{w}", [slots[t][m] for t in ts]))
                for m in ["oracle", "caption_copy", "llama"]}
        embs["frame_copy"] = V[[ts[0] - 1 if t - (ts[0] - 1) <= (ts[-1] + 1) - t else ts[-1] + 1 for t in ts]]
        for s in [0, 1, 2, 3]:
            gap = [t + s for t in ts]
            if gap[-1] + 1 >= T:
                continue
            pool = [c.second for c in build_same_video_near_candidates(vid, gap, window_sec=10, total_seconds=T)]
            for m, E in embs.items():
                if m == "frame_copy" and s:
                    continue
                for j, t in enumerate(ts):
                    sim = E[j] @ V.T
                    r, _ = compute_mid_rank(sim[t + s], sim[pool])
                    rows.append(dict(vid=vid, chan=channel(vid), W=w, method=m, shift=s,
                                     c=compute_calibrated_score(r, len(pool) + 1)))
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "lag_near_pool.csv", index=False)
    pv = df.groupby(["method", "shift", "vid", "chan"], as_index=False).c.mean()
    print("\n=== 5. Shared-target near pool (+-10 s), calibrated c, gap shifted by s (W = 3, 6) ===")
    print(pv.pivot_table(index="method", columns="shift", values="c").round(3).to_string())
    for m in ["oracle", "caption_copy", "llama"]:
        wide = pv[pv.method == m].pivot_table(index=["vid", "chan"], columns="shift", values="c").reset_index()
        r = boot_chan(wide, lambda g: (g[2] - g[0]).mean())
        print(f"  {m:12s} c(s=+2) - c(s=0): {r[0]:+.3f} [{r[1]:+.3f}, {r[2]:+.3f}]")


def part4():
    ps = pd.read_csv(OUT / "per_second.csv").sort_values(["vid", "t"])
    rows = []
    for s in SHIFTS:
        sh = ps.copy()
        sh["dv"] = sh.groupby("vid").dv.shift(-s)  # caption change at t vs frame change at t+s
        sh = sh.dropna()
        rho = sh.groupby("vid").apply(lambda g: spearmanr(g.dc, g.dv)[0], include_groups=False)
        rows.append(dict(shift=s, spearman_mean=rho.mean(), **coincidence(sh)))
    q3 = pd.DataFrame(rows)
    q3.to_csv(OUT / "lag_q3.csv", index=False)
    print("\n=== 4. Q3 with the lag: caption change at t vs frame change at t+s ===")
    print(q3.round(3).to_string(index=False))


WILD4 = {p.stem for p in Path("datasets/wildQA/captions__wild4").glob("*.json")}


def main():
    data = load_videos()
    print(f"{len(data)} videos, {len({channel(v) for v in data})} channels\n")
    part1_2(data)
    part3(data)
    part4()
    part5(data)


if __name__ == "__main__":
    main()
