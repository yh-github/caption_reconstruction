#!/usr/bin/env python3
"""Pilot analysis (pilot pre-registration §6): free arms, Claude calibration, continuity profiles (Prop. 3),
leak, robustness without text-tagged pieces, and power. No API calls.

    .venv/bin/python scripts/unshuffle/analyze.py [--mc-draws 500] [--skip-mc]
Reads results/unshuffle/free/scores.csv and results/unshuffle/claude/*.jsonl; prints the tables and writes
them to results/unshuffle/analysis/.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from unshuffle import core, data  # noqa: E402

AN = data.OUT / "analysis"
SETS = ["P12", "C6", "G6", "K4d1", "K4d2", "K4d3"]
FRAMES = "frames:-:siglip2"
CAP = "captions:isolated"
MAIN_ARMS = [f"{FRAMES}:edge", f"{FRAMES}:chain", f"{CAP}:siglip2_text:chain", f"{CAP}:mpnet:chain",
             f"{FRAMES}:probe", f"{CAP}:siglip2_text:probe", f"{CAP}:mpnet:probe",
             f"{FRAMES}:pairwise", f"{CAP}:siglip2_text:pairwise", f"{CAP}:mpnet:pairwise"]
SYMMETRIC = [f"{FRAMES}:chain"] + [f"captions:{cs}:{enc}:chain" for cs in data.CAPTION_SETS
                                    for enc in ("siglip2_text", "mpnet")]


def ci(per_video: pd.Series) -> str:
    est, lo, hi = core.cluster_boot(per_video.dropna().to_dict())
    return f"{est:+.3f} [{lo:+.3f}, {hi:+.3f}]"


def per_video(df: pd.DataFrame, metric: str) -> pd.Series:
    return df.groupby("video")[metric].mean()


def chance_row(set_name: str) -> dict:
    n = len(core.SETS[set_name][0])
    return core.chance(n)


# ----------------------------------------------------------------------------- free arms
def free_tables(sc: pd.DataFrame, variant: str) -> pd.DataFrame:
    d = sc[sc.variant == variant]
    rows = []
    for arm in MAIN_ARMS:
        for s in SETS:
            x = d[(d.arm == arm) & (d.set == s)]
            if x.empty:
                continue
            rows.append(dict(arm=arm, set=s, videos=x.video.nunique(),
                             tau=ci(per_video(x, "tau")), adj_und=ci(per_video(x, "adj_und")),
                             local_dir=ci(per_video(x, "local_dir")), exact=round(x.exact.mean(), 3)))
    for s in SETS:
        c = chance_row(s)
        rows.append(dict(arm="chance", set=s, videos=0, tau=f"{c['tau']:+.3f}", adj_und=f"{c['adj_und']:+.3f}",
                         local_dir="+0.500", exact=round(c["exact"], 3)))
    return pd.DataFrame(rows)


def symmetric_gate(sc: pd.DataFrame) -> pd.DataFrame:
    """Prop. 2: symmetric arms have local direction 0.5 in expectation (pooled over puzzle sets)."""
    d = sc[(sc.variant == "all") & sc.arm.isin(SYMMETRIC)]
    rows = []
    for arm, x in d.groupby("arm"):
        est, lo, hi = core.cluster_boot(per_video(x, "local_dir").to_dict())
        rows.append(dict(arm=arm, local_dir=f"{est:.3f} [{lo:.3f}, {hi:.3f}]", passes=lo <= 0.5 <= hi))
    return pd.DataFrame(rows)


def paired(sc: pd.DataFrame, a: str, b: str, metric: str, s: str, variant: str = "all") -> str:
    d = sc[(sc.variant == variant) & (sc.set == s)]
    w = d[d.arm.isin([a, b])].pivot_table(index=["video", "puzzle"], columns="arm", values=metric)
    if a not in w or b not in w:
        return "n/a"
    diff = (w[a] - w[b]).groupby(level="video").mean()
    return ci(diff)


def structure_contrasts(sc: pd.DataFrame, variant: str) -> pd.DataFrame:
    """H1 preview: frames chaining minus caption chaining, undirected adjacency, per puzzle set."""
    rows = []
    for enc in ("siglip2_text", "mpnet"):
        for s in SETS:
            rows.append(dict(contrast=f"frames chain - captions chain ({enc})", set=s,
                             adj_und=paired(sc, f"{FRAMES}:chain", f"{CAP}:{enc}:chain", "adj_und", s, variant)))
    for s in SETS:
        rows.append(dict(contrast="frames edge - frames chain", set=s,
                         adj_und=paired(sc, f"{FRAMES}:edge", f"{FRAMES}:chain", "adj_und", s, variant)))
    return pd.DataFrame(rows)


def leak_table(sc: pd.DataFrame) -> pd.DataFrame:
    d = sc[sc.variant == "all"]
    rows = []
    for enc in ("siglip2_text", "mpnet"):
        for method, metric in (("chain", "adj_und"), ("probe", "tau"), ("pairwise", "tau")):
            for s in ("P12", "G6"):
                row = dict(encoder=enc, method=method, metric=metric, set=s)
                for cs in data.CAPTION_SETS:
                    row[cs] = ci(per_video(d[(d.arm == f"captions:{cs}:{enc}:{method}") & (d.set == s)], metric))
                row["in_context - isolated"] = paired(d, f"captions:in_context:{enc}:{method}",
                                                      f"captions:isolated:{enc}:{method}", metric, s)
                row["in_context - nocue"] = paired(d, f"captions:in_context:{enc}:{method}",
                                                   f"captions:in_context_nocue:{enc}:{method}", metric, s)
                rows.append(row)
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- Prop. 3
def profile(E: np.ndarray) -> dict:
    """Continuity profile of one video and representation (Prop. 3).
    rho[d]  mean similarity of pieces d apart (d = 1..11; rho[0] unused)
    sigma   pooled residual SD around rho (the pre-registered i.i.d. model)
    u, sigma_e  the pre-registered fallback: residual = u_i + u_j + e (a per-piece effect, e.g. a generic
            piece that resembles every other one), fitted by least squares; sigma_e is the SD of e."""
    S = E @ E.T
    n = len(E)
    rho = np.full(n, np.nan)
    for d in range(1, n):
        rho[d] = np.mean([S[i, i + d] for i in range(n - d)])
    iu = np.triu_indices(n, 1)
    R = S[iu] - rho[iu[1] - iu[0]]
    sigma = float(np.sqrt(np.sum(R ** 2) / (len(R) - (n - 1))))
    A = np.zeros((len(R), n))
    A[np.arange(len(R)), iu[0]] = 1
    A[np.arange(len(R)), iu[1]] = 1
    u = np.linalg.lstsq(A, R, rcond=None)[0]
    e = R - A @ u
    sigma_e = float(np.sqrt(np.sum(e ** 2) / (len(R) - (n - 1) - n)))
    return dict(rho=rho, sigma=sigma, u=u, sigma_e=sigma_e)


def mc_adjacency(rho: np.ndarray, sigma: float, pieces: list[int], draws: int, rng,
                 u: np.ndarray | None = None) -> float:
    k = len(pieces)
    D = np.abs(np.subtract.outer(pieces, pieces))
    base = np.where(D > 0, rho[np.clip(D, 1, None)], 0.0)
    if u is not None:
        up = u[pieces]
        base = base + up[:, None] + up[None, :]
    eps = rng.normal(0, sigma, size=(draws, k, k))
    eps = np.triu(eps, 1)
    W = base[None] + eps + eps.transpose(0, 2, 1)
    paths = core.best_path_batch(W)
    order = np.asarray(pieces)[paths]
    return float(np.mean([core.score(list(o))["adj_und"] for o in order]))


def continuity(sc: pd.DataFrame, videos: list[str], draws: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    reps = {"frames (SigLIP 2)": (f"{FRAMES}:chain", lambda v: data.piece_means(v)),
            "captions (SigLIP 2 text)": (f"{CAP}:siglip2_text:chain",
                                         lambda v: data.caption_embeddings(v, "isolated", "siglip2_text")),
            "captions (MPNet)": (f"{CAP}:mpnet:chain", lambda v: data.caption_embeddings(v, "isolated", "mpnet"))}
    obs = sc[sc.variant == "all"].set_index(["arm", "puzzle"]).adj_und
    prof_rows, pred_rows = [], []
    rng = np.random.default_rng(0)
    puzzles = core.make_puzzles(videos)
    for name, (arm, emb) in reps.items():
        for v in videos:
            pr = profile(emb(v))
            rho, sigma = pr["rho"], pr["sigma"]
            kap = {d: (rho[d] - rho[2 * d]) / sigma for d in (1, 2, 3)}
            kap_e = {d: (rho[d] - rho[2 * d]) / pr["sigma_e"] for d in (1, 2, 3)}
            prof_rows.append(dict(rep=name, video=v, sigma=sigma, sigma_e=pr["sigma_e"], sd_u=float(pr["u"].std()),
                                  **{f"rho_{5 * d}s": rho[d] for d in range(1, 12)},
                                  **{f"kappa_{5 * d}s": kap[d] for d in kap},
                                  **{f"kappa_e_{5 * d}s": kap_e[d] for d in kap_e}))
            for pz in (p for p in puzzles if p["video"] == v):
                pred_rows.append(dict(rep=name, video=v, set=pz["set"], puzzle=pz["puzzle"],
                                      iid=mc_adjacency(rho, sigma, pz["pieces"], draws, rng),
                                      piece_effect=mc_adjacency(rho, pr["sigma_e"], pz["pieces"], draws, rng,
                                                                u=pr["u"]),
                                      observed=obs.get((arm, pz["puzzle"]), np.nan)))
    return pd.DataFrame(prof_rows), pd.DataFrame(pred_rows)


def calibration_table(pred: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (rep, s), x in pred.groupby(["rep", "set"]):
        pv = x.groupby("video")[["iid", "piece_effect", "observed"]].mean()
        row = dict(rep=rep, set=s, observed=round(pv.observed.mean(), 3))
        for model in ("iid", "piece_effect"):
            row[f"{model}: predicted"] = round(pv[model].mean(), 3)
            row[f"{model}: pred - obs"] = ci(pv[model] - pv.observed)
            row[f"{model}: r"] = round(pv[model].corr(pv.observed), 2)
        rows.append(row)
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- Claude
def claude_rows() -> pd.DataFrame:
    rows = []
    for f in sorted((data.OUT / "claude").glob("*.jsonl")):
        if f.name.endswith("__events.jsonl"):
            continue
        latest = {}
        for line in f.read_text().splitlines():
            r = json.loads(line)
            latest[r["item"]] = r
        rows += latest.values()
    return pd.DataFrame(rows)


def claude_tables(cl: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cl = cl.copy()
    cl["valid"] = cl.status == "ok"
    cl["in_tok"] = [(u or {}).get("input_tokens") for u in cl.usage]
    cl["out_tok"] = [(u or {}).get("output_tokens") for u in cl.usage]
    summ = []
    for (m, mod, s), x in cl.groupby(["model", "modality", "set"]):
        summ.append(dict(model=m, modality=mod, set=s, items=len(x), videos=x.video.nunique(),
                         valid=round(x.valid.mean(), 3), tau=ci(per_video(x, "m_tau")),
                         adj_und=ci(per_video(x, "m_adj_und")), local_dir=ci(per_video(x, "m_local_dir")),
                         exact=round(x.m_exact.mean(), 3), position_bias=round(x.position_bias.mean(), 3),
                         in_tok=int(x.in_tok.median()), out_tok=int(x.out_tok.median()),
                         secs=round(x.seconds.median(), 1)))
    pair = []
    for (m, s), x in cl.groupby(["model", "set"]):
        w = x.pivot_table(index=["video", "puzzle", "shuffle"], columns="modality",
                          values=["m_tau", "m_adj_und", "m_local_dir"])
        row = dict(model=m, set=s)
        for k in ("m_tau", "m_adj_und", "m_local_dir"):
            if (k, "frames") in w and (k, "captions") in w:
                row[f"frames - captions {k[2:]}"] = ci((w[(k, "frames")] - w[(k, "captions")]).groupby(level="video").mean())
        pair.append(row)
    quota = []
    for m, x in cl.sort_values("time").groupby("model"):
        u5 = x.util_5h.dropna()
        u7 = x.util_7d.dropna()
        quota.append(dict(model=m, calls=len(x), first_5h=u5.iloc[0] if len(u5) else None,
                          last_5h=u5.iloc[-1] if len(u5) else None,
                          first_7d=u7.iloc[0] if len(u7) else None, last_7d=u7.iloc[-1] if len(u7) else None,
                          in_tok_total=int(x.in_tok.sum()), out_tok_total=int(x.out_tok.sum())))
    return pd.DataFrame(summ), pd.DataFrame(pair), pd.DataFrame(quota)


# ----------------------------------------------------------------------------- power
def power(sc: pd.DataFrame, cl: pd.DataFrame) -> pd.DataFrame:
    """Projected 95% CI half-width with 305 videos in 40 channels, from per-video SDs of paired contrasts.
    The pilot has one video per channel, so the intraclass correlation is assumed (0 and 0.2)."""
    contrasts = {}
    d = sc[sc.variant == "all"]
    for s in ("P12", "G6", "K4d3"):
        w = d[d.set == s].pivot_table(index="video", columns="arm", values="adj_und")
        contrasts[f"frames chain - captions chain (SigLIP 2 text), adj_und, {s}"] = \
            w[f"{FRAMES}:chain"] - w[f"{CAP}:siglip2_text:chain"]
    if not cl.empty:
        for (m, s), x in cl.groupby(["model", "set"]):
            w = x.pivot_table(index=["video", "puzzle", "shuffle"], columns="modality", values="m_tau")
            if {"frames", "captions"} <= set(w):
                contrasts[f"{m} frames - captions, tau, {s}"] = (w.frames - w.captions).groupby(level="video").mean()
    rows = []
    m = 305 / 40
    for name, diff in contrasts.items():
        sd = diff.std(ddof=1)
        for icc in (0.0, 0.2):
            rows.append(dict(contrast=name, pilot_mean=round(diff.mean(), 3), per_video_sd=round(sd, 3), icc=icc,
                             half_width_305=round(1.96 * sd / np.sqrt(305) * np.sqrt(1 + (m - 1) * icc), 3)))
    return pd.DataFrame(rows)


def show(title: str, df: pd.DataFrame, name: str):
    AN.mkdir(parents=True, exist_ok=True)
    df.to_csv(AN / f"{name}.csv", index=False)
    print(f"\n## {title}\n")
    print(df.to_string(index=False))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mc-draws", type=int, default=500)
    ap.add_argument("--skip-mc", action="store_true")
    args = ap.parse_args()
    pd.set_option("display.width", 250, "display.max_colwidth", 60)
    sc = pd.read_csv(data.OUT / "free/scores.csv")
    videos = sorted(sc.video.unique())
    show("Free arms, all pieces (per-video means, 95% channel-cluster CI)", free_tables(sc, "all"), "free_all")
    show("Symmetric gate (Prop. 2): local direction must include 0.5", symmetric_gate(sc), "symmetric_gate")
    show("Structure: paired contrasts, undirected adjacency", structure_contrasts(sc, "all"), "structure_all")
    show("Structure without text-tagged pieces", structure_contrasts(sc, "notext"), "structure_notext")
    show("Leak sets (no-reader and learned arms)", leak_table(sc), "leak")
    cl = claude_rows()
    if not cl.empty:
        summ, pair, quota = claude_tables(cl)
        show("Claude readers (calibration)", summ, "claude_summary")
        show("Claude readers: frames minus captions, paired by puzzle and shuffle", pair, "claude_paired")
        show("Claude quota and tokens (account-wide quota readings)", quota, "claude_quota")
    show("Power: projected CI half-width with 305 videos", power(sc, cl), "power")
    if not args.skip_mc:
        prof, pred = continuity(sc, videos, args.mc_draws)
        prof.to_csv(AN / "continuity_profiles.csv", index=False)
        pred.to_csv(AN / "continuity_predictions.csv", index=False)
        cols = (["sigma", "sigma_e", "sd_u"] + [f"rho_{s}s" for s in (5, 10, 15, 20, 30, 55)]
                + ["kappa_5s", "kappa_10s", "kappa_15s", "kappa_e_5s", "kappa_e_10s", "kappa_e_15s"])
        show("Continuity profiles (Prop. 3), mean over videos", prof.groupby("rep")[cols].mean().round(3).reset_index(),
             "continuity_profile_means")
        show("Prop. 3 calibration: Monte Carlo predicted vs observed undirected adjacency of chaining",
             calibration_table(pred), "continuity_calibration")


if __name__ == "__main__":
    main()
