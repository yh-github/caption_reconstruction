"""The prior-metric checks of research log P1, repeated on Claude's reconstructions (Sonnet 50 videos, Opus 32).

Gain = caption copy's rank - Claude's rank (rank of the true second among the video's 60; positive = Claude better),
per gap, in text space (MPNet) and frame space (SigLIP 2). Lenient scores (order-free rank `perm`, segment `seg_c`)
come from results/blind_llm/recon_lenient__sonnet_opus.csv (32 videos).

  1. priors vs. Claude's score, vs. copy's score, and vs. the gain (per-video means over W; channel bootstrap)
  2. do Claude's wins over copy repeat within a video across W = 1, 4, 8, 16, and between Sonnet and Opus?

No model calls.  .venv/bin/python scripts/postmortem/claude_recon_priors.py
"""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

PRIORS = ["average_dynamism", "APCS_V", "text_average_dynamism", "APCS_T"]
rng = np.random.default_rng(0)


def clus_ci(d, x, y, B=2000):
    ch = d.chan.unique()
    g = {c: d[d.chan == c] for c in ch}
    rs = [spearmanr(*pd.concat([g[c] for c in rng.choice(ch, len(ch))])[[x, y]].T.to_numpy())[0] for _ in range(B)]
    return np.nanpercentile(rs, [2.5, 97.5])


def priors():
    a = pd.read_csv("results/apriori_full_scores.csv")
    a["vid"] = a.video_id.str.replace("Olly's-Farm", "Olly_s-Farm")
    return a[["vid"] + PRIORS]


def strict(model):
    d = pd.read_csv(f"results/blind_llm/recon__{model}__scores.csv")
    d["vid"] = d.vid.str.replace("Olly's-Farm", "Olly_s-Farm")
    w = d.pivot_table(index=["vid", "chan", "w"], columns="method", values=["text_rank", "frame_rank"]).dropna()
    out = pd.DataFrame(index=w.index)
    for sp in ["text_rank", "frame_rank"]:
        s = sp.split("_")[0]
        out[f"{s}: claude"] = -w[(sp, "claude")]          # higher = better
        out[f"{s}: copy"] = -w[(sp, "caption_copy")]
        out[f"{s}: gain"] = w[(sp, "caption_copy")] - w[(sp, "claude")]
    return out.reset_index()


def lenient(model):
    d = pd.read_csv("results/blind_llm/recon_lenient__sonnet_opus.csv")
    d["vid"] = d.vid.str.replace("Olly's-Farm", "Olly_s-Farm")
    w = d.pivot_table(index=["vid", "chan", "w"], columns=["space", "arm"], values=["perm", "seg_c"])
    out = pd.DataFrame(index=w.index)
    for sp in ["text", "frame"]:
        out[f"{sp} order-free: gain"] = w[("perm", sp, "caption_copy")] - w[("perm", sp, model)]
        out[f"{sp} segment c: gain"] = w[("seg_c", sp, model)] - w[("seg_c", sp, "caption_copy")]
    return out.dropna().reset_index()


def part1(name, g, a):
    v = g.drop(columns="w").groupby(["vid", "chan"]).mean().reset_index().merge(a, on="vid")
    print(f"\n=== {name}: priors vs. per-video mean score ({len(v)} videos, {v.chan.nunique()} channels)")
    cols = [c for c in g.columns if ":" in c]
    rows = []
    for c in cols:
        for p in PRIORS:
            r = spearmanr(v[p], v[c])[0]
            lo, hi = clus_ci(v, p, c)
            rows.append((c, p, r, lo, hi))
    t = pd.DataFrame(rows, columns=["score", "prior", "rho", "lo", "hi"])
    t["sig"] = np.where((t.lo > 0) | (t.hi < 0), "*", "")
    print(t.round(2).to_string(index=False))


def part2(name, g):
    print(f"\n=== {name}: does a video's gain over copy repeat across W? (Spearman between widths, per space)")
    for c in [c for c in g.columns if c.endswith("gain")]:
        w = g.pivot_table(index="vid", columns="w", values=c)
        rs = [spearmanr(w[a], w[b], nan_policy="omit")[0] for i, a in enumerate(w.columns) for b in w.columns[i + 1:]]
        win = (w > 0).mean().round(2).to_dict()
        print(f"  {c:24s} mean rho between widths {np.mean(rs):+.2f} (range {min(rs):+.2f} to {max(rs):+.2f}); "
              f"share of gaps won by W {win}")


def main():
    a = priors()
    S, O = strict("sonnet"), strict("opus")
    for name, g in [("Sonnet, strict", S), ("Opus, strict", O), ("Opus, lenient", lenient("opus")),
                    ("Sonnet, lenient", lenient("sonnet"))]:
        part1(name, g, a)
        part2(name, g)
    j = S.merge(O, on=["vid", "chan", "w"], suffixes=(" S", " O"))
    print(f"\n=== Same gap, Sonnet vs. Opus ({len(j)} gaps)")
    for s in ["text", "frame"]:
        print(f"  {s}: rho(Sonnet gain, Opus gain) = {spearmanr(j[f'{s}: gain S'], j[f'{s}: gain O'])[0]:+.2f};"
              f" rho(Sonnet gain, -copy score) = {spearmanr(j[f'{s}: gain S'], -j[f'{s}: copy S'])[0]:+.2f};"
              f" rho(Opus gain, -copy score) = {spearmanr(j[f'{s}: gain O'], -j[f'{s}: copy O'])[0]:+.2f}")


if __name__ == "__main__":
    main()
