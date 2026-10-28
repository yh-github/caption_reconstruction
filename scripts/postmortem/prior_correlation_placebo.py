"""Do the a-priori metrics predict reconstruction, or only the videos? A placebo test (research log P1).

The finding under test (`scripts/analyze_method_rank_differences.py` -> `results/method_rank_difference_correlations.csv`):
rank all 335 videos by Llama's cos_sim and by SigLIP interpolation's cos_sim, take the rank difference, and correlate
it with the a-priori metrics (rho up to 0.44, p ~ 1e-17).

Part 1 (placebo): repeat it with the text arm replaced by caption copy (Caption_RepeatClosest) or caption mean
  (Caption_MeanClosest), which reconstruct nothing. If the correlations stay, they describe the videos and the metric.
Part 2 (residual): correlate the priors with Llama's gain over caption copy, the only target that is about the model,
  with channel-cluster bootstrap CIs at a Bonferroni level over 6 priors x 3 W x 2 metrics.

Data: results/unified_benchmark_master.csv (gap start i = 29), results/apriori_full_scores.csv. No model calls.

    .venv/bin/python scripts/postmortem/prior_correlation_placebo.py
"""
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import spearmanr

R = "results/"
PRIORS = ["average_dynamism", "combined_dynamism", "APCS_V", "text_average_dynamism", "text_combined_dynamism", "APCS_T"]
LLAMA, COPY, MEAN, VIS = "Llama-3.1-8B", "Caption_RepeatClosest", "Caption_MeanClosest", "Visual_SigLIP_MeanClosest"
rng = np.random.default_rng(0)


def norm(s):
    return s.str.replace("Olly's-Farm", "Olly_s-Farm")


def load():
    m = pd.read_csv(R + "unified_benchmark_master.csv")
    m = m[m["index"] == 29].assign(video_id=lambda d: norm(d.video_id))
    a = pd.read_csv(R + "apriori_full_scores.csv").assign(video_id=lambda d: norm(d.video_id))
    ref = pd.read_csv(R + "method_rank_differences_per_video.csv").assign(video_id=lambda d: norm(d.video_id))
    return m, a, ref


def per_video(m, a, W):
    p = m[m.width == W].pivot_table(index="video_id", columns="method", values=["cos_sim", "mrr"])
    p.columns = [f"{v}|{k}" for v, k in p.columns]
    return p.reset_index().merge(a, on="video_id").dropna(subset=[f"cos_sim|{LLAMA}"])


def cluster_ci(d, x, y, level=0.95, B=2000):
    ch = d.movie_id.unique()
    g = {c: d[d.movie_id == c] for c in ch}
    rs = [spearmanr(*pd.concat([g[c] for c in rng.choice(ch, len(ch))])[[x, y]].T.to_numpy())[0] for _ in range(B)]
    a = (1 - level) / 2
    return np.percentile(rs, [100 * a, 100 * (1 - a)])


def part1(m, a, ref):
    print("=== Part 1: placebo. rho(prior, rank(text arm) - rank(SigLIP interpolation)), cos_sim, i = 29")
    for W in [3, 6, 12]:
        d = per_video(m, a, W)
        r = ref[ref.w == W][["video_id", "cos_sim_mean"]].merge(d, on="video_id")
        print(f"\n--- W={W}  videos={len(d)}  channels={d.movie_id.nunique()}  "
              f"(reproduces the original Llama scores: rho={spearmanr(r.cos_sim_mean, r[f'cos_sim|{LLAMA}'])[0]:.3f})")
        vr = d[f"cos_sim|{VIS}"].rank(ascending=False)
        for name, col in [("Llama", LLAMA), ("CapCopy", COPY), ("CapMean", MEAN)]:
            d[f"rd_{name}"] = d[f"cos_sim|{col}"].rank(ascending=False) - vr
        cols = {"rd_Llama": "Llama (orig.)", "rd_CapCopy": "caption copy", "rd_CapMean": "caption mean",
                f"cos_sim|{COPY}": "copy score", f"cos_sim|{VIS}": "SigLIP interp. score"}
        out = pd.DataFrame({lab: [spearmanr(d[p], d[c])[0] for p in PRIORS] for c, lab in cols.items()}, index=PRIORS)
        print(out.round(2).to_string())
        print(f"rho(rd_Llama, rd_CapCopy) = {spearmanr(d.rd_Llama, d.rd_CapCopy)[0]:.2f}")
        if W == 6:
            for p in ["text_average_dynamism", "average_dynamism"]:
                for t in ["rd_Llama", "rd_CapCopy"]:
                    print(f"  channel-bootstrap 95% CI rho({p}, {t}): {cluster_ci(d, p, t).round(2)}")
            z = d.rename(columns=lambda c: c.replace("|", "_").replace("-", "_").replace(".", "_"))
            for f in ["rd_Llama ~ rd_CapCopy",
                      "rd_Llama ~ rd_CapCopy + text_average_dynamism + average_dynamism + APCS_T + APCS_V",
                      "rd_Llama ~ text_average_dynamism + average_dynamism + APCS_T + APCS_V"]:
                print(f"  R2={smf.ols(f, z).fit().rsquared:.3f}  {f}")


def part2(m, a):
    print("\n=== Part 2: priors vs. Llama's gain over caption copy (channel bootstrap, Bonferroni over 36 tests)")
    alpha, n_sig, n = 0.05 / 36, 0, 0
    for W in [3, 6, 12]:
        d = per_video(m, a, W)
        for met in ["cos_sim", "mrr"]:
            d["gain"] = d[f"{met}|{LLAMA}"] - d[f"{met}|{COPY}"]
            print(f"W={W} {met}: Llama - copy mean {d.gain.mean():+.4f}; Llama ahead on {(d.gain > 0).mean():.0%} of videos")
            for p in PRIORS:
                lo, hi = cluster_ci(d, p, "gain", level=1 - alpha)
                rho, sig = spearmanr(d[p], d.gain)[0], lo > 0 or hi < 0
                n, n_sig = n + 1, n_sig + sig
                if abs(rho) > 0.12 or sig:
                    print(f"    {p:24s} rho={rho:+.2f}  CI [{lo:+.2f}, {hi:+.2f}] {'*' if sig else ''}")
    print(f"{n_sig}/{n} survive")


if __name__ == "__main__":
    m, a, ref = load()
    part1(m, a, ref)
    part2(m, a)
