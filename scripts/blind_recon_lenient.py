#!/usr/bin/env python3
"""
Lenient scoring of gap reconstructions: is "generation doesn't scale" a property of the models or of per-second rank?

Same gaps and embeddings as `blind_llm_runner.py score` (MPNet fill vs. the 60 true captions; SigLIP 2 text of the fill
vs. the 60 frames), scored four ways, from strictest to most lenient:
  rank      rank of the true second among the video's 60 (as in the paper, §4.1)
  tol1/2    the same, but seconds within ±1 / ±2 s of the target are not distractors: the target scores the best
            similarity in its window, and only seconds outside the window compete
  perm      order-free: fills are matched to the gap's seconds by the assignment that maximizes total similarity
            (Hungarian), then ranked as in `rank`. Credits right content in the wrong order. Equal to rank at W = 1
  seg_c     segment level: the mean of the fills against the mean of each length-W window of the video that does not
            overlap the gap. Calibrated AUC, c = 2·AUC − 1 (0 = chance, 1 = perfect)
Ranks: 1 = best, chance 30.5. Frame arms (frame copy, frame mean) are scored in frame space only; the true caption
("oracle") in frame space shows the headroom any caption-based fill has.

Paired differences vs. caption copy use gaps every listed model filled, per-video means, channel bootstrap.

Usage (from repo root):
    .venv/bin/python scripts/blind_recon_lenient.py sonnet opus
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from blind_llm_runner import OUT, latest, llama_fills, load_videos, out_path  # noqa: E402
from caption_vs_siglip_audit import VIS_DIR, channel, unit  # noqa: E402
from llm.local_embedder import LocalEmbedder, SiglipTextEmbedder  # noqa: E402

METRICS = ["rank", "tol1", "tol2", "perm", "seg_c"]


def per_second(sim: np.ndarray, gap: list[int], k: int) -> float:
    """sim: fills × 60. Mean over fills of 1 + #seconds outside ±k of t that beat the best second inside it."""
    out = []
    for j, t in enumerate(gap):
        near = np.abs(np.arange(sim.shape[1]) - t) <= k
        out.append(1 + int((sim[j, ~near] > sim[j, near].max()).sum()))
    return float(np.mean(out))


def scores(F: np.ndarray, T: np.ndarray, gap: list[int]) -> dict:
    """F: fills (W × d, unit rows), T: targets (60 × d, unit rows)."""
    sim = F @ T.T
    rows, cols = linear_sum_assignment(-sim[:, gap])
    perm = [gap[c] for c in cols[np.argsort(rows)]]
    w, n = len(gap), T.shape[0]
    seg = lambda s: unit(T[s:s + w].mean(0, keepdims=True))[0]  # noqa: E731
    f = unit(F.mean(0, keepdims=True))[0]
    true = f @ seg(gap[0])
    dis = np.array([f @ seg(s) for s in range(n - w + 1) if s + w - 1 < gap[0] or s > gap[-1]])
    auc = ((dis < true).sum() + 0.5 * (dis == true).sum()) / len(dis)
    return dict(rank=per_second(sim, gap, 0), tol1=per_second(sim, gap, 1), tol2=per_second(sim, gap, 2),
                perm=per_second(sim, perm, 0), seg_c=2 * auc - 1)


def main():
    models = sys.argv[1:] or ["sonnet", "opus"]
    videos, llama = load_videos(), llama_fills()
    mp, sg = LocalEmbedder("all-mpnet-base-v2"), SiglipTextEmbedder("google/siglip2-base-patch16-224")
    claude = {m: {(r["vid"], r["w"]): r for r in latest(out_path("recon", m)).values() if r["status"] == "ok"}
              for m in models}
    gaps = sorted(set.intersection(*[set(c) for c in claude.values()]) & set(llama))
    res = []
    for vid, w in gaps:
        gap = claude[models[0]][(vid, w)]["gap"]
        caps = [c.caption for c in videos[vid].clips[:60]]
        Mt = unit(mp.get_embeddings(f"{vid}_dense_caps", caps))
        St = unit(sg.get_embeddings(f"{vid}_dense_caps", caps))
        V = unit(np.load(VIS_DIR / f"{vid.replace(chr(39), '_')}.npy")[:60])
        lo, hi = gap[0] - 1, gap[-1] + 1
        near = [lo if t - lo <= hi - t else hi for t in gap]
        text = {m: ([claude[m][(vid, w)]["parsed"][str(t)] for t in gap], f"{vid}_blind_claude_{m}_w{w}")
                for m in models}
        text["llama"] = ([llama[(vid, w)][t] for t in gap], f"{vid}_blind_llama__w{w}")
        arms = {}  # arm -> (MPNet fills or None, frame-space fills)
        for m, (texts, key) in text.items():
            arms[m] = (unit(mp.get_embeddings(key, texts)), unit(sg.get_embeddings(key, texts)))
        arms["caption_copy"] = (Mt[near], St[near])
        arms["caption_mean"] = (np.repeat(unit(Mt[[lo, hi]].mean(0, keepdims=True)), w, 0),
                                np.repeat(unit(St[[lo, hi]].mean(0, keepdims=True)), w, 0))
        arms["oracle"] = (None, St[gap])
        arms["frame_copy"] = (None, V[near])
        arms["frame_mean"] = (None, np.repeat(unit(V[[lo, hi]].mean(0, keepdims=True)), w, 0))
        for arm, (Ft, Fv) in arms.items():
            for space, F, T in (("text", Ft, Mt), ("frame", Fv, V)):
                if F is not None:
                    res.append(dict(vid=vid, chan=channel(vid), w=w, arm=arm, space=space, **scores(F, T, gap)))
    df = pd.DataFrame(res)
    df.to_csv(OUT / f"recon_lenient__{'_'.join(models)}.csv", index=False)
    print(f"{len(gaps)} gaps, {df.vid.nunique()} videos, {df.chan.nunique()} channels, W = {sorted(df.w.unique())}")
    print("means (ranks: 1 = best, chance 30.5; seg_c: 0 = chance, 1 = perfect)")
    print(df.groupby(["space", "arm"])[METRICS].mean().round(2).to_string())

    rng = np.random.default_rng(0)

    def boot(diff: pd.Series) -> str:  # per-video mean, then resample channels
        v = diff.groupby(level=["vid", "chan"]).mean()
        chans = v.index.get_level_values("chan")
        by = [v.to_numpy()[chans == c] for c in chans.unique()]
        bs = [np.concatenate([by[j] for j in rng.integers(0, len(by), len(by))]).mean() for _ in range(2000)]
        lo, hi = np.percentile(bs, [2.5, 97.5])
        return f"{v.mean():+.2f} [{lo:+.2f}, {hi:+.2f}]"

    for space in ("text", "frame"):
        wide = df[df.space == space].pivot_table(index=["vid", "chan", "w"], columns="arm", values=METRICS)
        print(f"\n{space} space, arm - caption copy (ranks: negative is better; seg_c: positive is better)")
        for a in models + ["llama", "oracle", "frame_copy"]:
            if (METRICS[0], a) in wide:
                print(f"  {a:12s} " + "  ".join(f"{x} {boot(wide[(x, a)] - wide[(x, 'caption_copy')])}"
                                               for x in METRICS))
        if len(models) > 1:
            a, b = models[-1], models[0]
            print(f"  {a} - {b}: " + "  ".join(f"{x} {boot(wide[(x, a)] - wide[(x, b)])}" for x in METRICS))


if __name__ == "__main__":
    main()
