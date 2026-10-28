"""Do caption changes track frame changes within a video? Re-test of the 2026-10-07 audit (research log P1).

The audit (`scripts/caption_vs_siglip_audit.py`, handover 2026-10-07 Q3) found per-video Spearman about 0 between
the caption change curve and the frame change curve, on 1 s captions. This script asks whether that holds at 5 s.

Change curve: cosine distance between consecutive windows (captions: MPNet or SigLIP 2 text; frames: mean SigLIP 2).
  spearman  per-video Spearman(caption change, frame change), the audit's statistic
  rsa       per-video Spearman between all-pairs distance matrices (structure beyond neighbours)
Controls: frame split-half (alternate frames of each window) gives the ceiling on any agreement with the frame curve;
  caption-caption gives agreement between two caption sets of the same windows.
CIs: bootstrap over channels (wild) or videos (ego).

Subcommands (run from the repo root; no model calls):
  wild     30-video piece-caption pilot, 12 x 5 s pieces: Gemini isolated, in-context, old 1 s captions joined per piece
  windows  same 30 videos, old 1 s captions at 1, 2, 3, 5 s windows, with and without the +2 s caption lead
  nocut    wild, isolated captions, after dropping each video's largest frame changes (shot cuts)
  ego      EgoSchema ego_dev, 36 x 5 s windows: Sonnet and Haiku captions of the same 5 frames. Frames are embedded
           here with the SigLIP 2 image tower (timm v2_webli) and cached in local/egoschema/siglip2_windows/
           (about 35 min on 8 CPU cores the first time).
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
EGO_CACHE = ROOT / "local/egoschema/siglip2_windows"
rng = np.random.default_rng(0)


def unit(x):
    x = np.asarray(x, dtype=np.float32)
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-12)


def changes(X):
    return 1 - np.sum(X[1:] * X[:-1], axis=1)


def rsa(A, B):
    iu = np.triu_indices(len(A), 1)
    return spearmanr((1 - A @ A.T)[iu], (1 - B @ B.T)[iu])[0]


def boot(vals, groups, n=4000):
    vals, groups = np.asarray(vals, float), np.asarray(groups)
    ok = ~np.isnan(vals)
    vals, groups = vals[ok], groups[ok]
    ug = np.unique(groups)
    by = {g: vals[groups == g] for g in ug}
    bs = [np.concatenate([by[g] for g in rng.choice(ug, len(ug))]).mean() for _ in range(n)]
    return vals.mean(), *np.percentile(bs, [2.5, 97.5])


def report(name, rows, groups):
    print(f"\n=== {name}  (videos={len(rows)}, clusters={len(set(groups))})")
    for k in rows[0]:
        m, lo, hi = boot([r[k] for r in rows], groups)
        print(f"  {k:48s} {m:+.3f}  [{lo:+.3f}, {hi:+.3f}]")


def wild():
    from unshuffle import data as U
    rows, groups = [], []
    for v in U.pilot_videos(30):
        caps = U.captions(v)
        if not all(s in caps and len(caps[s]) == 12 for s in ("isolated", "in_context", "original")):
            continue
        F = U.frame_rows(v)
        V = U.piece_means(v)
        Va = unit(np.stack([F[p * 5:(p + 1) * 5][0::2].mean(0) for p in range(12)]))
        Vb = unit(np.stack([F[p * 5:(p + 1) * 5][1::2].mean(0) for p in range(12)]))
        dv = changes(V)
        r = {"frames split-half: dv_a vs dv_b": spearmanr(changes(Va), changes(Vb))[0]}
        C = {(s, e): U.caption_embeddings(v, s, e) for s in ("isolated", "in_context", "original")
             for e in ("mpnet", "siglip2_text")}
        for (s, e), X in C.items():
            r[f"{s}/{e}: spearman(dc, dv)"] = spearmanr(changes(X), dv)[0]
            r[f"{s}/{e}: rsa(captions, frames)"] = rsa(X, V)
        for a, b in [("isolated", "in_context"), ("isolated", "original")]:
            r[f"caption-caption mpnet: dc {a} vs {b}"] = spearmanr(changes(C[(a, "mpnet")]), changes(C[(b, "mpnet")]))[0]
        rows.append(r)
        groups.append(v.split("_")[0])
    report("WILD pilot, 12 x 5 s pieces, Gemini captions", rows, groups)


def windows():
    from llm.local_embedder import LocalEmbedder
    from piece_caption_check import CAP_DIRS
    from unshuffle import data as U
    mp = LocalEmbedder("all-mpnet-base-v2")
    res = {}
    for v in U.pilot_videos(30):
        cf = next(d / f"{v}.json" for d in CAP_DIRS if (d / f"{v}.json").exists())
        caps = [c["caption"] for c in json.loads(cf.read_text())["captions"]][:60]
        M, V = unit(mp.get_embeddings(f"{v}_dense_caps", caps)), U.frame_rows(v)
        for lag in (0, 2):
            Mx, Vx = (M[:60 - lag], V[lag:60]) if lag else (M, V)  # caption t vs. frame t + lag
            for w in (1, 2, 3, 5):
                n = len(Mx) // w
                Mw = unit(Mx[:n * w].reshape(n, w, -1).mean(1))
                Vw = unit(Vx[:n * w].reshape(n, w, -1).mean(1))
                res.setdefault((lag, w), []).append(spearmanr(changes(Mw), changes(Vw))[0])
    print("\n=== Old 1 s captions (MPNet) on the 30 pilot videos, by window size")
    for (lag, w), r in sorted(res.items()):
        print(f"  lag +{lag} s, window {w} s: mean per-video spearman(dc, dv) = {np.nanmean(r):+.3f}  (n={len(r)})")


def nocut():
    from unshuffle import data as U
    out = {k: [] for k in ("all 11 boundaries", "drop top-1 dv", "drop top-3 dv", "drop dv > 0.15")}
    big = []
    for v in U.pilot_videos(30):
        dv = changes(U.piece_means(v))
        dc = changes(U.caption_embeddings(v, "isolated", "mpnet"))
        o = np.argsort(dv)[::-1]
        out["all 11 boundaries"].append(spearmanr(dc, dv)[0])
        for k, keep in (("drop top-1 dv", o[1:]), ("drop top-3 dv", o[3:]), ("drop dv > 0.15", np.where(dv <= 0.15)[0])):
            out[k].append(spearmanr(dc[keep], dv[keep])[0] if len(keep) >= 5 else np.nan)
        big.append(np.mean(dv > 0.15))
    print("\n=== Wild pilot, isolated captions, without the largest frame changes")
    for k, r in out.items():
        print(f"  {k:18s} mean rho {np.nanmean(r):+.3f}")
    print(f"  share of piece boundaries with dv > 0.15: {np.mean(big):.2f}")


def ego_frames(vid, cap_files):
    cache = EGO_CACHE / f"{vid}.npy"
    if cache.exists():
        return np.load(cache)
    import timm
    import torch
    from PIL import Image
    global _M, _TF
    if "_M" not in globals():
        _M = timm.create_model("vit_base_patch16_siglip_224.v2_webli", pretrained=True, num_classes=0).eval()
        _TF = timm.data.create_transform(**timm.data.resolve_model_data_config(_M), is_training=False)
    out = []
    for f in cap_files:
        x = torch.stack([_TF(Image.open(ROOT / p).convert("RGB")) for p in json.loads(f.read_text())["frames"]])
        with torch.inference_mode():
            out.append(_M(x).numpy())
    arr = np.stack(out)  # windows x 5 frames x 768
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache, arr)
    return arr


def ego():
    from llm.local_embedder import LocalEmbedder, SiglipTextEmbedder
    emb = {"mpnet": LocalEmbedder("all-mpnet-base-v2"),
           "siglip2_text": SiglipTextEmbedder("google/siglip2-base-patch16-224")}
    vids = sorted({it["video_id"] for it in json.loads((ROOT / "results/index_qa/ego_dev_items.json").read_text())})
    base = ROOT / "results/index_qa/captions"
    sets = {"sonnet": "claude-sonnet-5-5-frames", "haiku": "claude-haiku-5-5-self"}
    rows, groups = [], []
    for v in vids:
        files = {k: sorted((base / d / v).glob("[0-9][0-9].json")) for k, d in sets.items()}
        n = len(files["sonnet"])
        if n < 10 or len(files["haiku"]) != n:
            print("skip", v, {k: len(f) for k, f in files.items()})
            continue
        # Both captioners saw the same 5 frames per window (the "frames" field of each caption file).
        F = unit(ego_frames(v, files["sonnet"]))
        V = unit(F.mean(1))
        Va, Vb = unit(F[:, 0::2].mean(1)), unit(F[:, 1::2].mean(1))
        dv = changes(V)
        r = {"frames split-half: dv_a vs dv_b": spearmanr(changes(Va), changes(Vb))[0]}
        C = {}
        for k, fl in files.items():
            texts = [json.loads(f.read_text())["caption"] for f in fl]
            for e, E in emb.items():
                C[(k, e)] = unit(E.get_embeddings(f"ego_{v}_{k}", texts))
        for (k, e), X in C.items():
            r[f"{k}/{e}: spearman(dc, dv)"] = spearmanr(changes(X), dv)[0]
            r[f"{k}/{e}: rsa(captions, frames)"] = rsa(X, V)
        r["caption-caption mpnet: dc sonnet vs haiku"] = spearmanr(changes(C[("sonnet", "mpnet")]),
                                                                    changes(C[("haiku", "mpnet")]))[0]
        r["caption-caption mpnet: rsa sonnet vs haiku"] = rsa(C[("sonnet", "mpnet")], C[("haiku", "mpnet")])
        r["frames: rsa split-half"] = rsa(Va, Vb)
        rows.append(r)
        groups.append(v)
    report("EGOSCHEMA dev, 36 x 5 s windows, Claude captions of the same 5 frames", rows, groups)


if __name__ == "__main__":
    cmds = {"wild": wild, "windows": windows, "nocut": nocut, "ego": ego}
    if len(sys.argv) != 2 or sys.argv[1] not in cmds:
        sys.exit(f"usage: change_tracking.py {{{','.join(cmds)}}}")
    cmds[sys.argv[1]]()
