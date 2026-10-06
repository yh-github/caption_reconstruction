"""Feasibility checks for new evaluations, using cached Llama outputs (MPNet text space, SigLIP frames).

(a) video-pool rank (sanity: should match the master CSV)
(b) stratify Llama-vs-baseline by *a-priori* boundary disagreement (no ground-truth leak)
(c) within-gap discrimination: rank each gap target among the gap seconds only
"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd

sys.path.insert(0, "src")
from llm.local_embedder import LocalEmbedder

H = next(p for p in (Path.home() / ".cache/huggingface/hub/datasets--Y3--dense_video_captions/snapshots").iterdir()
         if (p / "reconstruction/wild4_llama_multi_width").exists())
SNAPS = list((Path.home() / ".cache/huggingface/hub/datasets--Y3--dense_video_captions/snapshots").iterdir())
RUNS = [("wild4", "wild4_llama_multi_width", w) for w in (1, 2, 4, 8, 12, 16)] + \
       [("wild4", "wild4_llama_w6", 6)] + [("wild5", "wild5_llama_w3_w6", w) for w in (3, 6)]
E = LocalEmbedder("all-mpnet-base-v2")
VIS = Path("local/wild_videos_embs_siglip")

def unit(x):
    x = np.asarray(x, dtype=np.float32)
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-12)

def rank_in(pool, pred, t):
    s = pool @ pred
    return int((s > s[t] + 1e-9).sum()) + 1

cap_cache, rows = {}, []
for ds, run, w in RUNS:
    files = {}
    for s in SNAPS:
        d = s / "reconstruction" / run / f"llama-3.1-8b__whole_window__t=0.6__fixed_fill(w={w}, i=29)"
        if d.exists():
            for f in d.glob("*.json"):
                if not f.name.startswith("skip__"):
                    files[f.stem] = f
    for vid, f in files.items():
        rc = {int(k): v for k, v in json.load(open(f)).get("reconstructed_captions", {}).items() if v}
        gap = sorted(rc)
        if len(gap) != w:
            continue
        cf = Path(f"datasets/wildQA/captions__{ds}/{vid}.json")
        vf = VIS / f"{vid.replace(chr(39), '_')}.npy"
        if not cf.exists() or not vf.exists():
            continue
        if vid not in cap_cache:
            caps = [c["caption"] for c in json.load(open(cf))["captions"][:60]]
            cap_cache[vid] = unit(E.get_embeddings(f"{vid}_dense_caps", caps)) if len(caps) == 60 else None
        C = cap_cache[vid]
        V = unit(np.load(vf)[:60])
        if C is None or len(V) < 60:
            continue
        L, R = gap[0] - 1, gap[-1] + 1
        if L < 0 or R > 59:
            continue
        P = unit(E.get_embeddings(f"{vid}_llama_w{w}_i29", [rc[t] for t in gap]))
        c_lerp, v_lerp = unit(C[L] + C[R]), unit(V[L] + V[R])
        for k, t in enumerate(gap):
            near = L if (t - L) <= (R - t) else R
            a = (k + 1) / (w + 1)
            pos_c, pos_v = unit((1 - a) * C[L] + a * C[R]), unit((1 - a) * V[L] + a * V[R])
            G = np.array(gap)
            def within(pool, pred):  # rank of t among gap seconds, calibrated to [-1, 1]
                if w < 2:
                    return np.nan
                s = pool[G] @ pred
                r = (s > s[k] + 1e-9).sum() + 0.5 * ((np.abs(s - s[k]) <= 1e-9).sum() - 1) + 1
                return 1 - 2 * (r - 1) / (w - 1)
            rows.append(dict(
                ds=ds, vid=vid, chan=vid.replace("Olly's", "Olly_s").rsplit("_", 1)[0], w=w, t=t,
                bd_text=1 - float(C[L] @ C[R]), bd_vis=1 - float(V[L] @ V[R]),
                r_llama=rank_in(C, P[k], t), r_crep=rank_in(C, C[near], t), r_clerp=rank_in(C, c_lerp, t),
                r_vrep=rank_in(V, V[near], t), r_vlerp=rank_in(V, v_lerp, t),
                wg_llama=within(C, P[k]), wg_crep=within(C, C[near]), wg_cpos=within(C, pos_c),
                wg_vrep=within(V, V[near]), wg_vpos=within(V, pos_v),
            ))

df = pd.DataFrame(rows)
Path("results/analysis_controls").mkdir(parents=True, exist_ok=True)
df.to_csv("results/analysis_controls/feasibility_gap_level_rows.csv", index=False)
pd.set_option("display.width", 200)
print("videos per W:", df.groupby("w").vid.nunique().to_dict())
print("\n(a) mean video-pool rank")
print(df.groupby("w")[["r_llama", "r_crep", "r_clerp", "r_vrep", "r_vlerp"]].mean().round(1))

v = df.groupby(["vid", "chan", "w"]).mean(numeric_only=True).reset_index()
v["d_crep"] = v.r_llama - v.r_crep
v["d_vrep"] = v.r_llama - v.r_vrep
for col in ["bd_text", "bd_vis"]:
    v["tercile"] = v.groupby("w")[col].transform(lambda x: pd.qcut(x, 3, labels=["low", "mid", "high"]))
    print(f"\n(b) Llama minus baseline rank by {col} tercile (all W pooled; negative = Llama better)")
    print(v.groupby("tercile", observed=True).agg(
        d_crep=("d_crep", "mean"), win_crep=("d_crep", lambda x: (x < 0).mean()),
        d_vrep=("d_vrep", "mean"), win_vrep=("d_vrep", lambda x: (x < 0).mean()),
        n=("vid", "size")).round(2))
    print(v.groupby(["w", "tercile"], observed=True)[["d_crep", "d_vrep"]].mean().unstack().round(1))

print("\n(c) within-gap calibrated score c (0 = chance, 1 = perfect ordering), W >= 2")
wg = df[df.w >= 2].groupby(["vid", "chan", "w"]).mean(numeric_only=True).reset_index()
print(wg.groupby("w")[["wg_llama", "wg_crep", "wg_cpos", "wg_vrep", "wg_vpos"]].mean().round(3))
rng = np.random.default_rng(0)
for b in ["wg_crep", "wg_cpos", "wg_vrep", "wg_vpos"]:
    d = (wg.wg_llama - wg[b]).groupby(wg.chan).agg(["sum", "count"])
    s, c = d["sum"].values, d["count"].values
    bs = [s[i].sum() / c[i].sum() for i in (rng.integers(0, len(s), len(s)) for _ in range(2000))]
    print(f"Llama - {b}: {s.sum() / c.sum():+.3f} CI [{np.percentile(bs, 2.5):+.3f}, {np.percentile(bs, 97.5):+.3f}]")
