"""Intrinsic (video-pool rank) metrics restricted to WildQA evidence gaps, plus set-level (best-of-gap) rank."""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, "src")
from llm.local_embedder import LocalEmbedder
E = LocalEmbedder("all-mpnet-base-v2")
unit = lambda x: (lambda a: a / (np.linalg.norm(a, axis=-1, keepdims=True) + 1e-12))(np.asarray(x, dtype=np.float32))
SN = Path.home() / ".cache/huggingface/hub/datasets--Y3--dense_video_captions/snapshots"
srcs = [("wild4", d) for s in SN.iterdir() for d in (s / "reconstruction/wild4_llama_evidence").glob("*")] + \
       [("wild5", d) for d in Path("results/recon/manual_download/reconstruction/wild5_llama_evidence").glob("*")]
seen, rows = set(), []
for ds, d in srcs:
    for f in d.glob("*.json"):
        if f.name.startswith("skip__") or f.stem in seen: continue
        rc = {int(k): v for k, v in json.load(open(f)).get("reconstructed_captions", {}).items() if v}
        cf = Path(f"datasets/wildQA/captions__{ds}/{f.stem}.json"); vf = Path(f"local/wild_videos_embs_siglip/{f.stem.replace(chr(39), '_')}.npy")
        if not rc or not cf.exists() or not vf.exists(): continue
        seen.add(f.stem)
        caps = [c["caption"] for c in json.load(open(cf))["captions"][:60]]
        if len(caps) != 60: continue
        C = unit(E.get_embeddings(f"{f.stem}_dense_caps", caps)); V = unit(np.load(vf)[:60])
        gap = sorted(t for t in rc if t < 60); L, R = gap[0] - 1, gap[-1] + 1
        P = unit(E.get_embeddings(f"{f.stem}_llama_evidence", [rc[t] for t in gap]))
        def rk(pool, pred, t):
            s = pool @ pred; return int((s > s[t] + 1e-9).sum()) + 1
        for k, t in enumerate(gap):
            nb = [b for b in (L, R) if 0 <= b < 60]
            near = min(nb, key=lambda b: abs(b - t))
            rows.append(dict(vid=f.stem, chan=f.stem.replace("Olly's", "Olly_s").rsplit("_", 1)[0], w=len(gap),
                             r_llama=rk(C, P[k], t), r_crep=rk(C, C[near], t), r_vrep=rk(V, V[near], t)))
df = pd.DataFrame(rows)
v = df.groupby(["vid", "chan"]).mean(numeric_only=True).reset_index()
print(len(v), "videos; gap width median", v.w.median())
print(v[["r_llama", "r_crep", "r_vrep"]].mean().round(2).to_dict())
d = v.r_llama - v.r_crep
print(f"Llama - caption copy: {d.mean():+.2f}, Llama wins {(d<0).mean():.2f}")
d = v.r_llama - v.r_vrep
print(f"Llama - visual copy: {d.mean():+.2f}, Llama wins {(d<0).mean():.2f}")
