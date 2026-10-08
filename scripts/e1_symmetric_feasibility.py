#!/usr/bin/env python3
"""
Feasibility of the symmetric E1 design for the non-LLM arms.

For each WildQA question, mask the evidence span plus K-1 decoy spans of the same length
(non-overlapping, >= 2 s apart). Every arm fills all K spans; the question must pick the evidence
span among the K filled spans (score of a span = max similarity over its seconds). All candidates
come from the same arm, video and style, so per-video offsets and "distinctiveness" cancel.
Chance MRR for K=3 is 0.611.

Llama needs a joint evidence+decoy masking run (GPU); this script scores oracle, repeat and lerp
for captions (MPNet, SigLIP-text) and frames (SigLIP).

Usage (from repo root):
    .venv/bin/python scripts/e1_symmetric_feasibility.py [K]
"""
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from data.wildqa_loader import load_wildqa_dataset
from llm.local_embedder import LocalEmbedder, SiglipTextEmbedder

K = int(sys.argv[1]) if len(sys.argv) > 1 else 3
VIS_DIR = Path("local/wild_videos_embs_siglip")


def unit(x):
    x = np.asarray(x, dtype=np.float32)
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-12)


def pick_decoys(tg, k, rng, T=60, gap=2, tries=500):
    n, spans = len(tg), [tg]
    for _ in range(tries):
        if len(spans) == k:
            return spans
        s = int(rng.integers(1, T - n))  # keep a left boundary; right boundary may be the video end
        cand = list(range(s, s + n))
        if all(min(abs(a - b) for a in cand for b in sp) >= gap for sp in spans):
            spans.append(cand)
    return None


def fills(index, spans):
    """Fill every masked span (all spans masked jointly) with repeat / lerp from the nearest unmasked boundaries."""
    masked = {t for sp in spans for t in sp}
    rep, lerp = index.copy(), index.copy()
    for sp in spans:
        L, R = sp[0] - 1, sp[-1] + 1
        while L in masked:
            L -= 1
        while R in masked:
            R += 1
        has_l, has_r = L >= 0, R < len(index)
        for k, t in enumerate(sp):
            if has_l and has_r:
                a = (k + 1) / (len(sp) + 1)
                rep[t] = index[L] if (t - L) <= (R - t) else index[R]
                lerp[t] = (1 - a) * index[L] + a * index[R]
            else:
                rep[t] = lerp[t] = index[L] if has_l else index[R]
    return {"oracle": index, "repeat": unit(rep), "lerp": unit(lerp)}


def rr(q, filled, spans):
    s = filled @ q
    scores = [max(s[t] for t in sp) for sp in spans]
    return 1.0 / (1 + sum(x > scores[0] + 1e-9 for x in scores[1:]))


def main():
    rng = np.random.default_rng(2025)
    # SigLIP 2 text: the stored frames are SigLIP 2 (timm v2_webli); SigLIP 1 text is not in their space.
    mp, sg = LocalEmbedder("all-mpnet-base-v2"), SiglipTextEmbedder("google/siglip2-base-patch16-224")
    rows = []
    for split in ["dev", "test"]:
        cap_dir = Path(f"datasets/wildQA/captions__wild{4 if split == 'dev' else 5}")
        qas = load_wildqa_dataset(Path(f"datasets/wildQA/{split}.json"), filter_scene_only=True, max_duration=60.0,
                                  single_evidence_only=True, max_evidence_duration=15.0)
        for qi, qa in enumerate(qas):
            tg = sorted(qa.evidence_indices(max_duration=60))
            cf, vf = cap_dir / f"{qa.video_id}.json", VIS_DIR / f"{qa.video_id.replace(chr(39), '_')}.npy"
            if not tg or not cf.exists() or not vf.exists():
                continue
            caps = [c["caption"] for c in json.load(open(cf))["captions"][:60]]
            V = np.load(vf)[:60]
            spans = pick_decoys(tg, K, rng)
            if len(caps) != 60 or len(V) != 60 or spans is None:
                continue
            row = dict(split=split, vid=qa.video_id, chan=re.match(r"^(.*?)_\d+", qa.video_id).group(1), n=len(tg))
            for name, emb in [("mp", mp), ("sg", sg)]:
                idx = unit(emb.get_embeddings(f"{qa.video_id}_dense_caps", caps))
                q = unit(emb.get_embeddings(f"{qa.video_id}_q_sym_{split}_{qi}", [qa.question]))[0]
                for arm, f in fills(idx, spans).items():
                    row[f"{name}_{arm}"] = rr(q, f, spans)
                if name == "sg":
                    for arm, f in fills(unit(V), spans).items():
                        row[f"vis_{arm}"] = rr(q, f, spans)
            rows.append(row)
    df = pd.DataFrame(rows)
    out = Path("results/analysis_controls")
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / f"e1_symmetric_feasibility_K{K}.csv", index=False)
    chance = np.mean([1 / r for r in range(1, K + 1)])
    print(f"K={K}: {len(df)} questions, {df.vid.nunique()} videos, {df.chan.nunique()} channels; chance MRR {chance:.3f}")
    cols = [c for c in df.columns if c.startswith(("mp_", "sg_", "vis_"))]
    chans = df.chan.unique()
    boot = []
    for _ in range(2000):
        pick = rng.choice(chans, len(chans))
        boot.append(pd.concat([df[df.chan == c] for c in pick])[cols].mean())
    boot = pd.DataFrame(boot)
    summary = pd.DataFrame({"mrr": df[cols].mean(), "lo": boot.quantile(0.025), "hi": boot.quantile(0.975)})
    print(summary.round(3).to_string())


if __name__ == "__main__":
    main()
