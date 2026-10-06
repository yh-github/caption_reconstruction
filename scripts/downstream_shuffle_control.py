"""Shuffled-reconstruction control for the downstream WildQA evidence retrieval result.
# Run from the repo root: .venv/bin/python scripts/downstream_shuffle_control.py [all-mpnet-base-v2|siglip]

If filling the evidence gap with Llama text from a *different* video retrieves the evidence
about as well as the true reconstruction, the downstream gain is a style/genericness artifact.
"""
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, "src")
from data.wildqa_loader import load_wildqa_dataset
from evaluations.evidence_retrieval import calculate_evidence_retrieval_metrics, build_evidence_retrieval_index

RECON = {
    "dev": Path.home() / ".cache/huggingface/hub/datasets--Y3--dense_video_captions/snapshots/ab0a83c133adc18daa9e42273fdc0b2c6e9ab921/reconstruction/wild4_llama_evidence",
    "test": Path("results/recon/manual_download/reconstruction/wild5_llama_evidence"),
}
emb_name = sys.argv[1] if len(sys.argv) > 1 else "all-mpnet-base-v2"
if emb_name == "siglip":
    from llm.local_embedder import SiglipTextEmbedder
    E = SiglipTextEmbedder()
else:
    from llm.local_embedder import LocalEmbedder
    E = LocalEmbedder(emb_name)

def emb(key, texts):
    return np.array(E.get_embeddings(key, texts), dtype=np.float32)

rng = np.random.default_rng(0)
items = []
for split in ["dev", "test"]:
    rdir = next(RECON[split].iterdir())
    recon = {}
    for f in rdir.glob("*.json"):
        if f.name.startswith("skip__"):
            continue
        d = json.load(open(f))
        recon[f.stem] = {int(k): v for k, v in d.get("reconstructed_captions", {}).items()}
    cap_dir = Path(f"datasets/wildQA/captions__wild{4 if split == 'dev' else 5}")
    qas = load_wildqa_dataset(Path(f"datasets/wildQA/{split}.json"), filter_scene_only=True, max_duration=60.0,
                              single_evidence_only=True, max_evidence_duration=15.0)
    for qa in qas:
        vid = qa.video_id
        tg = sorted(qa.evidence_indices(max_duration=60))
        cf = cap_dir / f"{vid}.json"
        if not tg or vid not in recon or not cf.exists():
            continue
        caps = [c["caption"] for c in json.load(open(cf))["captions"][:60]]
        rt = [recon[vid].get(t, "") for t in tg]
        if len(caps) != 60 or not all(rt):
            continue
        items.append(dict(split=split, vid=vid, domain=qa.domain, q=qa.question, tg=tg, caps=caps, recon=rt))

print(f"{emb_name}: {len(items)} questions, {len({i['vid'] for i in items})} videos")
all_recon = [(i["vid"], i["domain"], i["recon"]) for i in items]
rows = []
for k, it in enumerate(items):
    idx = emb(f"{it['vid']}_dense_caps", it["caps"])
    q = emb(f"{it['vid']}_q_ctl_{k}", [it["q"]])[0]
    n = len(it["tg"])
    def score(texts, tag):
        e = emb(f"{it['vid']}_ctl_{tag}_{k}", texts)
        return calculate_evidence_retrieval_metrics(q, build_evidence_retrieval_index(idx, it["tg"], "reconstructed", e), it["tg"])["rank"]
    others_any = [r for v, d, r in all_recon if v != it["vid"]]
    others_dom = [r for v, d, r in all_recon if v != it["vid"] and d == it["domain"]] or others_any
    def pick(pool):
        r = pool[rng.integers(len(pool))]
        return [r[j % len(r)] for j in range(n)]
    rows.append(dict(
        split=it["split"], domain=it["domain"], n=n,
        oracle=calculate_evidence_retrieval_metrics(q, idx, it["tg"])["rank"],
        repeat=calculate_evidence_retrieval_metrics(q, build_evidence_retrieval_index(idx, it["tg"], "baseline_repeat"), it["tg"])["rank"],
        recon=score(it["recon"], "true"),
        shuf_any=score(pick(others_any), "any"),
        shuf_dom=score(pick(others_dom), "dom"),
    ))
df = pd.DataFrame(rows)
out = Path("results/analysis_controls") / f"downstream_shuffle_control_{emb_name}.csv"; out.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(out, index=False)
cols = ["oracle", "repeat", "recon", "shuf_any", "shuf_dom"]
print("MRR:", {c: round((1 / df[c]).mean(), 3) for c in cols})
print("R@5:", {c: round((df[c] <= 5).mean(), 3) for c in cols})
from scipy.stats import wilcoxon
for a, b in [("recon", "shuf_dom"), ("recon", "shuf_any"), ("recon", "repeat"), ("shuf_dom", "repeat")]:
    diff = df[a] - df[b]
    print(f"{a} vs {b}: wins={int((diff<0).sum())} losses={int((diff>0).sum())} p={wilcoxon(df[a], df[b]).pvalue:.3g}")
