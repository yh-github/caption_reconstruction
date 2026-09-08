#!/usr/bin/env python
import json
from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer

PROJECT_ROOT = Path(__file__).parent.parent
WILD4_DIR = PROJECT_ROOT / "datasets" / "wildQA" / "captions__wild4"
V2_DIR = PROJECT_ROOT / "results" / "recon" / "wild4_sim_text_v2_direct" / "reconstruction" / "wild4_sim_text_v2"

print("Loading local sentence transformer model: all-mpnet-base-v2...")
embedder = SentenceTransformer("all-mpnet-base-v2")

# Load GT
gt_map = {}
for p in WILD4_DIR.glob("*.json"):
    if p.name == "categories.json": continue
    with open(p) as f:
        d = json.load(f)
    gt_map[p.stem] = [c["caption"] for c in d.get("captions", [])]

strategies = [
    "phi-3_v2__t=0.1_rp=1.2__fixed_fill(w=3, i=0)",
    "phi-3_v2__t=0.1_rp=1.2__fixed_fill(w=3, i=29)",
    "phi-3_v2__t=0.6_rp=1.2__fixed_fill(w=3, i=0)",
    "phi-3_v2__t=0.6_rp=1.2__fixed_fill(w=3, i=29)",
    "phi-3_v2__t=1.0_rp=1.2__fixed_fill(w=3, i=0)",
    "phi-3_v2__t=1.0_rp=1.2__fixed_fill(w=3, i=29)",
]

print("\n=== SEMANTIC SIMILARITY EVALUATION (V2) ===")
for strat in strategies:
    strat_dir = V2_DIR / strat
    if not strat_dir.exists():
        continue
    
    sims = []
    for json_path in strat_dir.glob("*.json"):
        vid = json_path.stem
        if vid not in gt_map:
            continue
        gt_caps = gt_map[vid]
        
        with open(json_path) as f:
            data = json.load(f)
        recons = data.get("reconstructed_captions", {})
        
        for idx_str, recon_text in recons.items():
            idx = int(idx_str)
            if idx < len(gt_caps) and recon_text and recon_text.strip():
                gt_text = gt_caps[idx]
                emb_gt = embedder.encode(gt_text)
                emb_rec = embedder.encode(recon_text)
                cos_sim = np.dot(emb_gt, emb_rec) / (np.linalg.norm(emb_gt) * np.linalg.norm(emb_rec))
                sims.append(cos_sim)
                
    if sims:
        print(f"Strategy: {strat}")
        print(f"  Total evaluated clip captions: {len(sims)}")
        print(f"  Mean Cosine Similarity       : {np.mean(sims):.4f} (std={np.std(sims):.4f})")
        print(f"  Median Cosine Similarity     : {np.median(sims):.4f}")
