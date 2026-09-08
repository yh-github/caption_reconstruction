#!/usr/bin/env python
import json
import re
from pathlib import Path
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel

PROJECT_ROOT = Path(__file__).parent.parent
WILD4_DIR = PROJECT_ROOT / "datasets" / "wildQA" / "captions__wild4"
V2_DIR = PROJECT_ROOT / "results" / "recon" / "wild4_sim_text_v2_direct" / "reconstruction" / "wild4_sim_text_v2"

print("Loading local embedding model: sentence-transformers/all-mpnet-base-v2 via transformers...")
device = "cuda" if torch.cuda.is_available() else "cpu"
model_name = "sentence-transformers/all-mpnet-base-v2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name).to(device)
model.eval()

def embed_texts(texts: list[str]) -> np.ndarray:
    if not texts:
        return np.empty((0, 768))
    encoded = tokenizer(texts, padding=True, truncation=True, max_length=128, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model(**encoded)
        # Mean pooling with attention mask
        token_embeddings = out[0]
        input_mask_expanded = encoded['attention_mask'].unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        pooled = sum_embeddings / sum_mask
        # Normalize embeddings
        normalized = torch.nn.functional.normalize(pooled, p=2, dim=1)
    return normalized.cpu().numpy()

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
    "phi-3_v2__t=0.1_rp=1.2__fixed_fill(w=3, i=59)",
    "phi-3_v2__t=0.6_rp=1.2__fixed_fill(w=3, i=0)",
    "phi-3_v2__t=0.6_rp=1.2__fixed_fill(w=3, i=29)",
    "phi-3_v2__t=0.6_rp=1.2__fixed_fill(w=3, i=59)",
    "phi-3_v2__t=1.0_rp=1.2__fixed_fill(w=3, i=0)",
    "phi-3_v2__t=1.0_rp=1.2__fixed_fill(w=3, i=29)",
    "phi-3_v2__t=1.0_rp=1.0__fixed_fill(w=3, i=0)",
    "phi-3_v2__t=1.0_rp=1.0__fixed_fill(w=3, i=29)",
    "phi-3_v2__t=1.0_rp=1.0__fixed_fill(w=3, i=59)",
]

print("\n" + "="*80)
print("=== SEMANTIC COSINE SIMILARITY & RETRIEVAL (V2 RECONSTRUCTIONS) ===")
print("="*80)

for strat in strategies:
    strat_dir = V2_DIR / strat
    if not strat_dir.exists():
        continue
    
    gt_list = []
    recon_list = []
    
    for json_path in sorted(strat_dir.glob("*.json")):
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
                gt_list.append(gt_text)
                recon_list.append(recon_text)
                
    if gt_list:
        embs_gt = embed_texts(gt_list)
        embs_rec = embed_texts(recon_list)
        
        # Pairwise cosine similarities
        cos_sims = np.sum(embs_gt * embs_rec, axis=1)
        
        print(f"\nStrategy: {strat}")
        print(f"  Total Evaluated Captions : {len(cos_sims)}")
        print(f"  Mean Cosine Similarity   : {np.mean(cos_sims):.4f} (std={np.std(cos_sims):.4f})")
        print(f"  Median Cosine Similarity : {np.median(cos_sims):.4f}")
        print(f"  Min / Max Similarity     : {np.min(cos_sims):.4f} / {np.max(cos_sims):.4f}")
