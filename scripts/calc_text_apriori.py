import json
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
import sys
sys.path.append(str(Path(__file__).parent.parent / "src"))

from llm.local_embedder import LocalEmbedder

def compute_text_dynamism(embeddings):
    n = len(embeddings)
    if n < 2:
        return 0.0, 0.0, 0.0, 0.0
    
    seq_dists = []
    for i in range(n - 1):
        v1 = embeddings[i]
        v2 = embeddings[i+1]
        sim = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9)
        seq_dists.append(1.0 - sim)
    
    avg_dyn = np.mean(seq_dists)
    peak_dyn = np.percentile(seq_dists, 95)
    combined_dyn = 100.0 * (0.5 * avg_dyn + 0.5 * peak_dyn)
    
    sim_matrix = np.dot(embeddings, embeddings.T)
    norms = np.linalg.norm(embeddings, axis=1)
    norm_matrix = np.outer(norms, norms) + 1e-9
    sim_matrix = sim_matrix / norm_matrix
    
    i_upper, j_upper = np.triu_indices(n, k=1)
    apcs_t = np.mean(sim_matrix[i_upper, j_upper])
    
    return avg_dyn, peak_dyn, combined_dyn, apcs_t

def main():
    embedder = LocalEmbedder("all-mpnet-base-v2")
    
    dirs = [Path("datasets/wildQA/captions__wild4"), Path("datasets/wildQA/captions__wild5")]
    json_files = []
    for d in dirs:
        json_files.extend([f for f in d.glob("*.json") if f.stem != "categories"])
    
    results = []
    for jf in tqdm(json_files, desc="Processing text dynamism"):
        with open(jf) as f:
            data = json.load(f)
        
        video_id = data.get("video_id", jf.stem)
        captions = data.get("captions", [])
        texts = [c.get("caption", "") for c in captions]
        
        if not texts:
            continue
            
        emb_dict = embedder._embed_new(video_id, texts)
        embs = [emb_dict[t] for t in texts]
        
        avg_dyn, peak_dyn, comb_dyn, apcs_t = compute_text_dynamism(np.array(embs))
        
        results.append({
            "video_id": video_id,
            "text_average_dynamism": avg_dyn,
            "text_peak_dynamism": peak_dyn,
            "text_combined_dynamism": comb_dyn,
            "APCS_T": apcs_t,
            "num_captions": len(texts)
        })
    
    df = pd.DataFrame(results).drop_duplicates("video_id")
    df.to_csv("results/apriori_textual_dynamism.csv", index=False)
    print(f"Saved textual dynamism for {len(df)} videos.")

if __name__ == "__main__":
    main()
