import sys
import numpy as np
import torch
from pathlib import Path
sys.path.append(str(Path('.').absolute() / 'src'))

from llm.local_embedder import SiglipTextEmbedder
from data.video_embeddings import VideoEmbedder

def test_text_embedder():
    print("Testing Text Embedder...")
    embedder = SiglipTextEmbedder()
    texts = ["A sample caption"]
    res = embedder.get_embeddings("test_vid", texts)
    
    vec = res[0]
    assert len(vec) == 768, f"Expected 768 dims, got {len(vec)}"
    
    norm = np.linalg.norm(vec)
    assert np.isclose(norm, 1.0, atol=1e-3), f"Expected norm 1.0, got {norm}"
    print("✅ Text Embedder OK (dim=768, norm=1.0)")

def test_video_embedder():
    print("Testing Video Embedder (Model Load Only)...")
    embedder = VideoEmbedder(model_name="google/siglip-base-patch16-224")
    print("✅ Video Embedder initialized properly.")
    
def test_cross_modal():
    print("Testing Cross Modal Setup...")
    from evaluations.evaluation import ReconstructionEvaluator_CrossModal
    print("✅ Cross Modal class imported.")

if __name__ == "__main__":
    test_text_embedder()
    test_video_embedder()
    test_cross_modal()
    print("\nAll Sanity Checks Passed!")
