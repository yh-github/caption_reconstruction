import transformers
from llm.local_embedder import LocalEmbedder
import logging

logging.basicConfig(level=logging.DEBUG)

def verify():
    print("Initializing LocalEmbedder...")
    embedder = LocalEmbedder(model_name="all-MiniLM-L6-v2")
    
    texts = ["Hello world", "This is a test"]
    print(f"Embedding: {texts}")
    
    embs = embedder.get_embeddings("test_video", texts)
    
    print(f"Got {len(embs)} embeddings.")
    print(f"Dim: {len(embs[0])}")
    
    assert len(embs) == 2
    assert len(embs[0]) == 384 # Default for MiniLM
    print("Verification Successful!")

if __name__ == "__main__":
    verify()
