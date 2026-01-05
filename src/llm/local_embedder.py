import logging
import diskcache
import torch
import transformers
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

def get_cache_dir(model_name: str):
    return f"disk_cache/local_{model_name.replace('/', '_')}"

from llm.embedder import BaseEmbedder

class LocalEmbedder(BaseEmbedder):
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", device: str = None):
        """
        model_name: Name of the sentence-transformer model
        """
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        logger.info(f"Initializing LocalEmbedder with {model_name} on {self.device}")
        self.model = SentenceTransformer(model_name, device=self.device)
        
        # Determine output dimensionality dynamically or hardcode for known models?
        # SentenceTransformers usually have .get_sentence_embedding_dimension()
        out_dim = self.model.get_sentence_embedding_dimension()
        
        # Initialize Base (handles cache dir and diskcache)
        cache_dir = get_cache_dir(model_name)
        super().__init__(cache_dir, out_dim)

    def _embed_new(self, video_id: str, texts: list[str]) -> dict[str, list[float]]:
        """
        Compute new embeddings for the given texts using the local model.
        """
        if not texts:
            return {}
            
        logger.debug(f"Computing {len(texts)} new embeddings locally for {video_id}")
        
        # Compute
        embeddings = self.model.encode(texts, convert_to_tensor=False, show_progress_bar=False)
        
        # Map back to texts
        result = {}
        for text, emb in zip(texts, embeddings):
            # Ensure list format
            result[text] = emb.tolist() if hasattr(emb, 'tolist') else list(emb)
            
        return result
