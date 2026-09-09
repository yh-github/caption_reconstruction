import logging
import diskcache
import torch
import transformers
try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None
    from transformers import AutoTokenizer, AutoModel
from common_utils import device_setup

logger = logging.getLogger(__name__)

def get_cache_dir(model_name: str):
    return f"disk_cache/local_{model_name.replace('/', '_')}"

from llm.embedder import BaseEmbedder

class _TransformersEmbedder:
    def __init__(self, model_name: str, device: str):
        self.device = device
        # Handle 'all-mpnet-base-v2' or 'sentence-transformers/all-mpnet-base-v2'
        hf_name = model_name if "/" in model_name else f"sentence-transformers/{model_name}"
        self.tokenizer = AutoTokenizer.from_pretrained(hf_name)
        self.model = AutoModel.from_pretrained(hf_name).to(device)
        self.model.eval()

    def get_sentence_embedding_dimension(self) -> int:
        return getattr(self.model.config, "hidden_size", 768)

    def encode(self, texts: list[str], convert_to_tensor=False, show_progress_bar=False):
        if not texts:
            return []
        encoded = self.tokenizer(texts, padding=True, truncation=True, max_length=128, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.model(**encoded)
            token_embeddings = out[0]
            input_mask_expanded = encoded['attention_mask'].unsqueeze(-1).expand(token_embeddings.size()).float()
            sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
            sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            pooled = sum_embeddings / sum_mask
            normalized = torch.nn.functional.normalize(pooled, p=2, dim=1)
        return normalized.cpu().numpy()

class LocalEmbedder(BaseEmbedder):
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", device: str = None):
        """
        model_name: Name of the sentence-transformer model
        """
        self.model_name = model_name
        self.device = device or device_setup.get_device()
        
        logger.info(f"Initializing LocalEmbedder with {model_name} on {self.device}")
        if SentenceTransformer is not None:
            self.model = SentenceTransformer(model_name, device=self.device)
        else:
            self.model = _TransformersEmbedder(model_name, device=self.device)
        
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

class SiglipTextEmbedder(BaseEmbedder):
    def __init__(self, model_name: str = "google/siglip-base-patch16-224", device: str = None):
        """
        model_name: Name of the SigLIP model
        """
        self.model_name = model_name
        self.device = device or device_setup.get_device()
        
        logger.info(f"Initializing SiglipTextEmbedder with {model_name} on {self.device}")
        
        from transformers import AutoTokenizer, SiglipTextModel
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.device != "cpu":
            self.model = SiglipTextModel.from_pretrained(model_name, torch_dtype=torch.float16, attn_implementation="sdpa").to(self.device)
        else:
            self.model = SiglipTextModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
        
        out_dim = getattr(self.model.config, "hidden_size", 768)
        
        cache_dir = get_cache_dir(model_name + "_text")
        super().__init__(cache_dir, out_dim)

    def _embed_new(self, video_id: str, texts: list[str]) -> dict[str, list[float]]:
        if not texts:
            return {}
            
        logger.debug(f"Computing {len(texts)} new SigLIP text embeddings locally for {video_id}")
        
        inputs = self.tokenizer(
            texts,
            padding="max_length",
            max_length=64,
            truncation=True,
            return_tensors="pt"
        )
        if hasattr(inputs, "to"):
            inputs = inputs.to(self.device)
        else:
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            raw_embeds = outputs.pooler_output
            embeddings = torch.nn.functional.normalize(raw_embeds, p=2, dim=-1)
            
        embeddings_np = embeddings.cpu().numpy()
        
        result = {}
        for text, emb in zip(texts, embeddings_np):
            result[text] = emb.tolist()
            
        return result
