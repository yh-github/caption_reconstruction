import logging
from collections import Counter
import diskcache
from google import genai
from google.api_core import retry
from google.genai import types
from google.genai.types import EmbedContentResponse

logger = logging.getLogger(__name__)

from abc import ABC, abstractmethod
import logging
from collections import Counter
import diskcache
from google import genai
from google.api_core import retry
from google.genai import types
from google.genai.types import EmbedContentResponse

logger = logging.getLogger(__name__)

def get_cache_dir(model, output_dimensionality, task_type):
    # Normalize model name for cache key if needed
    safe_model = str(model).replace('/', '_')
    return f"disk_cache/{safe_model}__{output_dimensionality}__{task_type}" 

class CacheMissError(Exception):
    pass

class BaseEmbedder(ABC):
    """
    Abstract base class for all embedders.
    Handles caching, deduplication, and batching logic.
    """
    def __init__(self, cache_dir: str, output_dimensionality: int):
        self.output_dimensionality = output_dimensionality
        self.cache_dir = cache_dir
        
        logger.info(f"Embedder cache dir: {cache_dir}")
        self.cache = diskcache.Cache(directory=cache_dir)
    
    @abstractmethod
    def _embed_new(self, video_id: str, texts: list[str]) -> dict[str, list[float]]:
        """
        Compute new embeddings for the given texts using the underlying model.
        Must return a dict mapping text -> embedding vector.
        """
        pass

    def _check_emb(self, emb: list[float]|None) -> bool:
        return emb is not None and len(emb) == self.output_dimensionality

    def _embed_save(self, video_id: str, all_texts: list[str], use_cache: bool = True) -> tuple[int, int, int]:
        ok = 0
        fail = 0

        # Optimization: Filter out texts already in cache ONLY if caching is enabled
        if use_cache:
            new_texts = {t for t in all_texts if t not in self.cache}
            if not new_texts:
                logger.debug(f"Embeddings cache full hit for {video_id}")
                return ok, fail, len(all_texts) # Hits = total
        else:
            # If not using cache, everything is "new"
            new_texts = set(all_texts)

        # Count how many times each new text appears in the input (handling duplicates)
        # We only compute once per unique text, but 'ok' count needs to reflect all occurrences
        counts = Counter([t for t in all_texts if t in new_texts])

        # Batch compute (delegated to subclass)
        computed_embeddings = self._embed_new(video_id, list(new_texts))
        
        for the_text, embs in computed_embeddings.items():
            if self._check_emb(embs):
                if use_cache:
                    self.cache[the_text] = embs
                ok += counts.get(the_text, 0)
            else:
                fail += counts.get(the_text, 0)
                
        # Calculate hits: total requested - total we had to compute (ok+fail)
        hits = len(all_texts) - counts.total()
        return ok, fail, hits

    def get_embeddings(self, video_id: str, all_texts: list[str], use_cache: bool = True) -> list[list[float]]:
        ok, fail, hits = self._embed_save(video_id, all_texts, use_cache=use_cache)
        
        if fail > 0 or ok + hits != len(all_texts):
             # Try to be robust: if we have failures, we raise, 
             # because misalignment is catastrophic for vectors
            raise Exception(f"Embeddings FAILED for {video_id} {ok=} {fail=} {hits=} {len(all_texts)=}")
            
        logger.debug(f"Embeddings GOOD for {video_id} {ok=} {fail=} {hits=} {len(all_texts)=}")
        
        # If not using cache, we must return from the computed set + whatever was in cache
        # This part is tricky because BaseEmbedder design relies heavily on diskcache.
        # To avoid refactoring everything: if use_cache=False, we compute and return but DON'T save.
        # BUT wait: _embed_save doesn't return the embeddings.
        
        # REFACTOR: _embed_new returns dict. We should use that.
        
        # NOTE: For simplicity in this codebase structure:
        # If use_cache=False, we re-compute everything and don't touch self.cache.
        # If use_cache=True, we use standard logic.
        
        if not use_cache:
             # Direct compute without cache interaction
             # Note: this might be inefficient if there are duplicates in `all_texts` list, 
             # but `_embed_new` usually handles batching.
             # Actually `_embed_new` expects unique texts probably? The implementation of `_embed_new` 
             # just maps input list to output dict.
             # Let's check `_embed_save` logic again. It computes for `new_texts` (set).
             
             # Re-implement simple logic for no-cache:
             unique_texts = list(set(all_texts))
             computed = self._embed_new(video_id, unique_texts)
             return [computed[t] for t in all_texts]
             
        
        return [self.cache[c] for c in all_texts]

class GeminiEmbedder(BaseEmbedder):
    """
    Embedder implementation using Google GenAI (Gemini) API.
    """
    def __init__(self, model="gemini-embedding-001", output_dimensionality=512, task_type="SEMANTIC_SIMILARITY", client=None):
        self.model = model
        self.embed_config = types.EmbedContentConfig(
            output_dimensionality=output_dimensionality,
            task_type=task_type
        )
        
        # Initialize Cache via Base
        cache_dir = get_cache_dir(model, output_dimensionality, task_type)
        super().__init__(cache_dir, output_dimensionality)

        if client is None:
            logger.warning("Embedder initialized with client=None. Creating default genai.Client()!")
            self.client = genai.Client()
        else:
            logger.info(f"Embedder initialized with provided client: {type(client)}")
            self.client = client
            
        self._try_num = 0

    def __repr__(self):
        return f"{self.__class__.__name__}({self.model=}, {', '.join(f'{k}={v}' for k,v in self.embed_config.model_dump(exclude_none=True))})"

    @staticmethod
    def log_retry(exception: Exception, try_num: int, video_id: str):
        logger.warning(f"Embedder log_retry {try_num=} {video_id=}, transient={retry.if_transient_error(exception)}, {type(exception)} -- {exception}")

    @staticmethod
    def should_retry(exception: Exception) -> bool:
        transient = retry.if_transient_error(exception)
        code = getattr(exception, 'code', 0)
        by_code = code == 429 or code >= 500
        logger.warning(f"Embedder should_retry {transient=}, {by_code=} {type(exception)=}")
        return transient or by_code

    @retry.Retry(
        predicate=should_retry,
        initial=5.0,
        maximum=60.0,
        multiplier=2.0,
        timeout=600.0,
    )
    def _invoke_llm(self, video_id: str, texts: list[str]) -> EmbedContentResponse:
        try:
            self._try_num += 1
            return self.client.models.embed_content(
                model=self.model,
                config=self.embed_config,
                contents=texts
            )
        except Exception as ex:
            self.log_retry(ex, self._try_num, video_id)
            raise

    def _embed_new(self, video_id: str, texts: list[str]) -> dict[str, list[float]]:
        embeddings_dict: dict[str, list[float]] = {}

        try:
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"Embeddings for {video_id}")
                for i, t in enumerate(texts, start=1):
                    logger.debug(f"  {i}. {t}")
            self._try_num = 0
            raw_res: EmbedContentResponse = self._invoke_llm(video_id, texts)
        except CacheMissError:
            raise
        except Exception as e:
            logger.error(f"Embeddings failed for {video_id}, ** {e.__class__.__qualname__} ** {e}")
            return embeddings_dict

        if not raw_res or not raw_res.embeddings:
            logger.error(f"Embeddings failed for {video_id}")
            return embeddings_dict

        if len(raw_res.embeddings) != len(texts):
            logger.warning(f"Embeddings failed for {video_id} {len(raw_res.embeddings)=} {len(texts)=}")

        logger.debug(f"New embeddings for {video_id} done {len(raw_res.embeddings)=} {len(texts)=}")

        for i, embedding in enumerate(raw_res.embeddings):
            es = embedding.values
            if es is None:
                logger.warning(f"Embeddings failed for {video_id} {i=}")
            embeddings_dict[texts[i]] = es
        return embeddings_dict

# Backward compatibility alias
Embedder = GeminiEmbedder
