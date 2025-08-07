import sys

from google import genai
from google.genai import types
import diskcache
import logging

from config_loader import load_config
from data_loaders import get_data_loader
from data_models import CaptionedVideo

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


logger = logging.getLogger()
logger.setLevel(logging.INFO)
# Create a handler
handler = logging.StreamHandler()  # or logging.FileHandler('filename.log')
handler.setLevel(logging.INFO)

# Create and set formatter
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)

# Add handler to logger
logger.addHandler(handler)

def get_cache_dir(model, output_dimensionality, task_type):
    return f"disk_cache/{model}__{output_dimensionality}__{task_type}"

class Embedder:
    def __init__(self, model="gemini-embedding-001", output_dimensionality=512, task_type="SEMANTIC_SIMILARITY"):
        """
        model: "gemini-embedding-001"
        output_dimensionality: 512, 768
        task_type: "SEMANTIC_SIMILARITY", "RETRIEVAL_DOCUMENT", "CLUSTERING"
        """
        self.model = model
        self.embed_config = types.EmbedContentConfig(
            output_dimensionality=output_dimensionality,
            task_type=task_type
        )

        cache_dir = get_cache_dir(model, output_dimensionality, task_type)

        logger.info(f"Embedder cache dir: {cache_dir}")
        self.client = genai.Client()
        self.cache = diskcache.Cache(directory=cache_dir)

    def _embed_new(self, video_id:str, texts:list[str]) -> dict[str, list[float]]:
        embeddings_dict:dict[str, list[float]] = {}

        try:
            raw_res = self.client.models.embed_content(
                model=self.model,
                config=self.embed_config,
                contents=texts,
            )
        except Exception as e:
            logger.error(f"Embeddings failed for {video_id}, ** {e.__class__.__qualname__} ** {e}")
            return embeddings_dict

        if not raw_res or not raw_res.embeddings:
            logger.error(f"Embeddings failed for {video_id}")
            return embeddings_dict

        if len(raw_res.embeddings) != len(texts):
            logger.warning(f"Embeddings failed for {video_id} {len(raw_res.embeddings)=} {len(texts)=}")

        for i, embedding in enumerate(raw_res.embeddings):
            es = embedding.values
            if es is None:
                logger.warning(f"Embeddings failed for {video_id} {i=}")
            embeddings_dict[texts[i]] = es
        return embeddings_dict

    def embed_save(self, video:CaptionedVideo):
        ok=0
        fail=0

        texts = [c.caption for c in video.clips if c.caption not in self.cache]
        if not texts:
            logger.debug(f"Embeddings cache full hit for {video.video_id}")
            return ok, fail, len(video.clips)

        for k,v in self._embed_new(video.video_id, texts).items():
            self.cache[k] = v
            if v:
                ok+=1
            else:
                fail+=1
        return ok, fail, len(video.clips)-len(texts)

    def get_embeddings(self, video:CaptionedVideo):
        ok, fail, hits = self.embed_save(video)
        if fail>0 or ok+hits!=len(video.clips):
            raise Exception(f"Embeddings failed for {video.video_id} {ok=} {fail=} {hits=} {len(video.clips)=}")
        return [self.cache[c.caption] for c in video.clips]


    def sim(self, video:CaptionedVideo):
        embeddings_matrix = np.array(self.get_embeddings(video))
        similarity_matrix = cosine_similarity(embeddings_matrix)



if __name__ == "__main__":
    if len(sys.argv) < 3:
        "Please provide the path to the experiment config file."

    config = load_config(sys.argv[1])
    cmd = sys.argv[2]
    data_loader = get_data_loader(config["data_config"])

    embedder = Embedder(model="gemini-embedding-001", output_dimensionality=512, task_type="SEMANTIC_SIMILARITY")
    data = data_loader.load()

    if cmd == "emb" or cmd == "embed":
        for _video in data:
            ok, fail, hits = embedder.embed_save(_video)
            logger.info(f"{_video.video_id} ok={ok} fail={fail} hits={hits} all={len(_video.clips)}")

    elif cmd == "cos" or cmd == "cosine":
        for _video in data:
            embedder.sim(_video)