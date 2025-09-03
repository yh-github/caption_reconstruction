import logging
import sys
import diskcache
from google import genai
from google.genai import types
from config_loader import load_config
from data_loaders import get_data_loader
from data_models.captions_only import CaptionedVideo

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
                contents=texts
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

    def embed_save(self, video_id:str, all_texts:list[str]) -> tuple[int,int,int]:
        ok=0
        fail=0

        texts = [c for c in all_texts if c not in self.cache]
        if not texts:
            logger.debug(f"Embeddings cache full hit for {video_id}")
            return ok, fail, len(all_texts)

        for k,v in self._embed_new(video_id, texts).items():
            self.cache[k] = v
            if v:
                ok+=1
            else:
                fail+=1
        return ok, fail, len(all_texts)-len(texts)

    def get_embeddings(self, video_id:str, all_texts:list[str]) -> list[list[float]]:
        ok, fail, hits = self.embed_save(video_id, all_texts)
        if fail>0 or ok+hits!=len(all_texts):
            raise Exception(f"Embeddings failed for {video_id} {ok=} {fail=} {hits=} {len(all_texts)=}")
        return [self.cache[c] for c in all_texts]

    # def sim(self, video:CaptionedVideo):
    #     embeddings_matrix = np.array(self.get_embeddings(video))
    #     similarity_matrix = cosine_similarity(embeddings_matrix)


def main(config, cmd):
    data_loader = get_data_loader(config["data_config"])
    embedder = Embedder(model="gemini-embedding-001", output_dimensionality=512, task_type="SEMANTIC_SIMILARITY")
    data = data_loader.load()
    if cmd == "emb" or cmd == "embed":
        for _video in data:
            ok, fail, hits = embedder.embed_save(
                video_id=_video.video_id,
                all_texts=[c.caption for c in _video.clips]
            )
            embs = embedder.get_embeddings(
                video_id=_video.video_id,
                all_texts=[c.caption for c in _video.clips]
            )
            log_msg = f"{_video.video_id} ok={ok} fail={fail} hits={hits} all={len(_video.clips)}"
            if len(embs) == len(_video.clips):
                logger.info(log_msg)
            else:
                logger.warning(f"{log_msg} BUT {len(embs)=}")

    # elif cmd == "cos" or cmd == "cosine":
    #     for _video in data:
    #         embedder.sim(_video)

def parse_args(argv):
    if len(argv) < 3:
        print("Please provide the path to the experiment config file.")
        sys.exit(1)

    config = load_config(argv[1])
    cmd = argv[2]

    return config, cmd

if __name__ == "__main__":
    main(*parse_args(sys.argv))