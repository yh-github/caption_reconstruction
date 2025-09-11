import logging
import sys
from collections import Counter
import diskcache
from google import genai
from google.api_core import retry
from google.genai import types
from google.genai.types import EmbedContentResponse
from config_loader import load_config
from data.data_loaders import get_data_loader

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
    return f"disk_cache/{model}__{output_dimensionality}__{task_type}" #TODO config

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

    @staticmethod
    def log_retry(exception:Exception, try_num:int, video_id:str):
        logger.warning(f"Embedder log_retry {try_num=} {video_id=}, transient={retry.if_transient_error(exception)}, {type(exception)} -- {exception}")

    @staticmethod
    def should_retry(exception:Exception) -> bool:
        transient=retry.if_transient_error(exception)
        code = getattr(exception, 'code', 0)
        by_code = code == 429 or code>=500
        logger.warning(f"Embedder should_retry {transient=}, {by_code=} {type(exception)=}")
        return transient or by_code

    @retry.Retry(
        predicate=should_retry,  # Retry on transient API errors (e.g., 500, 503)
        initial=5.0,  # Initial delay in seconds
        maximum=60.0,  # Maximum delay in seconds
        multiplier=2.0,  # Multiplier for exponential backoff
        timeout=600.0,  # Total timeout for all retries in seconds
    )
    def _invoke_llm(self, video_id:str, texts:list[str]) -> EmbedContentResponse:
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

    def _embed_new(self, video_id:str, texts:list[str]) -> dict[str, list[float]]:
        embeddings_dict:dict[str, list[float]] = {}

        try:
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"Embeddings for {video_id}")
                for i,t in enumerate(texts, start=1):
                    logger.debug(f"  {i}. {t}")
            self._try_num = 0
            raw_res: EmbedContentResponse = self._invoke_llm(video_id, texts)
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

    def _check_emb(self, emb:list[float]|None) -> bool:
        return emb is not None and len(emb)==self.embed_config.output_dimensionality

    def embed_save(self, video_id:str, all_texts:list[str]) -> tuple[int,int,int]:
        ok=0
        fail=0

        new_texts = {t for t in all_texts if t not in self.cache}
        if not new_texts:
            logger.debug(f"Embeddings cache full hit for {video_id}")
            return ok, fail, len(all_texts)

        counts = Counter([t for t in all_texts if t in new_texts])

        for the_text, embs in self._embed_new(video_id, list(new_texts)).items():
            if self._check_emb(embs):
                self.cache[the_text] = embs
                ok += counts.get(the_text)
            else:
                fail += counts.get(the_text)
        return ok, fail, len(all_texts)-counts.total()

    def get_embeddings(self, video_id:str, all_texts:list[str]) -> list[list[float]]:
        ok, fail, hits = self.embed_save(video_id, all_texts)
        if fail>0 or ok+hits != len(all_texts):
            raise Exception(f"Embeddings FAILED for {video_id} {ok=} {fail=} {hits=} {len(all_texts)=}")
        logger.debug(f"Embeddings GOOD for {video_id} {ok=} {fail=} {hits=} {len(all_texts)=}")
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
