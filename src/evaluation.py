import json
import logging
import statistics
from abc import abstractmethod, ABC

from bert_score import BERTScorer
from torch import Tensor

from data_models.captions_only import CaptionedVideo
from embedder import Embedder
from eval_vectors import calculate_elementwise_cosine, VectorStats
from reconstruction_strategies import Reconstructed

logger = logging.getLogger(__name__)

def round_metrics(metrics, ndigits=6) -> dict:
    m = {}
    for k,v in metrics.items():
        if isinstance(v, Tensor): #k.startswith('bs_'):
            m[k] = [round(x.item(), ndigits) for x in v]
        elif isinstance(v, float):
            m[k] = round(v, ndigits)
        else:
            m[k] = v
    return m

def metrics_to_json(metrics):
    return json.dumps(round_metrics(metrics))


class ReconstructionEvaluator(ABC):
    @abstractmethod
    def evaluate(self, reconstructed: Reconstructed, orig: CaptionedVideo) -> dict:
        return {}

    @staticmethod
    @abstractmethod
    def agg_metrics(all_metrics):
        return {}


# noinspection PyUnusedLocal
class EvaluatorNOP(ReconstructionEvaluator):

    def evaluate(self, reconstructed: Reconstructed, orig: CaptionedVideo) -> dict:
        return {}

    @staticmethod
    def agg_metrics(all_metrics):
        return {}

class ReconstructionEvaluator_BertScore(ReconstructionEvaluator):
    """
    Encapsulates the logic for evaluating caption reconstruction using BERTScore.
    """

    def __init__(self, model_type:str|None=None, idf:bool=False, verbose=False):
        """
        Initializes the evaluator with configuration for BERTScore.

        Args:
            model_type: The Hugging Face model to use for BERTScore.
            idf: A boolean indicating whether to use inverse-document-frequency weighting.
        """
        self.model_type = model_type
        self.idf = idf
        self.verbose = verbose
        self.bert_scorer = BERTScorer(
            model_type=self.model_type,
            idf=self.idf,
            use_fast_tokenizer=False,
            lang="en"
        )
        logger.info(f"ReconstructionEvaluator initialized with model: {self.model_type}, idf: {self.idf}")

    def evaluate(
            self,
            reconstructed: Reconstructed,
            orig: CaptionedVideo
    ) -> dict:
        logger.debug("Aligning clips for BERTScore evaluation...")

        candidates, references = reconstructed.align(orig.clips)

        if not candidates:
            logger.warning("No reconstructed clips found to evaluate.")
            return {}

        logger.debug(f"Calculating BERTScore for {len(candidates)} clip pairs.")

        bs_p, bs_r, bs_f1 = self.bert_scorer.score(
            cands=candidates,
            refs=references,
            batch_size=4
        )

        return {
            "bs_p": bs_p,
            "bs_r": bs_r,
            "bs_f1": bs_f1
        }

    def calc_idf(self, sents: list[str]):
        self.idf = True
        self.bert_scorer.compute_idf(sents=sents)
        # noinspection PyProtectedMember
        logger.info(f'finished calc_idf for {len(sents)} sentences, idf_dict size = {len(self.bert_scorer._idf_dict.keys())}')
        return self

    @staticmethod
    def agg_metrics(all_metrics):
        mean_f1 = statistics.mean([m['bs_f1'].min().item() for m in all_metrics])
        mean_precision = statistics.mean([m['bs_p'].min().item() for m in all_metrics])
        mean_recall = statistics.mean([m['bs_r'].min().item() for m in all_metrics])

        return {
            "num_of_instances": len(all_metrics),
            "mean_f1_score": mean_f1,
            "mean_precision": mean_precision,
            "mean_recall": mean_recall
        }


class ReconstructionEvaluator_EmbSimilarity(ReconstructionEvaluator):
    """
    Encapsulates the logic for evaluating caption reconstruction using BERTScore.
    """

    def __init__(self, embedder: Embedder):
        self.embedder = embedder

    def evaluate(
            self,
            reconstructed: Reconstructed,
            orig: CaptionedVideo
    ) -> dict:
        logger.debug("Aligning clips for BERTScore evaluation...")

        candidates, references = reconstructed.align(orig.clips)

        if not candidates:
            logger.warning("No reconstructed clips found to evaluate.")
            return {}

        logger.debug(f"Calculating sim score for {len(candidates)} clip pairs.")

        pred_vecs = self.embedder.get_embeddings(reconstructed.video_id+"(pred)", candidates)
        true_vecs = self.embedder.get_embeddings(reconstructed.video_id+"(orig)", references)

        return VectorStats.from_vector(calculate_elementwise_cosine(pred_vecs, true_vecs)).model_dump()

    @staticmethod
    def agg_metrics(all_metrics):
        vs = [VectorStats.model_validate(m) for m in all_metrics]
        means = VectorStats.from_vector([v.mean for v in vs])

        return {
            "num_of_instances": len(all_metrics),
            "mean_mean": means.mean,
            "mean_std": means.std,
            "mean_min": means.min,
            "mean_max": means.max
        }

