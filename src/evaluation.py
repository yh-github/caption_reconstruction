import json
import logging
from abc import abstractmethod, ABC
from collections import defaultdict
from typing import Any, Generic, TypeVar
from bert_score import BERTScorer
import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict
from torch import Tensor
from data_models.captions_only import CaptionedVideo
from llm.embedder import Embedder
from utils import UserFacingError
from vectors.eval_vectors import VectorStats, Matrix, calculate_elementwise_cosine
from reconstruction_strategies import Reconstructed

logger = logging.getLogger(__name__)

RAW_METRIC_OBJ=dict[str, NDArray[np.float64]]

def round_metrics(metrics, ndigits=6) -> dict:
    m = {}
    for k,v in metrics.items():
        if hasattr(v, "tolist"):
            v = v.tolist()
        if isinstance(v, list) and len(v) and isinstance(v[0], float):
            m[k] = [round(x, ndigits) for x in v]
        elif isinstance(v, float):
            m[k] = round(v, ndigits)
        else:
            m[k] = v
    return m

def metrics_to_json(metrics:dict):
    try:
        return json.dumps(metrics)
    except TypeError as te:
        s = "\n".join(f"  {k}:{v.__class__.__name__}={v}" for k,v in metrics.items())
        raise Exception(f"BAD_DICT=\n{s}\n") from te


class MetricsMetadata(BaseModel):
    model_config = ConfigDict(frozen=True)
    data_type: str
    recon_strategy: str
    video_id: str
    size: int # num_captions
    masked: list[int]


class MetricsRecord(BaseModel):
    model_config = ConfigDict(frozen=True)
    metadata: MetricsMetadata
    metrics: dict[str, VectorStats]

    def flat_metrics(self, *filter_stat:str) -> dict[str, float]:
        return {
            f"{k}_{f}":v
            for k, vs in self.metrics.items()
            for f,v in vs.model_dump().items() if not filter_stat or f in filter_stat
        }

    def to_flat_dict(self, *filter_stat:str) -> dict[str, Any]:
        d = self.metadata.model_dump()
        d.update(self.flat_metrics(*filter_stat))
        return d

class MetricsRecordRaw(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)
    metadata: MetricsMetadata
    raw_metrics: RAW_METRIC_OBJ

    def stats(self) -> MetricsRecord:
        return MetricsRecord(
            metadata=self.metadata,
            metrics={k:VectorStats.from_vector(v) for k,v in self.raw_metrics.items()}
        )

    def stats_z_score(self, global_stats:dict[str, VectorStats]) -> MetricsRecord:
        return MetricsRecord(
            metadata=self.metadata,
            metrics={k:VectorStats.from_vector((v-global_stats[k].mean)/global_stats[k].std) for k,v in self.raw_metrics.items()}
        )

########################################################################################################################

T_RECON = TypeVar('T_RECON')
T_ORIG = TypeVar('T_ORIG')

class ReconstructionEvaluator(ABC, Generic[T_RECON, T_ORIG]):
    @abstractmethod
    def evaluate(self, reconstructed: T_RECON, orig: T_ORIG) -> RAW_METRIC_OBJ:
        return {}

    @staticmethod
    def agg_metrics(all_metrics:list[MetricsRecordRaw], *filter_stat:str) -> dict[str, Any]:
        if not filter_stat:
            filter_stat = ("min", "mean")
        sums:dict[str, float] = defaultdict(float)
        counts:dict[str, int] = defaultdict(int)
        for m in all_metrics:
            for f,v in m.stats().flat_metrics(*filter_stat).items(): # XXX PARAM? INIT_PARAM? INIT_CONST?
                sums[f] += v
                counts[f] += 1

        d:dict[str, Any] = {"num_of_instances": len(all_metrics)}
        for f in sums.keys():
            d[f"mean_{f}"] = sums[f]/counts[f]
        return d

    @staticmethod
    def global_stats(all_metrics:list[MetricsRecordRaw]) -> dict[str, VectorStats]:
        d:dict[str,list[Matrix]] = defaultdict(list)
        for m in all_metrics:
            for f,v in m.raw_metrics.items():
                d[f].append(v)
        return {k:VectorStats.from_vector(np.concat(vs)) for k,vs in d.items()}

    @staticmethod
    def from_config(eval_conf:dict):
        eval_type = eval_conf.get('type', 'bert_score').lower()
        is_embeddings = 'embeddings' in eval_conf.get('data_type','') # video_embeddings

        if is_embeddings:
            eval_type = eval_conf.get('type', 'emb_sim').lower()
            if eval_type == 'emb_sim':
                return VectorReconstructionEvaluator()
            elif eval_type == 'nop':
                return VectorEvaluatorNOP()
            raise UserFacingError(f"VectorReconstructionEvaluator: Unknown evaluation type '{eval_type}'")
        else:
            if eval_type == 'bert_score':
                return ReconstructionEvaluator_BertScore(
                    model_type=eval_conf.get('model', 'microsoft/deberta-large-mnli'),
                    verbose=eval_conf.get('verbose', False),
                    idf=eval_conf.get('idf', True)
                )
            elif eval_type == 'emb_sim':
                return ReconstructionEvaluator_EmbSimilarity(Embedder())
            elif eval_type == 'nop':
                return EvaluatorNOP()
            else:
                raise UserFacingError(f"Unknown evaluation type '{eval_type}'")


class VectorReconstructionEvaluator(ReconstructionEvaluator[Matrix, Matrix]):
    def evaluate(self, pred_vecs:Matrix, true_vecs:Matrix) -> RAW_METRIC_OBJ:
        return {"cos_sim": calculate_elementwise_cosine(pred_vecs, true_vecs)}

class VectorEvaluatorNOP(VectorReconstructionEvaluator):
    def evaluate(self, pred_vecs: Matrix, true_vecs: Matrix) -> RAW_METRIC_OBJ:
        return {}

class TextReconstructionEvaluator(ReconstructionEvaluator[Reconstructed, CaptionedVideo], ABC):
    pass

class EvaluatorNOP(TextReconstructionEvaluator):

    def evaluate(self, reconstructed: Reconstructed, orig: CaptionedVideo) -> RAW_METRIC_OBJ:
        return {}

class ReconstructionEvaluator_BertScore(TextReconstructionEvaluator):
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

    @staticmethod
    def to_metric_obj(bs_p:Tensor, bs_r:Tensor, bs_f1:Tensor) -> RAW_METRIC_OBJ:
        return {
            "bs_p": bs_p.numpy(),
            "bs_r": bs_r.numpy(),
            "bs_f1": bs_f1.numpy()
        }

    def evaluate(
            self,
            reconstructed: Reconstructed,
            orig: CaptionedVideo
    ) -> RAW_METRIC_OBJ:
        logger.debug("Aligning clips for BERTScore evaluation...")

        candidates, references = reconstructed.align(orig.clips)

        if not candidates:
            logger.warning("No reconstructed clips found to evaluate.")
            return {}

        logger.debug(f"Calculating BERTScore for {len(candidates)} clip pairs.")

        return self.to_metric_obj(*self.bert_scorer.score(
            cands=candidates,
            refs=references,
            batch_size=4
        ))

    def calc_idf(self, sents: list[str]):
        if sents:
            self.idf = True
            self.bert_scorer.compute_idf(sents=sents)
            # noinspection PyProtectedMember
            logger.info(f'finished calc_idf for {len(sents)} sentences, idf_dict size = {len(self.bert_scorer._idf_dict.keys())}')
        else:
            logger.info('no IDF')
        return self

class ReconstructionEvaluator_EmbSimilarity(TextReconstructionEvaluator):
    """
    Encapsulates the logic for evaluating caption reconstruction using BERTScore.
    """

    def __init__(self, embedder: Embedder):
        self.embedder = embedder
        self.inner = VectorReconstructionEvaluator()

    def evaluate(self, reconstructed: Reconstructed, orig: CaptionedVideo) -> RAW_METRIC_OBJ:
        logger.debug("Aligning clips for EmbSimilarity evaluation...")

        candidates, references = reconstructed.align(orig.clips)

        if not candidates:
            logger.warning("No reconstructed clips found to evaluate.")
            return {}

        logger.debug(f"Calculating sim score for {len(candidates)} clip pairs.")

        pred_vecs = self.embedder.get_embeddings(reconstructed.video_id+"(pred)", candidates)
        true_vecs = self.embedder.get_embeddings(reconstructed.video_id+"(orig)", references)

        return self.inner.evaluate(pred_vecs=pred_vecs,true_vecs=true_vecs)
