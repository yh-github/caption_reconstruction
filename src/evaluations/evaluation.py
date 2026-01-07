import logging
from abc import abstractmethod, ABC
from collections import defaultdict
from typing import Any, Generic, TypeVar, Self
import numpy as np
from bert_score import BERTScorer
from torch import Tensor
from common_utils.error_handling import UserFacingError
from data_models.captions_only import CaptionedVideo
from evaluations.eval_vectors import VectorStats, Matrix, calculate_elementwise_cosine, context_projection, calculate_retrieval_metrics
from evaluations.metrics import MetricsRecordRaw, RAW_METRIC_OBJ
from llm.embedder import GeminiEmbedder
from reconstruction.text_reconstruction import Reconstructed

logger = logging.getLogger(__name__)

T_RECON = TypeVar('T_RECON')
T_ORIG = TypeVar('T_ORIG')

class ReconstructionEvaluator(ABC, Generic[T_RECON, T_ORIG]):

    def __repr__(self) -> str:
        attrs = {k: v for k, v in self.__dict__.items() if not k.startswith('_')}
        attr_str = ", ".join([f'{k}={v!r}' for k, v in attrs.items()]) # Use !r for unambiguous repr
        return f"{self.__class__.__name__}({attr_str})"

    def __str__(self) -> str:
            return self.__repr__()

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
    def from_config(eval_conf:dict, llm_client=None):
        eval_type = eval_conf.get('type', 'bert_score').lower()
        is_embeddings = 'embeddings' in eval_conf.get('data_type','') # video_embeddings

        if is_embeddings:
            eval_type = eval_conf.get('type', 'emb_sim').lower()
            if eval_type == 'emb_sim':
                return VectorReconstructionEvaluator()
            elif eval_type == 'emb_retrieval':
                return VectorReconstructionEvaluator_Retrieval()
            elif eval_type == 'nop':
                return VectorEvaluatorNOP()
            raise UserFacingError(f"VectorReconstructionEvaluator: Unknown evaluation type '{eval_type}'")
        else:
            if eval_type == 'bert_score':
                return ReconstructionEvaluator_BertScore.build(
                    model_type=eval_conf.get('model_type', 'microsoft/deberta-large-mnli'),
                    idf=eval_conf.get('idf', True)
                )
            elif eval_type in ('emb_sim', 'emb_retrieval', 'retrieval'):
                emb_model_name = eval_conf.get('embedding_model', 'gemini')
                
                if emb_model_name.startswith('local:'):
                    from llm.local_embedder import LocalEmbedder
                    # Format: local:all-MiniLM-L6-v2
                    model_id = emb_model_name.split('local:', 1)[1]
                    if not model_id: model_id = "all-MiniLM-L6-v2" # Default
                    
                    embedder = LocalEmbedder(model_name=model_id)
                else:
                    embedder = GeminiEmbedder(client=llm_client)

                if eval_type == 'emb_sim':
                    return ReconstructionEvaluator_EmbSimilarity(embedder)
                elif eval_type == 'emb_retrieval' or eval_type == 'retrieval':
                    return ReconstructionEvaluator_Retrieval(embedder)

            elif eval_type == 'nop':
                return EvaluatorNOP()
            else:
                raise UserFacingError(f"Unknown evaluation type '{eval_type}'")


class VectorReconstructionEvaluator(ReconstructionEvaluator[Matrix, Matrix]):
    def evaluate(self, pred_vecs:Matrix, true_vecs:Matrix) -> RAW_METRIC_OBJ:
        return {"cos_sim": calculate_elementwise_cosine(pred_vecs, true_vecs)}

    def evaluate_residual(self, pred_vecs:Matrix, true_vecs:Matrix, context:Matrix) -> RAW_METRIC_OBJ:
        if isinstance(context, list):
            context = np.array(context, dtype=np.float64)

        mean_vector = context.mean(axis=0)
        pred_proj = context_projection(pred_vecs, mean_vector)
        true_proj = context_projection(true_vecs, mean_vector)

        return {
            # "cos_sim": calculate_elementwise_cosine(pred_vecs, true_vecs),
            **self.evaluate(pred_vecs, true_vecs),
            "cos_sim_residual": calculate_elementwise_cosine(pred_proj, true_proj)
        }



class VectorReconstructionEvaluator_Retrieval(VectorReconstructionEvaluator):
    def evaluate(self, pred_vecs: Matrix, true_vecs: Matrix) -> RAW_METRIC_OBJ:
        # Base cosine similarity
        base_metrics = super().evaluate(pred_vecs, true_vecs)
        
        # Retrieval metrics
        # For pure vector evaluation, we assume true_vecs represents the "ground truth" pool 
        # from which we want to retrieve the correct concept.
        # This matches the logic: each reconstructed vector should map to its corresponding ground truth vector
        # while treating all other ground truth vectors as distractors.
        ranking_metrics = calculate_retrieval_metrics(
            reconstructed_vectors=pred_vecs,
            ground_truth_vectors=true_vecs,
            distractor_pool=true_vecs
        )
        return {**base_metrics, **ranking_metrics}

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

    def __init__(self, bert_scorer: BERTScorer):
        self._bert_scorer = bert_scorer

    @property
    def model_type(self):
        return self._bert_scorer.model_type

    @property
    def idf(self):
        return self._bert_scorer.idf

    @property
    def idf_dict_size(self):
        if self.idf:
            # noinspection PyProtectedMember
            return len(self._bert_scorer._idf_dict.keys())
        return 0


    @classmethod
    def build(cls, model_type: str | None=None, idf:bool=False) -> Self:
        logger.info(f"ReconstructionEvaluator initialized with model: {model_type}, idf: {idf}")
        return cls(bert_scorer = BERTScorer(
            model_type=model_type,
            idf=idf,
            use_fast_tokenizer=False,
            lang="en"
        ))

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

        return self.to_metric_obj(*self._bert_scorer.score(
            cands=candidates,
            refs=references,
            batch_size=4
        ))

    def calc_idf(self, sents: list[str]):
        if sents:
            self._bert_scorer._idf = True
            self._bert_scorer.compute_idf(sents=sents)
            logger.info(f'finished calc_idf for {len(sents)} sentences, {self.idf_dict_size=}')
        else:
            logger.warning('no sentences, no IDF')
        return self

class ReconstructionEvaluator_EmbSimilarity(TextReconstructionEvaluator):
    """
    Encapsulates the logic for evaluating caption reconstruction using BERTScore.
    """

    def __init__(self, embedder: Any):
        self._embedder = embedder
        self._inner = VectorReconstructionEvaluator()

    def evaluate(self, reconstructed: Reconstructed, orig: CaptionedVideo) -> RAW_METRIC_OBJ:
        from llm.embedder import CacheMissError
        logger.debug("Aligning clips for EmbSimilarity evaluation...")

        candidates, references = reconstructed.align(orig.clips)

        if not candidates:
            logger.warning("No reconstructed clips found to evaluate.")
            return {}

        logger.debug(f"Calculating sim score for {len(candidates)} clip pairs.")

        try:
            pred_vecs = self._embedder.get_embeddings(reconstructed.video_id + "(pred)", candidates)
            true_vecs = self._embedder.get_embeddings(reconstructed.video_id + "(orig)", references)

            ##
            masked_inds = set(reconstructed.reconstructed_captions.keys())
            unmaksed = [x.caption for x in orig.clips if x.index not in masked_inds]
            context_vecs = self._embedder.get_embeddings(reconstructed.video_id + "(unmaksed)", unmaksed)
            ##

            return self._inner.evaluate_residual(pred_vecs=pred_vecs, true_vecs=true_vecs, context=context_vecs)
        except CacheMissError as e:
            logger.warning(f"Cache miss during evaluation for {reconstructed.video_id}: {e}")
            return {}


class ReconstructionEvaluator_Retrieval(ReconstructionEvaluator_EmbSimilarity):
    """
    Evaluates reconstruction as a retrieval task.
    In addition to cosine similarity, it ranks each reconstructed caption against
    the pool of all original captions in the video (distractors).
    """
    def evaluate(self, reconstructed: Reconstructed, orig: CaptionedVideo) -> RAW_METRIC_OBJ:
        # Get the basic sim metrics first
        base_metrics = super().evaluate(reconstructed, orig)
        if not base_metrics:
            return {}

        try:
            # We need the vectors again.
            # Optimization: ReconstructionEvaluator_EmbSimilarity doesn't expose them easily
            # without refactoring, so we might re-fetch.
            # However, since they are cached/local, it should be cheap.
            
            # Align again to be sure we match the super() logic
            candidates, references = reconstructed.align(orig.clips)
            if not candidates: return base_metrics

            pred_vecs = self._embedder.get_embeddings(reconstructed.video_id + "(pred)", candidates)
            true_vecs = self._embedder.get_embeddings(reconstructed.video_id + "(orig)", references)

            # For retrieval, the distractor pool constitutes all TRUE concepts in this video.
            # We strictly want to find the MATCHING true vec among all TRUE vecs.
            ranking_metrics = calculate_retrieval_metrics(
                reconstructed_vectors=np.array(pred_vecs),
                ground_truth_vectors=np.array(true_vecs),
                distractor_pool=np.array(true_vecs)
            )
            
            # Merge dictionary
            return {**base_metrics, **ranking_metrics}

        except Exception as e:
            logger.error(f"Error calculating retrieval metrics: {e}")
            return base_metrics
