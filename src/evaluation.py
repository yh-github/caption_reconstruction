import logging
from bert_score import BERTScorer
from data_models import CaptionedVideo
import json
from reconstruction_strategies import Reconstructed

logger = logging.getLogger(__name__)

def round_metrics(metrics, ndigits=6) -> dict:
    m = {}
    for k,v in metrics.items():
        if k.startswith('bs_'):
            m[k] = [round(x.item(), ndigits) for x in v]
        else:
            m[k] = v
    return m

def metrics_to_json(metrics):
    return json.dumps(metrics)

class ReconstructionEvaluator:
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

        metrics = {
            "bs_p": bs_p,
            "bs_r": bs_r,
            "bs_f1": bs_f1
        }

        return metrics

    def calc_idf(self, sents: list[str]):
        self.idf = True
        self.bert_scorer.compute_idf(sents=sents)
        logger.info(f'finished calc_idf for {len(sents)} sentences, idf_dict size = {len(self.bert_scorer._idf_dict.keys())}')
        return self
