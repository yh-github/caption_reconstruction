import logging
from bert_score import BERTScorer
from data_models import CaptionedClip
import json

logger = logging.getLogger(__name__)

def round_metrics(metrics, ndigits=6):
    m = {}
    for k,v in metrics.items():
        if k.startswith('bs_'):
            m[k] = [round(x.item(), ndigits) for x in v]
        else:
            m[k] = v
    return m

def metrics_to_json(metrics):
    return json.dumps(round_metrics(metrics))

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
            reconstructed_clips: list[CaptionedClip],
            ground_truth_clips: list[CaptionedClip],
            masked_indices: set[int]
    ) -> dict:
        """
        Evaluates the quality of the reconstruction.

        It aligns the generated descriptions with the original ground truth
        and calculates the semantic similarity using BERTScore.

        Args:
            reconstructed_clips: The list of clips with reconstructed captions.
            ground_truth_clips: The original, unmasked list of clips.
            masked_indices: The masked indices

        Returns:
            A dictionary containing the precision, recall, and F1 score.
        """
        logger.debug("Aligning clips for BERTScore evaluation...")

        candidates, references = self._align_clips(reconstructed_clips, ground_truth_clips, masked_indices)

        if not candidates:
            logger.warning("No reconstructed clips found to evaluate.")
            return {}

        logger.debug(f"Calculating BERTScore for {len(candidates)} clip pairs.")

        P, R, F1 = self.bert_scorer.score(
            cands=candidates,
            refs=references,
            batch_size=4
        )

        # Return the results as a dictionary of floats
        metrics = {
            "num_captions": len(ground_truth_clips),
            "masked": list(masked_indices),
            "bs_p": P,
            "bs_r": R,
            "bs_f1": F1
        }

        if self.verbose:
            print('---')
            self.print_out(reconstructed_clips, ground_truth_clips, masked_indices, F1, P, R)
            print(metrics)
            print('---')

        return metrics

    def print_out(self, reconstructed_clips, ground_truth_clips, masked_indices, F1, P, R):
        j = 0
        for i in range(len(ground_truth_clips)):
            orig = ground_truth_clips[i].data.caption
            recon = reconstructed_clips[i].data.caption
            print(f'|original|reconstructed|F1|P|R|')
            if i in masked_indices:
                s = f'|~~{orig}~~|{recon}|{F1[j]:.2f}|{P[j]:.2f}|{R[j]:.2f}|'
                j += 1
            else:
                s = f'{orig}|{recon}|-|-|-|'
            print(s)

    def _align_clips(self, reconstructed_clips: dict[int, CaptionedClip], ground_truth_clips):
        """
        Private helper method to extract reference and candidate sentences.
        """
        references = []
        candidates = []

        for i, c in reconstructed_clips.items():
            candidates.append(c.data.caption)
            references.append(ground_truth_clips[i].data.caption)

        return candidates, references

    def calc_idf(self, sents: list[str]):
        self.idf = True
        print(f'calc_idf for {len(sents)} sentences')
        self.bert_scorer.compute_idf(sents=sents)
        print(f'finished calc_idf for {len(sents)} sentences, idf_dict size = {len(self.bert_scorer._idf_dict.keys())}')
