import logging
from bert_score import score as bert_score_func
from data_models import CaptionedClip

logger = logging.getLogger(__name__)


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
            return {"bert_score_precision": 0.0, "bert_score_recall": 0.0, "bert_score_f1": 0.0}

        logger.debug(f"Calculating BERTScore for {len(candidates)} clip pairs.")

        # Calculate BERTScore using the instance attributes
        P, R, F1 = bert_score_func(
            cands=candidates,
            refs=references,
            model_type=self.model_type,
            idf=self.idf,
            verbose=False,
            use_fast_tokenizer=False,
            lang="en"
        )

        # Return the results as a dictionary of floats
        metrics = {
            "bert_score_precision": P.mean().item(),
            "bert_score_recall": R.mean().item(),
            "bert_score_f1": F1.mean().item()
        }

        if self.verbose:
            print('---')
            self.print_out(reconstructed_clips, ground_truth_clips, masked_indices)
            print(metrics)
            print('---')

        return metrics

    def print_out(self, reconstructed_clips, ground_truth_clips, masked_indices):
        for i in range(len(ground_truth_clips)):
            s = ground_truth_clips[i].data.description
            if i in masked_indices:
                s = f'~~{s}~~'
            s = f'{s}\t{reconstructed_clips[i].data.description}'
            print(s)

    def _align_clips(self, reconstructed_clips, ground_truth_clips, masked_indices):
        """
        Private helper method to extract reference and candidate sentences.
        """
        references = []
        candidates = []

        for i in masked_indices:
            candidates.append(reconstructed_clips[i].data.description)
            references.append(ground_truth_clips[i].data.description)

        return candidates, references