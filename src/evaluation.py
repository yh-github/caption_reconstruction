import logging
from bert_score import score as bert_score
from data_models import CaptionedClip
from constants import DATA_MISSING

def evaluate_reconstruction(
    reconstructed_clips: list[CaptionedClip],
    ground_truth_clips: list[CaptionedClip]
) -> dict:
    """
    Evaluates the quality of the reconstruction using BERTScore.

    It aligns the reconstructed descriptions with the original descriptions
    from the ground truth and calculates the semantic similarity.

    Args:
        reconstructed_clips: The full caption list returned and parsed from the LLM.
        ground_truth_clips: The original, unmasked caption.

    Returns:
        A dictionary containing the precision, recall, and F1 score from BERTScore.
    """
    logging.info("Evaluating reconstruction using BERTScore...")
    
    # Align the ground truth and reconstructed descriptions
    # We only want to compare the clips that were originally masked.
    references = []
    candidates = []

    for i, recon_clip in enumerate(reconstructed_clips):
        gt_clip = ground_truth_clips[i]
        
        # Check if this clip was a masked one in the final LLM output
        # A simple heuristic: if the data is not our MASK token, it was generated.
        if recon_clip.data != DATA_MISSING:
            # And if the original had actual data (it wasn't a mistake)
            if gt_clip.data != DATA_MISSING:
                # We have a pair to compare
                candidates.append(recon_clip.data.description)
                references.append(gt_clip.data.description)

    if not candidates:
        logging.warning("No reconstructed clips found to evaluate.")
        return {"bert_score_precision": 0, "bert_score_recall": 0, "bert_score_f1": 0}

    # Calculate BERTScore
    # The 'lang="en"' argument defaults to a standard English model.
    # We can make this configurable later.
    P, R, F1 = bert_score(candidates, references, lang="en", verbose=False)

    # Return the results as a dictionary of floats
    metrics = {
        "bert_score_precision": P.mean().item(),
        "bert_score_recall": R.mean().item(),
        "bert_score_f1": F1.mean().item()
    }
    
    logging.info(f"Evaluation complete. BERTScore F1: {metrics['bert_score_f1']:.4f}")
    return metrics
