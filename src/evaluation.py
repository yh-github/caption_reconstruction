# src/evaluation.py
import logging
from joblib import Memory
from bert_score import score as bert_score
from data_models import TranscriptClip
from constants import DATA_MISSING

# --- Caching Setup ---
# Initialize a memory object. This will create a cache directory and
# store the results of the decorated function there.
# We will get the cache path from our config.
memory = None

def initialize_cache(cache_path: str):
    """Initializes the joblib cache at a specified path."""
    global memory
    logging.info(f"Initializing cache at: {cache_path}")
    memory = Memory(cache_path, verbose=0)

# --- Embedding and Evaluation Functions ---

# This function is a placeholder for a real embedding function.
# In a real scenario, this would call an LLM's embedding endpoint.
# The @memory.cache decorator ensures the result is cached.
def get_embedding(text: str):
    """
    (SIMULATED) A placeholder for a function that would fetch text embeddings.
    NOTE: For the purpose of this example, we return a dummy value.
    The real power comes from the caching, not the function itself.
    """
    logging.debug(f"Cache miss. Generating dummy embedding for: '{text}'")
    # In a real implementation, this would be a high-dimensional vector.
    return [len(text)] 

def evaluate_reconstruction(
    reconstructed_clips: list[TranscriptClip],
    ground_truth_clips: list[TranscriptClip]
) -> dict:
    """
    Evaluates the quality of the reconstruction using BERTScore.

    It aligns the reconstructed descriptions with the original descriptions
    from the ground truth and calculates the semantic similarity.

    Args:
        reconstructed_clips: The full transcript list returned and parsed from the LLM.
        ground_truth_clips: The original, unmasked transcript.

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
