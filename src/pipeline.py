import logging
import json
import mlflow
import random

# Local application imports
from data_models import TranscriptClip
from toy_data import create_toy_transcript
from constants import DATA_MISSING
from llm_interaction import call_llm
from parsers import parse_llm_response
from evaluation import evaluate_reconstruction

def load_data(config):
    source_type = config.get('data', {}).get('source_type', 'toy_example')
    logging.info(f"Loading data from source: {source_type}")
    if source_type == 'toy_example':
        return create_toy_transcript()
    else:
        raise NotImplementedError(f"Data source type '{source_type}' is not supported yet.")

def apply_masking(transcript: list[TranscriptClip], config: dict) -> list[TranscriptClip]:
    """Applies a masking strategy to the transcript based on the config."""
    masking_config = config['masking']
    scheme = masking_config['scheme']
    seed = config.get('random_seed', None)

    if seed:
        random.seed(seed)

    num_clips = len(transcript)
    indices_to_mask = set()

    if scheme == 'random':
        ratio = masking_config['ratio']
        num_to_mask = int(num_clips * ratio)
        indices_to_mask = set(random.sample(range(num_clips), k=num_to_mask))

    elif scheme == 'contiguous_random':
        ratio = masking_config['ratio']
        num_to_mask = int(num_clips * ratio)
        start_index = random.randint(0, num_clips - num_to_mask)
        indices_to_mask = set(range(start_index, start_index + num_to_mask))

    elif scheme == 'contiguous_controlled':
        start_index = masking_config.get('mask_index')
        num_to_mask = masking_config.get('num_to_mask')
        if start_index is None or num_to_mask is None:
            raise ValueError("'mask_index' and 'num_to_mask' must be provided for contiguous_controlled scheme.")
        indices_to_mask = set(range(start_index, start_index + num_to_mask))

    elif scheme == 'systematic':
        # Mask every Nth item. Calculate N to get the desired ratio.
        num_to_mask = masking_config.get('num_to_mask')
        if num_to_mask == 0: return transcript # Avoid division by zero
        step = num_clips // num_to_mask
        # Start at a random offset to avoid always masking the same first elements
        start_offset = random.randint(0, step - 1)
        indices_to_mask = set(range(start_offset, num_clips, step))

    else:
        raise NotImplementedError(f"Masking scheme '{scheme}' is not implemented yet.")

    masked_transcript = []
    for i, clip in enumerate(transcript):
        if i in indices_to_mask:
            masked_clip = clip.model_copy()
            masked_clip.data = DATA_MISSING
            masked_transcript.append(masked_clip)
        else:
            masked_transcript.append(clip)

    logging.info(f"Masked {len(indices_to_mask)} out of {num_clips} clips using '{scheme}' scheme.")
    return masked_transcript

def build_prompt(masked_transcript: list[TranscriptClip], config: dict) -> str:
    """Builds the final JSON prompt to be sent to the LLM."""
    logging.info("Building LLM prompt as JSON...")
    transcript_for_json = [clip.model_dump() for clip in masked_transcript]
    json_prompt_data = json.dumps(transcript_for_json, indent=2)
    
    instruction = (
        "You are an expert video analyst. Reconstruct the full data object for any "
        f"timestamp where the 'data' field is the token '{DATA_MISSING}'. "
        "Return the complete JSON list with all masks filled."
    )
    
    final_prompt = f"{instruction}\n\n---\n\n{json_prompt_data}"
    
    # Check if MLflow is active before logging, which is safer for tests
    if mlflow.active_run():
        mlflow.log_text(final_prompt, "prompt.txt")
        
    return final_prompt

