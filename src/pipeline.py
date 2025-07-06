# src/pipeline.py
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
    # ... (function is the same)
    source_type = config.get('data', {}).get('source_type', 'toy_example')
    logging.info(f"Loading data from source: {source_type}")
    if source_type == 'toy_example':
        return create_toy_transcript()
    else:
        raise NotImplementedError(f"Data source type '{source_type}' is not supported yet.")

def apply_masking(transcript: list[TranscriptClip], config: dict) -> list[TranscriptClip]:
    # ... (function is the same)
    masking_config = config['masking']
    ratio = masking_config['ratio']
    scheme = masking_config['scheme']
    seed = config['random_seed']
    logging.info(f"Applying '{scheme}' masking with ratio {ratio:.2f}...")
    random.seed(seed)
    num_clips = len(transcript)
    num_to_mask = int(num_clips * ratio)
    if num_to_mask == 0:
        logging.warning("Masking ratio is too low, no clips will be masked.")
        return transcript
    if scheme == 'random':
        indices_to_mask = sorted(random.sample(range(num_clips), k=num_to_mask))
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
    logging.info(f"Masked {num_to_mask} out of {num_clips} clips.")
    return masked_transcript


# Corrected function signature and logic
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

def run_experiment(config, llm_model):
    # ... (function is the same)
    ground_truth = load_data(config)
    masked_transcript = apply_masking(ground_truth, config)
    # Pass the config to build_prompt
    prompt = build_prompt(masked_transcript, config)
    
    llm_response_text = call_llm(llm_model, prompt)
    
    # Check if MLflow is active before logging
    if mlflow.active_run():
        mlflow.log_text(llm_response_text, "llm_response.txt")
    
    parsed_reconstruction = parse_llm_response(llm_response_text)
    
    if parsed_reconstruction:
        metrics = evaluate_reconstruction(parsed_reconstruction, ground_truth)
        if mlflow.active_run():
            mlflow.log_metrics(metrics)
        logging.info("Pipeline finished successfully!")
        logging.info(f"Final Metrics: {metrics}")
    else:
        logging.error("Could not parse LLM response. Halting evaluation.")
        if mlflow.active_run():
            mlflow.log_metric("parsing_failed", 1)
