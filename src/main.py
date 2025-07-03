# src/main.py
import os
import yaml
import logging
import mlflow
import git
import platform
import random
import json
from importlib.metadata import version
import google.generativeai as genai
from tenacity import retry, stop_after_attempt, wait_random_exponential

# Local application imports
from data_models import TranscriptClip
from toy_data import create_toy_transcript
from constants import DATA_MISSING
from parsers import parse_llm_response
from evaluation import evaluate_reconstruction, initialize_cache


# =================================================================
# == Basic Logging Setup
# =================================================================
# Configures a logger to print messages to the console.
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# =================================================================
# == Pipeline Functions
# =================================================================

def check_git_repository_is_clean():
    """
    Checks if the Git repository has uncommitted changes.
    Halts execution if the repository is dirty.
    Returns the current commit hash if clean.
    """
    logging.info("Performing Git repository cleanliness check...")
    repo = git.Repo(search_parent_directories=True)
    if repo.is_dirty(untracked_files=True):
        error_message = "Git repository is dirty. Commit or stash changes before running."
        logging.error(error_message)
        raise Exception(error_message)
    logging.info("Git repository is clean.")
    return repo.head.object.hexsha

def load_config(config_path="config/base.yaml"):
    """Loads the YAML configuration file."""
    logging.info(f"Loading configuration from: {config_path}")
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def setup_mlflow(config, git_commit_hash):
    """
    Sets up the MLflow experiment and logs all parameters.
    """
    logging.info("Setting up MLflow and logging parameters...")
    mlflow.set_experiment(config['experiment_name'])

    # Log key parameters for reproducibility
    mlflow.log_param("git_commit_hash", git_commit_hash)
    mlflow.log_param("python_version", platform.python_version())
    mlflow.log_param("mlflow_version", version('mlflow'))
    mlflow.log_param("google_generativeai_version", version('google-generativeai'))

    # Log the entire configuration for full traceability
    mlflow.log_params(config)
    logging.info("Reproducibility parameters logged.")

def load_data(config):
    """Loads the transcript data based on the configuration."""
    source_type = config.get('data', {}).get('source_type', 'toy_example')
    logging.info(f"Loading data from source: {source_type}")

    if source_type == 'toy_example':
        return create_toy_transcript()
    else:
        # This will later handle loading from CSV files, etc.
        raise NotImplementedError(f"Data source type '{source_type}' is not supported yet.")

def apply_masking(transcript: list[TranscriptClip], config: dict) -> list[TranscriptClip]:
    """
    Applies a masking strategy by replacing the entire 'data' payload of a
    clip with the DATA_MISSING token.
    """
    masking_config = config['masking']
    ratio = masking_config['ratio']
    scheme = masking_config['scheme']
    seed = config['random_seed']

    logging.info(f"Applying '{scheme}' masking with ratio {ratio:.2f}...")

    # Initialize the random number generator for reproducible results
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
            # Create a new clip where the 'data' field is our special token
            masked_clip = clip.model_copy()
            masked_clip.data = DATA_MISSING
            masked_transcript.append(masked_clip)
        else:
            masked_transcript.append(clip)

    logging.info(f"Masked {num_to_mask} out of {num_clips} clips.")
    return masked_transcript


def build_prompt(masked_transcript: list[TranscriptClip], config: dict) -> str:
    """
    Builds the final prompt by serializing the list of clips into a
    clean JSON string for the LLM.
    """
    logging.info("Building LLM prompt as JSON...")

    # Convert our list of Pydantic objects to a list of dictionaries
    transcript_for_json = [clip.model_dump() for clip in masked_transcript]

    instruction = (
        "You are an expert video analyst. The following is a JSON list representing a "
        "video transcript. Each object has a 'timestamp' and a 'data' field. "
        f"If the 'data' field contains the token '{DATA_MISSING}', your task is to "
        "replace that token with a valid JSON object containing a plausible 'description'.\n\n"
        "Return the full, corrected JSON list."
    )

    # Serialize the list to a pretty-printed JSON string
    json_prompt_data = json.dumps(transcript_for_json, indent=2)
    final_prompt = f"{instruction}\n\n---\n\n{json_prompt_data}"

    # For debugging, log the full prompt as an MLflow artifact
    mlflow.log_text(final_prompt, "prompt.txt")
    return final_prompt


@retry(
    wait=wait_random_exponential(min=1, max=60), # Wait 1-60 seconds between retries
    stop=stop_after_attempt(5) # Stop after 5 attempts
)
def call_llm(prompt, config):
    """Calls the LLM API, now with retry logic."""
    logging.info("Attempting to call LLM API...")

    # Configure the API key securely
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable not set.")
    genai.configure(api_key=api_key)

    model_name = config['llm']['model_name']
    model = genai.GenerativeModel(model_name)

    response = model.generate_content(prompt)

    llm_response_text = response.text
    mlflow.log_text(llm_response_text, "llm_response.txt")
    logging.info(f"LLM response received. Length: {len(llm_response_text)} characters.")

    return llm_response_text


def evaluate_reconstruction(llm_response, ground_truth_transcript):
    """
    (SIMULATED) Evaluates the quality of the LLM's reconstruction.
    """
    logging.info("Evaluating reconstruction... (SIMULATED)")
    # This will later use BERTScore with a proper embedding cache.
    simulated_metrics = {"bert_score_f1": 0.95, "bert_score_precision": 0.94}
    mlflow.log_metrics(simulated_metrics)
    return simulated_metrics

def main():
    """The main experimental pipeline."""
    logging.info("Starting experiment pipeline...")

    # --- Critical Prerequisite Check ---
    # Fail fast if the repository is not in a clean state.
    git_commit_hash = check_git_repository_is_clean()

    # Now that we've passed the check, start the MLflow run
    with mlflow.start_run():
        try:
            config = load_config()
            cache_path = config.get('paths', {}).get('embedding_cache', 'cache/')
            initialize_cache(cache_path)
            setup_mlflow(config, git_commit_hash)

            ground_truth = load_data(config)

            masked_transcript = apply_masking(ground_truth, config)
            prompt = build_prompt(masked_transcript, config)

            llm_response_text = call_llm(prompt, config)
            parsed_reconstruction = parse_llm_response(llm_response_text)

            if parsed_reconstruction:
                metrics = evaluate_reconstruction(parsed_reconstruction, ground_truth)
                mlflow.log_metrics(metrics) # Log the real metrics

                logging.info("Pipeline finished successfully!")
                logging.info(f"Final Metrics: {metrics}")
            else:
                logging.error("Could not parse LLM response. Halting evaluation.")

            logging.info("Pipeline finished successfully!")
            logging.info(f"Final Metrics: {metrics}")

        except Exception as e:
            logging.error(f"Pipeline failed with an error: {e}", exc_info=True)
            # The 'with mlflow.start_run()' context manager ensures the run is
            # still logged, but its status will be marked as FAILED.
            raise # Re-raise the exception to halt the script

    print("\nRun `mlflow ui` in your terminal to view the full results.")

if __name__ == "__main__":
    main()
