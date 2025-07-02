# src/main.py
import os
import yaml
import logging
import mlflow
import git
import platform
import random
from importlib.metadata import version
import google.generativeai as genai

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

def load_config(config_path="config/base.yaml"):
    """Loads the YAML configuration file."""
    logging.info(f"Loading configuration from: {config_path}")
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def setup_mlflow_and_reproducibility(config):
    """
    Sets up the MLflow experiment, logs parameters, and performs
    reproducibility checks to ensure a clean experimental environment.
    """
    logging.info("Setting up MLflow and reproducibility checks...")
    mlflow.set_experiment(config['experiment_name'])

    repo = git.Repo(search_parent_directories=True)

    # Critical Check: Ensure the repository is clean before running
    if repo.is_dirty(untracked_files=True):
        error_message = "Git repository is dirty. Commit or stash changes before running."
        logging.error(error_message)
        raise Exception(error_message)

    # Log key parameters for reproducibility
    mlflow.log_param("git_commit_hash", repo.head.object.hexsha)
    mlflow.log_param("python_version", platform.python_version())
    mlflow.log_param("mlflow_version", version('mlflow'))
    mlflow.log_param("google_generativeai_version", version('google-generativeai'))

    # Log the entire configuration for full traceability
    mlflow.log_params(config)
    logging.info("Reproducibility checks passed and parameters logged.")

def load_data(config):
    """Loads the transcript data based on the configuration."""
    source_type = config.get('data', {}).get('source_type', 'toy_example')
    logging.info(f"Loading data from source: {source_type}")

    if source_type == 'toy_example':
        return create_toy_transcript()
    else:
        # This will later handle loading from CSV files, etc.
        raise NotImplementedError(f"Data source type '{source_type}' is not supported yet.")

def apply_masking(transcript, config):
    """
    Applies a masking strategy to the transcript based on the config.
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
            # Create a new clip object with a masked description
            masked_clip = clip.model_copy()
            masked_clip.description = "[MASK]"
            masked_transcript.append(masked_clip)
        else:
            masked_transcript.append(clip)
            
    logging.info(f"Masked {num_to_mask} out of {num_clips} clips.")
    return masked_transcript


def build_prompt(masked_transcript, formatter, config):
    """Builds the final prompt string to be sent to the LLM."""
    logging.info("Building LLM prompt...")
    # This creates a list of formatted strings, one for each clip
    formatted_clips = [formatter(clip) for clip in masked_transcript]

    # TODO: Add a more sophisticated instruction based on the task
    instruction = "You are an expert in video analysis. Your task is to fill in the [MASK] tokens in the following video transcript. Provide only the completed descriptions for the masked parts, each on a new line."
    final_prompt = instruction + "\n\n---\n\n" + "\n".join(formatted_clips)

    # For debugging, log the full prompt as an MLflow artifact
    mlflow.log_text(final_prompt, "prompt.txt")
    return final_prompt

def call_llm(prompt, config):
    """Calls the LLM API to get the reconstructed transcript."""
    logging.info("Calling LLM API...")
    
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
    with mlflow.start_run():
        try:
            config = load_config()
            setup_mlflow_and_reproducibility(config)

            ground_truth = load_data(config)
            formatter = get_clip_formatter(config['transcript_richness'])

            masked_transcript = apply_masking(ground_truth, config)
            prompt = build_prompt(masked_transcript, formatter, config)

            llm_response = call_llm(prompt, config)

            metrics = evaluate_reconstruction(llm_response, ground_truth)

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

