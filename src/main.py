import sys
import os
import logging
import mlflow
import git
import platform
from importlib.metadata import version
from filelock import FileLock, Timeout

# Local application imports
from config_loader import load_config
from llm_interaction import initialize_llm
from pipeline import run_experiment
from evaluation import initialize_cache
from exceptions import UserFacingError

def setup_logging(run_id: str):
    """
    Configures logging to write to both the console and a unique file
    for the given MLflow run ID.
    """
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"{run_id}.log")

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Clear existing handlers to prevent duplicate logs
    if logger.hasHandlers():
        logger.handlers.clear()

    # Setup console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(console_handler)

    # Setup file handler
    file_handler = logging.FileHandler(log_path, mode='w')
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)

    return log_path


def check_git_repository_is_clean():
    logging.info("Performing Git repository cleanliness check...")
    repo = git.Repo(search_parent_directories=True)
    if repo.is_dirty(untracked_files=True):
        error_message = "Git repository is dirty. Commit or stash changes before running."
        logging.error(error_message)
        raise UserFacingError(error_message)
    logging.info("Git repository is clean.")
    return repo.head.object.hexsha

def setup_mlflow(config, git_commit_hash):
    logging.info("Setting up MLflow and logging parameters...")
    mlflow.set_tracking_uri(config['paths']['mlflow_tracking_uri'])
    mlflow.set_experiment(config['experiment_name'])
    mlflow.log_param("git_commit_hash", git_commit_hash)
    mlflow.log_param("python_version", platform.python_version())
    mlflow.log_param("mlflow_version", version('mlflow'))
    mlflow.log_params(config)
    logging.info("Reproducibility parameters logged.")

def main():
    """
    Main entry point for running a single experiment.
    Orchestrates initialization, setup, and execution.
    """
    # Initial console-only logging for pre-checks
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    lock_filename = ".lock"
    lock = FileLock(lock_filename)
    
    try:
        logging.info(f"Waiting for exclusive resource lock (filelock '{lock_filename}') by PID {os.getpid()}...")
        with lock:
            logging.info(f"Lock acquired by PID {os.getpid()}. Starting experiment.")

            git_commit_hash = check_git_repository_is_clean()
            config = load_config("config/base.yaml")
            initialize_cache(config.get('paths', {}).get('joblib_cache', 'cache/'))
            llm_model = initialize_llm(config)

            with mlflow.start_run() as run:
                # Get the unique run ID from MLflow
                run_id = run.info.run_id
                
                # Set up proper logging for this run
                log_file_path = setup_logging(run_id)
                
                logging.info(f"--- Starting New Experiment Run ---")
                logging.info(f"MLflow Run ID: {run_id}")

                setup_mlflow(config, git_commit_hash)
                run_experiment(config, llm_model)
                
                # Log the full run log as an MLflow artifact
                mlflow.log_artifact(log_file_path)

    except UserFacingError as e:
        # no stack trace, clear actionable message.
        print(f"\n❌ Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        logging.error(f"Experiment failed with a critical error: {e}", exc_info=True)
        raise

    logging.info(f'PID {os.getpid()} DONE.')
    print("\n✅ Finished successfully.")
    print("\nRun `mlflow ui` in your terminal to view the full results.")

if __name__ == "__main__":
    main()
