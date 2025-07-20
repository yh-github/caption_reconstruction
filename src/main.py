# src/main.py
import sys
import os
import logging
import mlflow
from filelock import FileLock, Timeout

# Local application imports
from config_loader import load_config
from llm_interaction import initialize_llm
from pipeline import run_experiment
from evaluation import initialize_cache
from exceptions import UserFacingError
from utils import setup_logging, check_git_repository_is_clean, setup_mlflow

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
            config = load_config("config/base.yaml") # This will need to be made dynamic later
            initialize_cache(config.get('paths', {}).get('joblib_cache', 'cache/'))
            llm_model = initialize_llm(config)

            with mlflow.start_run() as run:
                run_id = run.info.run_id
                log_file_path = setup_logging(run_id)
                
                logging.info(f"--- Starting New Experiment Run ---")
                logging.info(f"MLflow Run ID: {run_id}")

                setup_mlflow(config, git_commit_hash)
                run_experiment(config, llm_model)
                
                mlflow.log_artifact(log_file_path)

    except UserFacingError as e:
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
