# src/utils.py
import os
import logging
import mlflow
import git
import platform
from importlib.metadata import version
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
    """Checks for uncommitted changes and raises a specific error if dirty."""
    logging.info("Performing Git repository cleanliness check...")
    repo = git.Repo(search_parent_directories=True)
    if repo.is_dirty(untracked_files=True):
        error_message = "Git repository is dirty. Commit or stash changes before running."
        logging.error(error_message)
        raise UserFacingError(error_message)
    logging.info("Git repository is clean.")
    return repo.head.object.hexsha

def setup_mlflow(config, git_commit_hash):
    """Sets up MLflow experiment and logs all parameters."""
    logging.info("Setting up MLflow and logging parameters...")
    mlflow.set_tracking_uri(config['paths']['mlflow_tracking_uri'])
    mlflow.set_experiment(config['experiment_name'])
    mlflow.log_param("git_commit_hash", git_commit_hash)
    mlflow.log_param("python_version", platform.python_version())
    mlflow.log_param("mlflow_version", version('mlflow'))
    mlflow.log_params(config)
    logging.info("Reproducibility parameters logged.")
