# src/utils.py
import os
import logging
import mlflow
import git
import platform
from importlib.metadata import version
from exceptions import UserFacingError

def setup_logging(log_dir: str, run_id: str):
    """
    Configures logging to write to both the console and a unique file
    for the given MLflow run ID.
    """
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

def setup_mlflowOLD(config, git_commit_hash):
    """Sets up MLflow experiment and logs all parameters."""
    logging.info("Setting up MLflow and logging parameters...")
    mlflow.set_tracking_uri(config['paths']['mlflow_tracking_uri'])
    mlflow.set_experiment(config['experiment_name'])
    mlflow.log_param("git_commit_hash", git_commit_hash)
    mlflow.log_param("python_version", platform.python_version())
    mlflow.log_param("mlflow_version", version('mlflow'))
    mlflow.log_params(config)
    logging.info("Reproducibility parameters logged.")


def setup_mlflow(
    experiment_name: str,
    tracking_uri: str
):
    """
    Sets up the MLflow experiment and logs all specified parameters.
    All dependencies are now explicit arguments.
    """
    logging.info("Setting up MLflow and logging parameters...")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

def object_to_dict(obj: object) -> dict:
    """
    Recursively converts an object and its attributes into a dictionary
    that is safe for logging as MLflow parameters.
    """
    if not hasattr(obj, '__dict__'):
        return {"type": obj.__class__.__name__}

    # Start with the object's class name
    param_dict = {"type": obj.__class__.__name__}

    for key, value in vars(obj).items():
        # If the attribute is another custom object, recurse
        if hasattr(value, '__dict__'):
            param_dict[key] = object_to_dict(value)
        # Only include simple, loggable types
        elif isinstance(value, (str, int, float, bool)):
            param_dict[key] = value

    return param_dict
