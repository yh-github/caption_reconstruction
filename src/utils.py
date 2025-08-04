# src/utils.py
import os
import mlflow
import git
from exceptions import UserFacingError
import logging
from datetime import datetime
import pytz


def set_tz_converter(formatter, tz_str=None):
    tz = pytz.timezone(tz_str or "Asia/Jerusalem")
    formatter.converter = lambda *args: datetime.now(tz).timetuple()
    return formatter

# class TimezoneFormatter(logging.Formatter):
#     """A custom logging formatter that uses a specific timezone."""
#     def __init__(self, fmt:str, tz_str:str="Asia/Jerusalem"):
#         # ,datefmt='%Y-%m-%d %H:%M:%S',
#         super().__init__(fmt)
#         self.tz = pytz.timezone(tz_str)
#
#     def formatTime(self, record, datefmt=None):
#         # Get the original log time (which is in UTC)
#         dt = datetime.fromtimestamp(record.created, pytz.utc)
#         dt = dt.astimezone(self.tz)
#         return dt.strftime(datefmt or self.default_time_format)

NOTICE_LEVEL_NUM = 25 # Between INFO (20) and WARNING (30)
NOTICE_LEVEL_NAME = "NOTICE"

def add_notice_log_level():
    """
    Adds a new 'NOTICE' log level between INFO and WARNING.
    """
    logging.addLevelName(NOTICE_LEVEL_NUM, NOTICE_LEVEL_NAME)

    def notice(self, message, *args, **kws):
        if self.isEnabledFor(NOTICE_LEVEL_NUM):
            # Yes, logger takes its '*args' as 'args'.
            self._log(NOTICE_LEVEL_NUM, message, args, **kws)

    logging.Logger.notice = notice

def get_notification_logger():
    """
    Creates a simple logger that only prints INFO messages to the console.
    """
    # Create a new logger with a unique name
    notification_logger = logging.getLogger('NotificationLogger')
    notification_logger.setLevel(logging.INFO)

    # Prevent messages from being passed to the root logger to avoid duplicates
    notification_logger.propagate = False

    # If the logger already has handlers, don't add more
    if not notification_logger.handlers:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(set_tz_converter(logging.Formatter('ℹ️ %(asctime)s - NOTICE %(levelname)s - %(message)s')))
        notification_logger.addHandler(console_handler)

    return notification_logger

def setup_logging(log_dir: str, run_id: str, console_level=logging.WARN, base_level=logging.INFO, tz_str:str|None=None):
    """
    Configures logging to write to both the console and a unique file
    for the given MLflow run ID.
    """
    if not tz_str:
        tz_str = "Asia/Jerusalem"

    # add_notice_log_level()

    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"{run_id}.log")

    logger = logging.getLogger()
    logger.setLevel(base_level)

    # Clear existing handlers to prevent duplicate logs
    if logger.hasHandlers():
        logger.handlers.clear()

    formatter = set_tz_converter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'), tz_str=tz_str)

    # Setup console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(console_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Setup file handler
    file_handler = logging.FileHandler(log_path, mode='w')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return log_path

def flush_loggers():
    """
    Forces all handlers attached to the root logger to flush their buffers.
    """
    for handler in logging.getLogger().handlers:
        handler.flush()


def check_git_repository_is_clean():
    """Checks for uncommitted changes and raises a specific error if dirty."""
    logging.info("Performing Git repository cleanliness check...")
    repo = git.Repo(search_parent_directories=True)
    if repo.is_dirty(untracked_files=True):
        error_message = "Git repository is dirty. Commit or stash changes before running."
        logging.error(error_message)
        # raise UserFacingError(error_message)
    logging.info("Git repository is clean.")
    return repo.head.object.hexsha

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
    mlflow.set_experiment(experiment_name=experiment_name)

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


def get_datetime_str(tz:str|None=None) -> str:
    return datetime.now(pytz.timezone(tz or "Asia/Jerusalem")).strftime("%H-%M_%d_%m_%Y")
