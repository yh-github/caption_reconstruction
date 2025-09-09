import argparse
import logging
from pathlib import Path

import argcomplete
from argcomplete import FilesCompleter
from pydantic import BaseModel

class ExecArgs(BaseModel):
    config_path: Path
    verbose: bool = False
    dry_run: bool = False
    validate_cache: bool = False
    debug: bool = False

    def log_level(self, log_level:int) -> int:
        return log_level if not self.debug else logging.DEBUG

def args_parser():
    parser = argparse.ArgumentParser(description="Command-line argument parser for experiment runner.")

    parser.add_argument(
        "config_path",
        type=Path,
        help="Path to the experiment configuration file.",
    ).completer = FilesCompleter(allowednames=[".yaml", ".yml"])

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Display detailed information about experiments."
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Display debug information."
    )

    parser.add_argument(
        "--dry-run",
        action="store_true"
    )

    parser.add_argument(
        "--validate-cache",
        action="store_true"
    )

    # Add argcomplete support
    argcomplete.autocomplete(parser)

    # Parse arguments
    args = parser.parse_args()

    return ExecArgs.model_validate(vars(args))
