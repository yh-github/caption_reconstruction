import argparse
from pathlib import Path

import argcomplete
from argcomplete import FilesCompleter
from pydantic import BaseModel

class ExecArgs(BaseModel):
    config_path: Path
    verbose: bool
    dry_run: bool
    validate_cache: bool

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
