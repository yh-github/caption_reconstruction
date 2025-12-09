import argparse
import logging
from pathlib import Path

import argcomplete
from argcomplete import FilesCompleter
from pydantic import BaseModel, Field


DEFAULT_SYSTEM_CONFIG_PATH = Path("config/system.yaml")


class ExecArgs(BaseModel):
    config_path: Path
    system_config_path: Path = DEFAULT_SYSTEM_CONFIG_PATH
    override: list[str] = Field(default_factory=list, description="raw override key-value pairs")
    verbose: bool = False
    dry_run: bool = False
    validate_cache: bool = False
    debug: bool = False
    ignore_unsafe: bool = False

    def should_ignore_unsafe(self) -> bool:
        return self.debug or self.ignore_unsafe

    def log_level(self, log_level:int) -> int:
        return log_level if not self.debug else logging.DEBUG

def args_parser() -> ExecArgs:
    parser = argparse.ArgumentParser(description="Command-line argument parser for experiment runner.")

    parser.add_argument(
        "config_path",
        type=Path,
        help="Path to the experiment configuration file.",
    ).completer = FilesCompleter(allowednames=[".yaml", ".yml"])

    parser.add_argument(
        "--system_config_path",
        type=Path,
        help="Path to the system configuration file.",
    ).completer = FilesCompleter(allowednames=[".yaml", ".yml"])

    parser.add_argument(
        "--override",
        type=str,
        metavar="KEY=VALUE",
        nargs='+',
        help="Config key value pairs to override.",
    )


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

    # Filter out None values to let Pydantic handle defaults
    args_dict = {k: v for k, v in vars(args).items() if v is not None}

    return ExecArgs.model_validate(args_dict)


import sys
def get_dargs() -> dict[int, str]:
    return dict(enumerate(sys.argv[1:], start=1))
