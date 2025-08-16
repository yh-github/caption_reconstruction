#!/usr/bin/env python
# PYTHON_ARGCOMPLETE_OK
import argcomplete
from argcomplete.completers import FilesCompleter
import argparse

import logging
import sys
from pathlib import Path

from data_models.exec_args import ExecArgs
from run_experiments import ExperimentPipeline
from utils import UserFacingError


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

def dry_run(xs, count:int, verbose=False):
    print(f"prepared {len(xs)} experiments, with {count} videos. Total runs = {len(xs)*count}")
    if verbose:
        print()
        for r, conf in xs:
            print(r.run_name, '\t', conf)
        print()

def validate_cache(xs):
    logging.getLogger().setLevel(logging.WARNING)
    for r, conf in xs:
        r.run()


if __name__ == "__main__":
    exec_args = args_parser()
    ep = ExperimentPipeline(exec_args)

    try:
        if exec_args.dry_run:
            dry_run(list(ep.build_experiments()), ep.data_loader.count(), verbose=exec_args.verbose)
        elif exec_args.validate_cache:
            validate_cache(ep.build_experiments())
        else: # Run experiments
            ep.main()
            ep.done()
    except UserFacingError as e:
        print(f"\n❌ Error: {e}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n\n🛑 Experiment batch cancelled by user. Shutting down gracefully.")
        sys.exit(130) # 130 is the standard exit code for Ctrl+C
    except Exception as e:
        logging.error(f"Experiment failed with a critical error: {e}", exc_info=True)
        raise
