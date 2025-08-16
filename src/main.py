#!/usr/bin/env python
# PYTHON_ARGCOMPLETE_OK
import argcomplete
from argcomplete.completers import FilesCompleter
import argparse

import logging
import sys

from data_models.exec_args import ExecArgs
from run_experiments import ExperimentPipeline
from utils import UserFacingError


def args_parser():
    parser = argparse.ArgumentParser(description="Command-line argument parser for experiment runner.")
    # Add argument with file completion
    parser.add_argument(
        "config_path",
        type=str,
        help="Path to the experiment configuration file.",
    ).completer = FilesCompleter(allowednames=[".yaml", ".yml"])

    subparsers = parser.add_subparsers(dest="command", required=True)

    # Subparser for '--dry-run'
    dry_run_parser = subparsers.add_parser('dry-run', help="Dry run: Prepare experiments without executing them.")
    dry_run_parser.add_argument('--verbose', action='store_true',
                                help="Display detailed information about experiments.")

    # Subparser for '--validate-cache'
    validate_cache_parser = subparsers.add_parser('validate-cache', help="Validate cached experiment configurations.")

    # Default behavior when no subcommand is mentioned
    run_parser = subparsers.add_parser('run', help="Run experiments and process results.")

    # Add argcomplete support
    argcomplete.autocomplete(parser)

    # Parse arguments
    args = parser.parse_args()

    return ExecArgs.model_validate(vars(args))

def dry_run(xs, verbose=False):
    print(f"prepared {len(xs)} experiments")
    if verbose:
        print()
        for r, conf in xs:
            print(r.run_name, '\t', conf)
        print()

def validate_cache(xs):
    for r, conf in xs:
        r.run()


if __name__ == "__main__":
    exec_args = args_parser()
    ep = ExperimentPipeline(exec_args)

    try:
        if exec_args.command == 'dry-run':
            dry_run(list(ep.build_experiments()), verbose=exec_args.verbose)
        elif exec_args.command == 'validate-cache':
            validate_cache(ep.build_experiments())
        elif exec_args.command == 'run':
            ep.main()
            ep.done()
        else:
            raise ValueError(f"Unknown command: {exec_args.command}")
    except UserFacingError as e:
        print(f"\n❌ Error: {e}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n\n🛑 Experiment batch cancelled by user. Shutting down gracefully.")
        sys.exit(130) # 130 is the standard exit code for Ctrl+C
    except Exception as e:
        logging.error(f"Experiment failed with a critical error: {e}", exc_info=True)
        raise
