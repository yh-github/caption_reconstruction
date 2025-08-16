#!/usr/bin/env python
# PYTHON_ARGCOMPLETE_OK
import argcomplete
import argparse
from argcomplete.completers import FilesCompleter

from typing import Iterable, Any
from run_experiments import *

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

def dry_run(xs):
    print(f"prepared {len(xs)} experiments")
    if len(sys.argv) > 3 and sys.argv[3] == '--verbose':
        print()
        for r, conf in xs:
            print(r.run_name, '\t', conf)
        print()


def validate_cache(xs:Iterable[tuple[ExperimentRunner, dict[str, Any]]]):
    for r, conf in xs:
        r.run()


if __name__ == "__main__":
    try:
        exec_args = args_parser()
        config = init(exec_args)
        if len(sys.argv) > 2:
            flag=sys.argv[2]
            if flag=='--dry-run':
                dry_run(list(build_experiments(config)))
            elif flag=='--validate-cache':
                validate_cache(build_experiments(config))
        else:
            paths = main(config)
            done(*paths)
    except UserFacingError as e:
        print(f"\n❌ Error: {e}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n\n🛑 Experiment batch cancelled by user. Shutting down gracefully.")
        sys.exit(130) # 130 is the standard exit code for Ctrl+C
    except Exception as e:
        logging.error(f"Experiment failed with a critical error: {e}", exc_info=True)
        raise

