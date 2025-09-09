#!/usr/bin/env python
# PYTHON_ARGCOMPLETE_OK
from data_models.exec_args import ExecArgs, args_parser
_exec_args:ExecArgs = args_parser() if __name__ == "__main__" else None

import logging
import sys
from run_experiments import ExperimentPipeline
from utils import UserFacingError
from experiment_runner import ExperimentRunner

def dry_run(xs:list[ExperimentRunner], count:int, verbose=False):
    print(f"prepared {len(xs)} experiments, with {count} videos. Total runs = {len(xs)*count}")
    if verbose:
        print()
        for r in xs:
            print(r.run_name, '\t', r.conf_for_log)
        print()

def validate_cache(xs):
    logging.getLogger().setLevel(logging.WARNING)
    for r in xs:
        r.run()


def main(exec_args:ExecArgs):
    ep = None
    try:
        ep = ExperimentPipeline.build(exec_args)
        if exec_args.dry_run:
            dry_run(*ep.dry_run(), verbose=exec_args.verbose)
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
        if ep:
            ep.done()
        sys.exit(130) # 130 is the standard exit code for Ctrl+C
    except Exception as e:
        logging.error(f"Experiment failed with a critical error: {e}", exc_info=True)
        if ep:
            ep.done(e)



if __name__ == "__main__":
    main(_exec_args)