#!/usr/bin/env python
# PYTHON_ARGCOMPLETE_OK
from data_models.exec_args import ExecArgs, args_parser
_exec_args:ExecArgs = args_parser() if __name__ == "__main__" else None

import logging
import sys
from common_utils.error_handling import UserFacingError, handle_ctrl_c
from experiment_executor.experiment_runner import ExperimentRunner
from experiment_executor.pipeline import ExperimentPipeline
from experiment_executor.pipeline_executor import Executor


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



def check_remote_status(pipeline: ExperimentPipeline):
    if not pipeline.hf_manager:
        print("\n❌ No Hugging Face repo configured (hf_repo_id missing in system config).")
        return

    print(f"\n📡 Checking remote status for repo: {pipeline.hf_manager.repo_id}")
    
    # We need to know what folders to look for. 
    # The pipeline builds runners, each has a remote path.
    # We can iterate over runners to find unique remote paths.
    
    remote_paths = set()
    for runner in pipeline.build_experiments():
        if hasattr(runner, 'remote_run_path'):
            remote_paths.add(runner.remote_run_path)

    print(f"Found {len(remote_paths)} unique experiment paths in configuration.")
    
    total_files = 0
    for path in sorted(remote_paths):
        print(f"\n📂 Scanning: {path}")
        files = pipeline.hf_manager.list_files(path)
        if files:
            print(f"   ✅ Found {len(files)} files.")
            # Optional: Print first few files?
            # for f in list(files)[:3]:
            #    print(f"      - {f}")
            # if len(files) > 3: print("      ...")
            total_files += len(files)
        else:
            print("   ⚠️  No files found (or path doesn't exist yet).")

    print(f"\nTotal remote files found matching current config: {total_files}")


def main(exec_args:ExecArgs):
    executor = None
    try:
        handle_ctrl_c()
        pipeline = ExperimentPipeline.build(exec_args)
        
        if exec_args.check_remote:
            check_remote_status(pipeline)
            pipeline.shutdown()
            return

        executor = Executor(pipeline)
        if exec_args.dry_run:
            dry_run(*pipeline.dry_run(), verbose=exec_args.verbose)
        elif exec_args.validate_cache:
            validate_cache(pipeline.build_experiments())
        else:
            executor.main()
            executor.done()
    except UserFacingError as e:
        print(f"\n❌ Error: {e}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n\n🛑 Experiment batch cancelled by user. Shutting down gracefully.")
        if executor:
            executor.done()
        sys.exit(130) # 130 is the standard exit code for Ctrl+C
    except Exception as e:
        logging.error(f"Experiment failed with a critical error: {e}", exc_info=True)
        if executor:
            executor.done(e)



if __name__ == "__main__":
    main(_exec_args)