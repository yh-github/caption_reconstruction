
import sys
from pathlib import Path
import logging
import argparse

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from experiment_executor.pipeline import ExperimentPipeline
from data_models.exec_args import ExecArgs
from common_utils.tracking import setup_logging
from experiment_executor.experiment_runner import ExperimentRunner
from experiment_executor.batch_runner import BatchExperimentRunner
import re

def check_runner(runner, all_videos, verbose=False):
    """Checks status for a single ExperimentRunner."""
    # runner._sync_hf_state() # Skip full sync to avoid metadata download overhead
    if runner.hf_manager:
        runner.remote_files = runner.hf_manager.list_files(runner.remote_run_path)
    runner._save_path.mkdir(parents=True, exist_ok=True)
    
    total = len(all_videos)
    done_local = 0
    done_remote = 0
    pending = 0
    
    for video in all_videos:
        filename = runner._filename(video.video_id)
        skip_filename = f"skip__{filename}"
        local_path = runner._save_path / filename
        local_skip_path = runner._save_path / skip_filename
        
        is_local = local_path.exists() or local_skip_path.exists()
        is_remote = runner.hf_manager and (filename in runner.remote_files or skip_filename in runner.remote_files)
        
        if is_local:
            done_local += 1
        elif is_remote:
            done_remote += 1 # We assume we skip if remote exists (new logic)
        else:
            pending += 1
            if verbose and pending <= 5:
                print(f"  [PENDING] {video.video_id}")
                
    return total, done_local, done_remote, pending

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config_path", type=Path)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    
    # Mock ExecArgs
    exec_args = ExecArgs(
        config_path=args.config_path, 
        dry_run=True, # Prevent heavy init
        no_download_existing=True,
        ignore_unsafe=True,
        verbose=args.verbose
    )
    
    try:
        pipeline = ExperimentPipeline.build(exec_args)
    except Exception as e:
        print(f"Failed to build pipeline: {e}")
        return

    print(f"--- Checking Status for {pipeline.config.get('__parent_run_name__')} ---")
    
    # Load data once
    if pipeline.experiment_type == 'RECON':
        all_videos = pipeline.data_loader.load()
    else:
        print("Only RECON supported for this check script.")
        return

    print(f"Total Videos in Dataset: {len(all_videos)}")
    
    total_experiments = 0
    total_pending = 0
    
    # We need to manually initialize HF manager if pipeline didn't (because of dry_run=True in logic?)
    # Pipeline.__init__ sets hf_manager based on config, but dry_run might skip prefetch (which is good).
    # But checking internal state: hf_manager should be initialized.
    
    pending_details = set()
    pattern = re.compile(r"w=(\d+),\s*i=(\d+)")

    for runner in pipeline.build_experiments():
        if isinstance(runner, BatchExperimentRunner):
            print(f"\nBatch Runner: {runner.run_name} ({len(runner.runners)} configs)")
            for sub_runner in runner.runners:
                t, l, r, p = check_runner(sub_runner, all_videos, args.verbose)
                print(f"  Config {sub_runner.run_name}: {l} Local, {r} Remote, {p} Pending")
                total_experiments += t
                total_pending += p
                if p > 0:
                    match = pattern.search(sub_runner.run_name)
                    if match:
                        pending_details.add((int(match.group(1)), int(match.group(2))))

        elif isinstance(runner, ExperimentRunner):
            t, l, r, p = check_runner(runner, all_videos, args.verbose)
            print(f"\nRunner {runner.run_name}: {l} Local, {r} Remote, {p} Pending")
            total_experiments += t
            total_pending += p
            if p > 0:
                match = pattern.search(runner.run_name)
                if match:
                    pending_details.add((int(match.group(1)), int(match.group(2))))

    print("\n" + "="*30)
    print("Pending Parameters (width, start_index):")
    for w, i in sorted(list(pending_details)):
        print(f"  Width={w}, Index={i}")
            
    print("\n" + "="*30)
    print(f"Total Configs Checked: {total_experiments}")
    print(f"Total Pending Computations: {total_pending}")
    print("="*30)

    # regex to capture w and i
    # expected format: ...fixed_fill(w=9, i=29)...
    pattern = re.compile(r"w=(\d+),\s*i=(\d+)")
    
    pending_params = set()

    print("\n--- Pending Parameter details ---")
    
    # We need to reiterate or store state. Let's just do it in the loop above?
    # Refactoring main to store runners with pending > 0
    
    # Actually, let's modify the loop in main slightly to accomplish this without full rewrite
    pass

if __name__ == "__main__":
    main()
