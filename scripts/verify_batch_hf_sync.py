import logging
import sys
from pathlib import Path
from unittest.mock import MagicMock
from experiment_executor.batch_runner import BatchExperimentRunner
from experiment_executor.experiment_runner import ExperimentRunner
from data.hf_sync import HFFileManager
from reconstruction.text_reconstruction import BatchGridSearchStrategy
from common_utils.tracking import setup_logging

def verify_batch_hf_sync():
    # Setup logging
    setup_logging(log_dir="logs", run_id="verify_batch_hf", console_level=logging.INFO)
    
    repo_id = "Y3/dense_video_captions"
    logging.info(f"Verifying HF Sync Logic for repo: {repo_id}")

    # 1. Initialize Real HF Manager (Read Only to be safe at first)
    # We want to verified LIST and DOWNLOAD capabilities
    hf_manager = HFFileManager(repo_id=repo_id, read_only=True)
    
    # 2. Mock Components
    data_loader = MagicMock()
    data_loader.get_data_type_name.return_value = "mock_data"
    
    masking_strategy = MagicMock()
    # Return a dummy mask so it doesn't bail out early on masking
    masking_strategy.mask_video.return_value = (MagicMock(), {0}) 
    
    evaluator = MagicMock()
    batch_strategy = MagicMock()
    batch_strategy.reconstruct.return_value = [] # Return empty list just to bypass execution loop

    # 3. Create Mock Experiment Runners
    # We will simulate:
    # - Run A: Exists on HF (we pick a file we know exists or list one first)
    # - Run B: Does NOT exist on HF
    
    # First, let's list files to find a real existing file to test download logic
    logging.info("Listing files from remote to pick a target...")
    remote_files = hf_manager.api.list_repo_files(repo_id=repo_id, repo_type="dataset")
    
    # Pick a JSON file if available
    target_file = next((f for f in remote_files if f.endswith(".json") and "reconstruction" in f), None)
    
    if not target_file:
        logging.warning("No existing JSON files found in repo to test download. Skipping download verification.")
        existing_run_name = "existing_run"
        video_id_existing = "vid_123"
    else:
        logging.info(f"Found target file: {target_file}")
        # Parse structure: reconstruction/STEM/RUN_NAME/VIDEO_ID.json
        parts = target_file.split('/')
        # e.g. reconstruction/default/phi-3__masking/video_1.json
        run_name = parts[-2]
        video_id_existing = parts[-1].replace(".json", "")
        existing_run_name = run_name
        
        logging.info(f"Will verify download for Run: {run_name}, Video: {video_id_existing}")

    # Runner A (Should find remote match)
    runner_a = MagicMock(spec=ExperimentRunner)
    runner_a.run_name = existing_run_name
    runner_a.remote_run_path = str(Path(target_file).parent) if target_file else "reconstruction/mock/existing"
    runner_a._filename.side_effect = lambda vid: f"{vid}.json"
    runner_a._save_path = Path("results/verify_batch") / existing_run_name
    runner_a.hf_manager = hf_manager
    runner_a.remote_files = set() # Empty initially, sync should fill strictly if we call sync,
                                  # BUT BatchRunner calls _sync_hf_state explicitly.
    
    # We need to implement _sync_hf_state on the mock or wrap a real runner
    # Let's verify the logic in BatchRunner._process_single_video_batch
    # It accesses runner.remote_files directly.
    # So we need to emulate what _sync_hf_state DOES: populating remote_files.
    
    # Manually populate for simulation
    if target_file:
        runner_a.remote_files = {f"{video_id_existing}.json"}
    
    # Runner B (Should NOT find match)
    runner_b = MagicMock(spec=ExperimentRunner)
    runner_b.run_name = "non_existent_run"
    runner_b.remote_run_path = "reconstruction/mock/missing"
    runner_b._filename.side_effect = lambda vid: f"{vid}.json"
    runner_b._save_path = Path("results/verify_batch") / "non_existent_run"
    runner_b.hf_manager = hf_manager
    runner_b.remote_files = set()

    batch_runner = BatchExperimentRunner(
        base_run_name="batch_verify",
        runners=[runner_a, runner_b],
        batch_strategy=batch_strategy,
        data_loader=data_loader,
        masking_strategy=masking_strategy,
        evaluator=evaluator
    )

    # 4. Run Logic Verification
    
    # CASE 1: Video present in A (remote) but not B (anywhere)
    video = MagicMock()
    video.video_id = video_id_existing
    video.clips = []
    
    logging.info(f"--- Testing Video {video.video_id} ---")
    
    # We want to capture what indices BatchRunner decides to run.
    # But _process_single_video_batch is internal. We can check the calls to batch_strategy.reconstruct
    # batch_strategy.reconstruct(..., active_indices=?)
    
    batch_runner._process_single_video_batch(video)
    
    # Check calls
    args, kwargs = batch_strategy.reconstruct.call_args
    active_indices = kwargs.get('active_indices')
    logging.info(f"Active Indices for {video.video_id}: {active_indices}")
    
    # Expectations:
    # Runner A: Exists remote -> Should download -> Should NOT be in active_indices
    # Runner B: Does not exist -> Should be in active_indices
    
    # Since we didn't actually downloading file (mock runner), it might fail download and fall back to calculate?
    # BatchRunner logic:
    # if runner.hf_manager.download_file(...): continue
    # else: active_indices.append(i)
    
    # We are using REAL hf_manager with read_only=True.
    # download_file works in read_only mode.
    # So if file exists, it should return True and download it.
    
    if target_file:
        expected_indices = [1] # Only Runner B
        if 0 in active_indices:
             logging.error("❌ FAILURE: Runner A was scheduled for run despite existing remotely!")
        else:
             logging.info("✅ SUCCESS: Runner A detected remote file and skipped calculation.")
             
        if 1 not in active_indices:
             logging.error("❌ FAILURE: Runner B was NOT scheduled for run!")
        else:
             logging.info("✅ SUCCESS: Runner B scheduled for calculation.")
             
    else:
        logging.warning("Skipping assertion for A due to no remote file found.")

    logging.info("Verification Complete.")

if __name__ == "__main__":
    verify_batch_hf_sync()
