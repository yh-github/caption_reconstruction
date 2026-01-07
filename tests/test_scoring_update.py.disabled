
import pytest
from unittest.mock import MagicMock, patch
import json
from pathlib import Path
import sys

# Add src to path if needed (it usually is in current pytest env)
# sys.path.append("src")

from experiment_executor.score_dataset import main
from data_models.captions_only import CaptionedVideo, CaptionedClip, TimestampRange
from data.hf_sync import HFResultsSync

@pytest.fixture
def mock_dependencies():
    with patch("experiment_executor.score_dataset.get_data_loader") as mock_loader, \
         patch("experiment_executor.score_dataset.PriorSurpriseScorer") as mock_prior, \
         patch("experiment_executor.score_dataset.PMIScorer") as mock_pmi, \
         patch("experiment_executor.score_dataset.HFResultsSync") as mock_sync_cls, \
         patch("experiment_executor.score_dataset.config_from_args") as mock_config, \
         patch("experiment_executor.score_dataset.parse_scoring_args") as mock_args, \
         patch("experiment_executor.score_dataset.ExecArgs") as mock_exec_args, \
         patch("torch.cuda.is_available", return_value=True):
         
        # Setup Data
        video = CaptionedVideo(
            video_id="test_vid_1",
            clips=[
                CaptionedClip(index=0, timestamp=TimestampRange(start=0, duration=5), caption="Hello world"),
                CaptionedClip(index=1, timestamp=TimestampRange(start=5, duration=5), caption="Segment to mask")
            ]
        )
        mock_loader.return_value.load.return_value = [video]
        
        # Setup Config
        mock_config.return_value = {
            "scoring_model_key": "test_model",
            "data_config": {},
            "masking_configs": [{"scheme": "fixed_fill", "start_ind": [1], "width": [1]}],
            "base_params": {"master_seed": 42, "run_name": "test_run"}
        }
        
        # Setup Scorers
        prior_instance = mock_prior.return_value
        prior_instance.calculate_whole_log_surprisal.return_value = [] # Return simple empty for now or mock data
        
        pmi_instance = mock_pmi.return_value
        pmi_instance.calculate_informativeness_batch.return_value = [{"pmi_score": 0.5}]
        
        # Setup Sync (In-Memory State)
        sync_instance = mock_sync_cls.return_value
        
        # We need a stateful mock for sync to simulate persistence between runs
        class StatefulSyncMock:
            def __init__(self):
                self.data = {}
            
            def pull(self, force_download=False):
                return self.data
            
            def push(self, data, commit_message=""):
                self.data = data
                
            def merge_results(self, existing, new, config):
                merged = existing.copy()
                if "scores" not in merged: merged["scores"] = {}
                for k, v in new.items():
                    merged["scores"][k] = v
                return merged

        stateful_sync = StatefulSyncMock()
        mock_sync_cls.return_value = stateful_sync
        
        yield {
            "loader": mock_loader,
            "args": mock_args,
            "sync": stateful_sync,
            "pmi_mock": pmi_instance
        }


def test_scoring_update_flow(mock_dependencies):
    """
    Test 1: Run WITHOUT PMI.
    Test 2: Run WITH PMI.
    Expectation: Video should be re-processed in Test 2.
    Current Behavior (Suspected): Video skipped in Test 2.
    """
    deps = mock_dependencies
    
    # --- RUN 1: No PMI ---
    deps["args"].return_value = MagicMock(
        calc_pmi=False, 
        score_all=False, 
        ignore_gpu=True, # Allow test to run anywhere
        hf_repo_id="test/repo",
        upload_interval=10
    )
    
    print("\n--- Running Step 1 (No PMI) ---")
    main()
    
    # Check Result 1
    data_step_1 = deps["sync"].data
    assert "test_vid_1" in data_step_1["scores"]
    assert data_step_1["scores"]["test_vid_1"]["segments_pmi"] == []
    print("Step 1 Success: Video processed (Surprisal only)")
    
    # --- RUN 2: With PMI ---
    deps["args"].return_value.calc_pmi = True
    
    print("\n--- Running Step 2 (With PMI) ---")
    main()
    
    # Check Result 2
    data_step_2 = deps["sync"].data
    video_data = data_step_2["scores"]["test_vid_1"]
    
    # If the logic is FLAWED, segments_pmi will still be empty because it skipped
    # If the logic is FIXED, segments_pmi should contain the mocked [{"pmi_score": 0.5}]
    
    pmi_scores = video_data.get("segments_pmi", [])
    print(f"PMI Scores after Step 2: {pmi_scores}")
    
    if not pmi_scores:
        pytest.fail("Regression: Video was skipped in Step 2 despite adding --calc-pmi flag! It did not update.")
    
    assert len(pmi_scores) > 0
    assert pmi_scores[0]["pmi_score"] == 0.5
