import pytest
from unittest.mock import Mock, patch, mock_open
from pathlib import Path
import json
import numpy as np

from experiment_executor.experiment_runner import ExperimentRunner
from data.data_loaders import BaseDataLoader
from data_models.captions_only import CaptionedVideo, CaptionedClip, TimestampRange
from evaluations.evaluation import ReconstructionEvaluator
from reconstruction.masking import MaskingStrategy
from reconstruction.text_reconstruction import ReconstructionStrategy, Reconstructed

@pytest.fixture
def runner_mocks():
    mock_loader = Mock(spec=BaseDataLoader)
    mock_loader.get_data_type_name.return_value = "mock_data"
    
    video = CaptionedVideo(
        video_id="vid1", 
        clips=[CaptionedClip(index=0, timestamp=TimestampRange(start=0, duration=1), caption="hello")]
    )
    mock_loader.load.return_value = [video]
    
    mock_masking = Mock(spec=MaskingStrategy)
    mock_recon = Mock(spec=ReconstructionStrategy)
    mock_eval = Mock(spec=ReconstructionEvaluator)
    
    runner = ExperimentRunner(
        run_name="test_run",
        data_loader=mock_loader,
        masking_strategy=mock_masking,
        reconstruction_strategy=mock_recon,
        evaluator=mock_eval,
        save_path=Path("/tmp/mock_results"),
        conf_for_log={}
    )
    
    return runner, mock_loader, mock_masking, mock_recon, mock_eval, video

def test_process_single_video_resumption(runner_mocks):
    """
    Verify that if a result file exists:
    1. It is loaded.
    2. Masking strategy IS CALLED (to advance PRN).
    3. Reconstruction strategy IS NOT CALLED.
    """
    runner, _, mock_masking, mock_recon, _, video = runner_mocks
    
    # We mock valid existing result content
    # In reality, the file contains JSON (lists), but the code uses Reconstructed.model_validate_json
    # which we will mock to return an object that already has what the Runner needs.
    # The runner creates MetricsRecordRaw(raw_metrics=reconstructed.metrics).
    # MetricsRecordRaw expects dict[str, Any] (relaxed recently) or NPArray.
    # Since we relaxed MetricsRecordRaw to accept Any, passing lists (from JSON) works fine.
    # The failure might have been due to previous strict typing. 
    # Let's ensure the mock returns a Reconstructed object with a metrics dict.

    mock_recon_obj = Reconstructed(
        video_id="vid1",
        reconstructed_captions={0: "hello"},
        metrics={"score": 1.0} # Valid scalar
    )

    with patch('experiment_executor.experiment_runner.Reconstructed.model_validate_json', return_value=mock_recon_obj):
        with patch.object(Path, 'exists', return_value=True), \
             patch('builtins.open', mock_open(read_data="{}")):
            
            mock_masking.mask_video.return_value = (video, {0})
            
            result = runner._process_single_video(video)
            
            assert result is not None
            assert result.raw_metrics["score"] == 1.0
            
            mock_masking.mask_video.assert_called_once_with(video)
            mock_recon.reconstruct.assert_not_called()


def test_resumption_skips_if_masking_returns_none(runner_mocks):
    """Verify handling where masking fails during resumption (e.g. config mismatch)."""
    runner, _, mock_masking, _, _, video = runner_mocks
    
    mock_recon_obj = Reconstructed(
        video_id="vid1",
        reconstructed_captions={},
        metrics={}
    )
    
    with patch('experiment_executor.experiment_runner.Reconstructed.model_validate_json', return_value=mock_recon_obj):
        with patch.object(Path, 'exists', return_value=True), \
             patch('builtins.open', mock_open(read_data="{}")):
            
            # Masking returns None (e.g. couldn't mask)
            mock_masking.mask_video.return_value = (None, None)
            
            result = runner._process_single_video(video)
            
            assert result is None
            mock_masking.mask_video.assert_called_once()


def test_process_single_video_new_run(runner_mocks):
    """Verify normal flow for new video."""
    runner, _, mock_masking, mock_recon, mock_eval, video = runner_mocks

    with patch.object(Path, 'exists', return_value=False):
        mock_masked_video = video.model_copy()
        mock_masking.mask_video.return_value = (mock_masked_video, {0})
        
        mock_reconstructed = Reconstructed(video_id="vid1", reconstructed_captions={0: "hello"}, metrics=None) # Start null
        mock_recon.reconstruct.return_value = mock_reconstructed
        
        # Evaluate returns valid metrics
        mock_eval.evaluate.return_value = {"score": 0.9}
        
        with patch.object(runner, '_save_result') as mock_save:
            result = runner._process_single_video(video)
            
            assert result is not None
            assert result.raw_metrics["score"] == 0.9
            
            mock_masking.mask_video.assert_called_once()
            mock_recon.reconstruct.assert_called_once()
            mock_eval.evaluate.assert_called_once()
            mock_save.assert_called_once()
