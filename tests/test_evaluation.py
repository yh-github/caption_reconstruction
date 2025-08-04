import pytest
import torch
from unittest.mock import MagicMock

# Import the class and functions we are testing
from evaluation import ReconstructionEvaluator, round_metrics, metrics_to_json

from data_models import (
    CaptionedVideo,
    CaptionedClip,
    TimestampRange
)
from reconstruction_strategies import Reconstructed

# --- Test Fixtures ---

@pytest.fixture
def mock_bert_scorer(mocker):
    """A fixture to mock the BERTScorer object."""
    scorer_instance = MagicMock()
    scorer_instance.score.return_value = (
        torch.tensor([0.9, 0.95]),  # Mock Precision
        torch.tensor([0.8, 0.85]),  # Mock Recall
        torch.tensor([0.85, 0.9]),   # Mock F1
    )
    # Patch the BERTScorer class in the evaluation module
    mocker.patch('evaluation.BERTScorer', return_value=scorer_instance)
    return scorer_instance

@pytest.fixture
def sample_data():
    """Provides sample original and reconstructed data for tests using the new models."""
    original_video = CaptionedVideo(
        video_id="test_vid",
        clips=[
            CaptionedClip(index=0, timestamp=TimestampRange(start=0.0, duration=1.0),caption="clip one original"),
            CaptionedClip(index=1, timestamp=TimestampRange(start=1.0, duration=1.0), caption="clip two original"),
            CaptionedClip(index=2, timestamp=TimestampRange(start=2.0, duration=1.0), caption="clip three original")
        ]
    )
    
    # Create a mock for the Reconstructed object that has a working 'align' method for the test
    reconstructed_data = MagicMock(spec=Reconstructed)
    reconstructed_data.align.return_value = (
        ["clip two recon", "clip three recon"], # candidates
        ["clip two original", "clip three original"]  # references
    )
    
    return original_video, reconstructed_data

# --- Tests ---

def test_reconstruction_evaluator_evaluate_method(mock_bert_scorer, sample_data):
    """
    Tests the main 'evaluate' method to ensure it correctly calls the 
    mocked BERTScorer with the right inputs from the align method.
    """
    # Arrange
    original_video, reconstructed_data = sample_data
    evaluator = ReconstructionEvaluator(model_type="mock-model")

    # Act
    metrics = evaluator.evaluate(reconstructed_data, original_video)

    # Assert
    # 1. Check that the align method was called correctly
    reconstructed_data.align.assert_called_once_with(original_video.clips)
    
    # 2. Check that the score method of our mock was called with the aligned sentences
    mock_bert_scorer.score.assert_called_once()
    call_args, call_kwargs = mock_bert_scorer.score.call_args
    assert call_kwargs['cands'] == ["clip two recon", "clip three recon"]
    assert call_kwargs['refs'] == ["clip two original", "clip three original"]
    
    # 3. Check that the metrics returned are the tensors from our mock
    assert torch.equal(metrics['bs_f1'], torch.tensor([0.85, 0.9]))


def test_round_metrics():
    """
    Tests the 'round_metrics' helper function.
    """
    # Arrange
    raw_metrics = {
        "bs_p": torch.tensor([0.912345]),
        "other_metric": 10
    }
    
    # Act
    rounded_metrics = round_metrics(raw_metrics, ndigits=4)

    # Assert
    assert rounded_metrics['bs_p'] == [0.9123]
    assert rounded_metrics['other_metric'] == 10


def test_metrics_to_json():
    """
    Tests the 'metrics_to_json' helper function.
    """
    # Arrange
    metrics = {"bs_f1": [0.85, 0.9], "num_clips": 2}
    
    # Act
    json_string = metrics_to_json(metrics)

    # Assert
    assert json_string == '{"bs_f1": [0.85, 0.9], "num_clips": 2}'
