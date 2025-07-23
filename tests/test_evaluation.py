import pytest
from unittest.mock import MagicMock

# Assuming your new class is in 'evaluation' and data models are in 'data_models'
from evaluation import ReconstructionEvaluator
from data_models import CaptionedClip, NarrativeOnlyPayload

# Helper mock class to simulate the tensor returned by bert_score
class MockTensor:
    def __init__(self, value):
        self._value = value

    def mean(self):
        return self
    
    def item(self):
        return self._value

# --- Test Setup: Fixture ---
@pytest.fixture
def mock_video_data():
    """Provides mock data using your Pydantic data models for the tests."""
    ground_truth = [
        CaptionedClip(timestamp=1.0, data=NarrativeOnlyPayload(description="a dog is running")),
        CaptionedClip(timestamp=2.0, data=NarrativeOnlyPayload(description="a cat is sleeping")),
        CaptionedClip(timestamp=3.0, data=NarrativeOnlyPayload(description="a bird is flying")),
        CaptionedClip(timestamp=4.0, data=NarrativeOnlyPayload(description="a fish is swimming")),
    ]
    
    reconstructed = [
        CaptionedClip(timestamp=1.0, data=NarrativeOnlyPayload(description="a dog is running")), # Not masked
        CaptionedClip(timestamp=2.0, data=NarrativeOnlyPayload(description="a feline rests")),   # Masked
        CaptionedClip(timestamp=3.0, data=NarrativeOnlyPayload(description="an avian soars")),   # Masked
        CaptionedClip(timestamp=4.0, data=NarrativeOnlyPayload(description="a fish is swimming")), # Not masked
    ]

    masked_indices = {1, 2}

    return ground_truth, reconstructed, masked_indices

# --- Tests ---

def test_evaluate_correctly_aligns_clips_and_calls_bertscore(mocker, mock_video_data):
    """
    Tests that only the clips corresponding to the masked_indices are passed to bert_score.
    """
    # Arrange
    gt_clips, recon_clips, masked_indices = mock_video_data
    
    # Mock the bert_score function so we don't do a real API call
    mock_bert_score = mocker.patch('evaluation.bert_score_func', return_value=(
        MockTensor(0.95), MockTensor(0.85), MockTensor(0.90)
    ))
    
    evaluator = ReconstructionEvaluator()

    # Act
    metrics = evaluator.evaluate(recon_clips, gt_clips, masked_indices)

    # Assert
    # 1. Check that bert_score was called exactly once
    mock_bert_score.assert_called_once()
    
    # 2. Extract the arguments it was called with
    _, kwargs = mock_bert_score.call_args
    
    # 3. Verify that only the masked descriptions were passed
    expected_candidates = ["a feline rests", "an avian soars"]
    expected_references = ["a cat is sleeping", "a bird is flying"]
    
    assert kwargs['cands'] == expected_candidates
    assert kwargs['refs'] == expected_references
    
    # 4. Verify the returned metrics match the mock's output
    assert metrics["bert_score_precision"] == 0.95
    assert metrics["bert_score_recall"] == 0.85
    assert metrics["bert_score_f1"] == 0.90

def test_evaluate_handles_no_masked_clips(mocker, mock_video_data):
    """
    Tests that if no clips are masked, the function returns 0 and doesn't call bert_score.
    """
    # Arrange
    gt_clips, recon_clips, _ = mock_video_data
    empty_masked_indices = set()
    
    mock_bert_score = mocker.patch('evaluation.bert_score_func')
    
    evaluator = ReconstructionEvaluator()

    # Act
    metrics = evaluator.evaluate(recon_clips, gt_clips, empty_masked_indices)

    # Assert
    # 1. Check that bert_score was NEVER called
    mock_bert_score.assert_not_called()
    
    # 2. Check that the metrics are all zero
    assert metrics["bert_score_precision"] == 0.0
    assert metrics["bert_score_recall"] == 0.0
    assert metrics["bert_score_f1"] == 0.0

def test_evaluator_uses_init_parameters(mocker, mock_video_data):
    """
    Tests that the model_type and idf parameters from the constructor are correctly passed to bert_score.
    """
    # Arrange
    gt_clips, recon_clips, masked_indices = mock_video_data
    
    mock_bert_score = mocker.patch('evaluation.bert_score_func', return_value=(
        MockTensor(0.1), MockTensor(0.1), MockTensor(0.1)
    ))
    
    # Initialize with custom, non-default parameters
    custom_model = "test-model/custom"
    custom_idf = False
    evaluator = ReconstructionEvaluator(model_type=custom_model, idf=custom_idf)

    # Act
    evaluator.evaluate(recon_clips, gt_clips, masked_indices)

    # Assert
    # Check that bert_score was called with the custom parameters
    _, kwargs = mock_bert_score.call_args
    assert kwargs['model_type'] == custom_model
    assert kwargs['idf'] == custom_idf
