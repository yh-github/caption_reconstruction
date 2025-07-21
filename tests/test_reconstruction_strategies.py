import pytest
from unittest.mock import patch

# Import the classes we are going to test
from reconstruction_strategies import BaselineRepeatStrategy, LLMStrategy

# Import the data models we need to create test data
from data_models import CaptionedVideo, CaptionedClip, NarrativeOnlyPayload
from constants import DATA_MISSING

from prompting import JSONPromptBuilder

# --- Test for BaselineStrategy ---

def test_baseline_strategy_reconstruction():
    """
    Tests that the BaselineStrategy correctly fills masked clips by repeating
    the last known valid data payload.
    """
    # Arrange
    # Create a masked video object for the test
    masked_video = CaptionedVideo(
        video_id="test_video",
        clips=[
            CaptionedClip(timestamp=1.0, data=NarrativeOnlyPayload(description="first")),
            CaptionedClip(timestamp=2.0, data=DATA_MISSING),
            CaptionedClip(timestamp=3.0, data=DATA_MISSING),
            CaptionedClip(timestamp=4.0, data=NarrativeOnlyPayload(description="fourth")),
            CaptionedClip(timestamp=5.0, data=DATA_MISSING),
        ]
    )
    
    # Instantiate the strategy with a dummy config
    baseline_strategy = BaselineRepeatStrategy()

    # Act
    reconstructed_video = baseline_strategy.reconstruct(masked_video)
    reconstructed_clips = reconstructed_video.clips

    # Assert
    assert reconstructed_clips[1].data.description == "first"
    assert reconstructed_clips[2].data.description == "first"
    assert reconstructed_clips[4].data.description == "fourth"

def test_baseline_strategy_handles_initial_mask():
    """
    Tests the edge case where the first clip is masked. The baseline should
    correctly back-fill it with the first available valid data.
    """
    # Arrange
    masked_video = CaptionedVideo(
        video_id="test_video_initial_mask",
        clips=[
            CaptionedClip(timestamp=1.0, data=DATA_MISSING),
            CaptionedClip(timestamp=2.0, data=NarrativeOnlyPayload(description="second")),
        ]
    )
    baseline_strategy = BaselineRepeatStrategy()

    # Act
    reconstructed_video = baseline_strategy.reconstruct(masked_video)

    # Assert
    assert reconstructed_video.clips[0].data.description == "second"


# --- Test for LLMStrategy ---

@patch('reconstruction_strategies.call_llm')
@patch('reconstruction_strategies.parse_llm_response')
def test_llm_strategy_initialization_and_call(
    mock_call, mock_parse
):
    """
    Tests that the LLMStrategy is initialized correctly and that its
    reconstruct method calls the necessary helper functions.
    This is a unit test that mocks all external dependencies.
    """
    # Arrange
    config = {
        "name": "test_llm_strategy",
        "model_name": "gemini-test-model"
    }
    # Create a dummy masked video object
    masked_video = CaptionedVideo(video_id="test", clips=[])
    
    # Act
    llm_strategy = LLMStrategy(name='llm_test_strat', llm_model=None, prompt_builder=JSONPromptBuilder.from_string("test prompt"))
    llm_strategy.reconstruct(masked_video)

    # Assert
    
    # 2. Check that the core functions were called once during reconstruction
    mock_call.assert_called_once()
    mock_parse.assert_called_once()
