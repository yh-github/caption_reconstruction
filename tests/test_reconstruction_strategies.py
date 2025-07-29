import pytest
from unittest.mock import patch, MagicMock

from reconstruction_strategies import BaselineRepeatStrategy, LLMStrategy, ReconstructionStrategyBuilder
from data_models import CaptionedVideo, CaptionedClip, NarrativeOnlyPayload, TimestampRange, DATA_MISSING
from exceptions import UserFacingError

# --- Tests for BaselineRepeatStrategy ---

def test_baseline_strategy_reconstruction():
    """
    Tests that the BaselineRepeatStrategy correctly fills masked clips by
    repeating the last known valid data payload.
    """
    # Arrange
    masked_video = CaptionedVideo(
        video_id="test_video",
        clips=[
            CaptionedClip(timestamp=TimestampRange(start=0.0, end=1.0), data=NarrativeOnlyPayload(caption="first")),
            CaptionedClip(timestamp=TimestampRange(start=0.0, end=1.0), data=DATA_MISSING),
            CaptionedClip(timestamp=TimestampRange(start=0.0, end=1.0), data=DATA_MISSING),
            CaptionedClip(timestamp=TimestampRange(start=0.0, end=1.0), data=NarrativeOnlyPayload(caption="fourth")),
            CaptionedClip(timestamp=TimestampRange(start=0.0, end=1.0), data=DATA_MISSING),
        ]
    )
    baseline_strategy = BaselineRepeatStrategy()

    # Act
    r = baseline_strategy.reconstruct(masked_video)

    # Assert
    assert r.reconstructed_clips[1].data.caption == "first"
    assert r.reconstructed_clips[2].data.caption == "first"
    assert r.reconstructed_clips[4].data.caption == "fourth"

def test_baseline_strategy_handles_initial_mask():
    """
    Tests the edge case where the first clip is masked. The baseline should
    correctly back-fill it with the first available valid data.
    """
    # Arrange
    masked_video = CaptionedVideo(
        video_id="test_video_initial_mask",
        clips=[
            CaptionedClip(timestamp=TimestampRange(start=0.0, end=1.0), data=DATA_MISSING),
            CaptionedClip(timestamp=TimestampRange(start=0.0, end=1.0), data=NarrativeOnlyPayload(caption="second")),
        ]
    )
    baseline_strategy = BaselineRepeatStrategy()

    # Act
    r = baseline_strategy.reconstruct(masked_video)

    # Assert
    assert r.reconstructed_clips[0].data.caption == "second"


# --- Test for LLMStrategy ---

@patch('reconstruction_strategies.parse_llm_response')
def test_llm_strategy_reconstruction_flow(mock_parse):
    """
    Tests the orchestration logic of the LLMStrategy's reconstruct method.
    """
    # Arrange
    # Mock the dependencies that are passed into the constructor
    mock_llm_manager = MagicMock()
    mock_prompt_builder = MagicMock()
    
    # Configure the mocks to return specific values
    mock_prompt_builder.build_prompt.return_value = "This is a test prompt."
    mock_llm_manager.call.return_value = "This is a raw response from the LLM."
    
    strategy = LLMStrategy(
        name="test_llm",
        llm_model=mock_llm_manager,
        prompt_builder=mock_prompt_builder
    )
    masked_video = CaptionedVideo(video_id="test", clips=[])

    # Act
    strategy.reconstruct(masked_video)

    # Assert
    # Verify that the internal methods were called in the correct order
    mock_prompt_builder.build_prompt.assert_called_once_with(masked_video)
    mock_llm_manager.call.assert_called_once_with("This is a test prompt.")
    mock_parse.assert_called_once_with("This is a raw response from the LLM.")


# --- Tests for ReconstructionStrategyBuilder ---

@patch('reconstruction_strategies.build_llm_manager')
@patch('reconstruction_strategies.JSONPromptBuilder')
def test_builder_creates_llm_strategy(mock_prompt_builder, mock_build_llm):
    """
    Tests that the builder correctly creates an LLMStrategy.
    """
    # Arrange
    builder = ReconstructionStrategyBuilder(config={})
    strategy_config = {"type": "llm", "name": "test_llm"}

    # Act
    strategy = builder.get_strategy(strategy_config)

    # Assert
    assert isinstance(strategy, LLMStrategy)
    assert strategy.name == "test_llm"
    mock_build_llm.assert_called_once() # Verify the LLM manager was created

def test_builder_creates_baseline_strategy():
    """
    Tests that the builder correctly creates a BaselineRepeatStrategy.
    """
    # Arrange
    builder = ReconstructionStrategyBuilder(config={})
    strategy_config = {"type": "baseline_repeat_last"}

    # Act
    strategy = builder.get_strategy(strategy_config)

    # Assert
    assert isinstance(strategy, BaselineRepeatStrategy)

def test_builder_raises_error_for_unknown_type():
    """
    Tests that the builder raises an error for an unknown strategy type.
    """
    # Arrange
    builder = ReconstructionStrategyBuilder(config={})
    strategy_config = {"type": "unknown_strategy"}

    # Act & Assert
    with pytest.raises(NotImplementedError):
        builder.get_strategy(strategy_config)
