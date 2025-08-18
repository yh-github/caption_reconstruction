from unittest.mock import patch, MagicMock

import pytest
from data_models.captions_only import CaptionedVideo, CaptionedClip, TimestampRange
from reconstruction_strategies import BaselineRepeatStrategy, LLMStrategy, ReconstructionStrategyBuilder


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
            CaptionedClip(index=0, timestamp=TimestampRange(start=0.0, duration=1.0), caption="first"),
            CaptionedClip(index=1, timestamp=TimestampRange(start=1.0, duration=1.0), caption=None),
            CaptionedClip(index=2, timestamp=TimestampRange(start=2.0, duration=1.0), caption=None),
            CaptionedClip(index=3, timestamp=TimestampRange(start=3.0, duration=1.0), caption="fourth"),
            CaptionedClip(index=4, timestamp=TimestampRange(start=4.0, duration=1.0), caption=None),
        ]
    )
    baseline_strategy = BaselineRepeatStrategy()

    # Act
    r = baseline_strategy.reconstruct(masked_video)

    # Assert
    assert r.reconstructed_captions[1] == "first"
    assert r.reconstructed_captions[2] == "first"
    assert r.reconstructed_captions[4] == "fourth"

def test_baseline_strategy_handles_initial_mask():
    """
    Tests the edge case where the first clip is masked. The baseline should
    correctly back-fill it with the first available valid data.
    """
    # Arrange
    masked_video = CaptionedVideo(
        video_id="test_video_initial_mask",
        clips=[
            CaptionedClip(index=0, timestamp=TimestampRange(start=0.0, duration=1.0), caption=None),
            CaptionedClip(index=1, timestamp=TimestampRange(start=1.0, duration=1.0), caption="second")
        ]
    )
    baseline_strategy = BaselineRepeatStrategy()

    # Act
    r = baseline_strategy.reconstruct(masked_video)

    # Assert
    assert r.reconstructed_captions[0] == "second"

# --- Tests for ReconstructionStrategyBuilder ---

# @patch('reconstruction_strategies.build_llm_manager')
# @patch('reconstruction_strategies.JSONPromptBuilder.from_config')
# def test_builder_creates_llm_strategy(mock_prompt_builder, mock_build_llm):
#     """
#     Tests that the builder correctly creates an LLMStrategy.
#     """
#     # Arrange
#     builder = ReconstructionStrategyBuilder(None,666)
#     strategy_config = {"type": "llm", "name": "test_llm", "llm": {}}
#
#     # Act
#     strategy = builder.get_strategy(strategy_config)
#
#     # Assert
#     assert isinstance(strategy, LLMStrategy)
#     assert strategy.name == "test_llm"
#     mock_prompt_builder.assert_called_once()
#     mock_build_llm.assert_called_once() # Verify the LLM manager was created


def test_get_llm_response():
    """
    Tests the helper method '_get_llm_response' of LLMStrategy with a dummy prompt.
    """
    # Arrange
    mock_llm_model = MagicMock()
    mock_prompt_builder = MagicMock()
    mock_prompt_builder.build_prompt.return_value = "Test prompt"
    mock_llm_model.call.return_value.text = "LLM response text"
    strategy = LLMStrategy(name="test_llm", llm_model=mock_llm_model, prompt_builder=mock_prompt_builder)
    masked_video = MagicMock()

    # Act
    llm_response = strategy._get_llm_response(masked_video)

    # Assert
    mock_prompt_builder.build_prompt.assert_called_once_with(masked_video)
    mock_llm_model.call.assert_called_once_with("Test prompt")
    assert llm_response == "LLM response text"


# def test_parse_and_validate_response():
#     """
#     Tests the '_parse_and_validate_response' helper method with a valid LLM response.
#     """
#     # Arrange
#     mock_llm_response = '{"0": "first caption", "1": "second caption"}'
#     # noinspection PyTypeChecker
#     strategy = LLMStrategy(name="test_llm", llm_model=None, prompt_builder=None)
#
#     # Mock the parser
#     with patch('reconstruction_strategies.parse_llm_response') as mock_parse:
#         mock_parse.return_value.to_dict.return_value = (
#             {0: "first caption", 1: "second caption"},
#             {}
#         )
#
#         # Act
#         parsed_response, dups = strategy._parse_and_validate_response(mock_llm_response)
#
#         # Assert
#         mock_parse.assert_called_once_with(mock_llm_response)
#         assert parsed_response == {0: "first caption", 1: "second caption"}
#         assert dups == {}


def test_builder_creates_baseline_strategy():
    """
    Tests that the builder correctly creates a BaselineRepeatStrategy.
    """
    # Arrange
    builder = ReconstructionStrategyBuilder(None, 666, None)
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
    builder = ReconstructionStrategyBuilder(None, 666, None)
    strategy_config = {"type": "unknown_strategy"}

    # Act & Assert
    with pytest.raises(NotImplementedError):
        builder.get_strategy(strategy_config)
