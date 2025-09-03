from unittest.mock import MagicMock
from data_models.captions_only import CaptionedVideo, CaptionedClip
from reconstruction_strategies import Reconstructed
from reconstruction_strategies import ReconstructionStrategy, BaselineRepeatStrategy, LLMStrategy


def test_create_error_result():
    video_id = "1234"
    error_message = "Test error"
    extra_debug_data = {"debug_key": "debug_value"}

    result = ReconstructionStrategy.create_error_result(
        video_id=video_id,
        error_message=error_message,
        extra_debug_data=extra_debug_data
    )

    assert result.video_id == video_id
    assert result.reconstructed_captions == {}
    assert result.debug_data["error"] == error_message
    assert result.debug_data["debug_key"] == "debug_value"
    assert result.skip_reason == "error"


from data_models.captions_only import TimestampRange


def test_baseline_repeat_strategy():
    strategy = BaselineRepeatStrategy()

    masked_video = CaptionedVideo(
        video_id="test_video_1",
        clips=[
            CaptionedClip(index=0, timestamp=TimestampRange(start=0.0, duration=1.0), caption="first"),
            CaptionedClip(index=1, timestamp=TimestampRange(start=1.0, duration=1.0), caption=None),
            CaptionedClip(index=2, timestamp=TimestampRange(start=2.0, duration=1.0), caption=None),
            CaptionedClip(index=3, timestamp=TimestampRange(start=3.0, duration=1.0), caption="fourth"),
            CaptionedClip(index=4, timestamp=TimestampRange(start=4.0, duration=1.0), caption=None),
        ]
    )

    reconstructed = strategy.reconstruct(masked_video)

    assert isinstance(reconstructed, Reconstructed)
    assert reconstructed.video_id == "test_video_1"
    assert reconstructed.reconstructed_captions == {
        1: "first",
        2: "first",
        4: "fourth"
    }


# def test_llm_strategy_successful_response():
#     mock_llm_model = MagicMock()
#     mock_prompt_builder = MagicMock()
#
#     # Mock LLM response
#     mock_llm_model.call.return_value.text = '{"1": "Caption A", "2": "Caption B"}'
#     mock_prompt_builder.build_prompt.return_value = "Mock Prompt"
#
#     strategy = LLMStrategy("TestLLM", mock_llm_model, mock_prompt_builder)
#
#     masked_video = CaptionedVideo(
#         video_id="test_video_2",
#         clips=[
#             CaptionedClip(index=0, caption="Original Caption", timestamp=TimestampRange(start=0.0, duration=1.0)),
#             CaptionedClip(index=1, caption=None, timestamp=TimestampRange(start=1.0, duration=1.0)),  # Masked
#             CaptionedClip(index=2, caption=None, timestamp=TimestampRange(start=2.0, duration=1.0))  # Masked
#         ]
#     )
#
#     reconstructed = strategy.reconstruct(masked_video)
#
#     assert isinstance(reconstructed, Reconstructed)
#     assert reconstructed.video_id == "test_video_2"
#     assert reconstructed.reconstructed_captions == {
#         1: "Caption A",
#         2: "Caption B"
#     }


def test_llm_strategy_empty_response():
    mock_llm_model = MagicMock()
    mock_prompt_builder = MagicMock()

    # Mock LLM response
    mock_llm_model.call.return_value.text = None
    mock_prompt_builder.build_prompt.return_value = "Mock Prompt"

    strategy = LLMStrategy("TestLLM", mock_llm_model, mock_prompt_builder)

    masked_video = CaptionedVideo(
        video_id="test_video_3",
        clips=[
            CaptionedClip(index=0, caption="Test Caption", timestamp=TimestampRange(start=0.0, duration=1.0)),
            CaptionedClip(index=1, caption=None, timestamp=TimestampRange(start=1.0, duration=1.0))
        ]
    )

    reconstructed = strategy.reconstruct(masked_video)

    assert isinstance(reconstructed, Reconstructed)
    assert reconstructed.skip_reason == "error"
    assert "LLM error - llm_response_text empty" in reconstructed.debug_data["error"]

# from unittest.mock import MagicMock
# from reconstruction_strategies import ReconstructionStrategyBuilder, LLMStrategy, BaselineRepeatStrategy

# def test_reconstruction_strategy_builder():
#     mock_llm_manager_builder = MagicMock()
#     mock_llm_manager_builder.from_config.return_value = "MockLLMClient"
#
#     builder = ReconstructionStrategyBuilder(llm_cache=None, master_seed=123, llm_client=mock_llm_manager_builder)
#
#     # Test LLM strategy creation
#     llm_strategy_config = {
#         "type": "llm",
#         "name": "Test LLM Strategy",
#         "llm": {"model_name": "test_llm_config"}
#     }
#     llm_strategy = builder.get_strategy(llm_strategy_config)
#     assert isinstance(llm_strategy, LLMStrategy)
#
#     # Test baseline strategy creation
#     baseline_strategy_config = {
#         "type": "baseline_repeat_last",
#         "name": "Baseline Strategy"
#     }
#     baseline_strategy = builder.get_strategy(baseline_strategy_config)
#     assert isinstance(baseline_strategy, BaselineRepeatStrategy)