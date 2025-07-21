from abc import ABC, abstractmethod
import mlflow

from data_models import CaptionedClip, CaptionedVideo
from llm_interaction import call_llm
from parsers import parse_llm_response
from baselines import repeat_last_known_baseline

from prompting import BasePromptBuilder
from constants import DATA_MISSING

class ReconstructionStrategy(ABC):
    """An abstract base class for all reconstruction methods."""
    def __init__(self, name: str):
        self.name = name

    def __repr__(self):
        return self.name

    @abstractmethod
    def reconstruct(self, masked_video: CaptionedVideo) -> CaptionedVideo | None:
        """Takes a masked CaptionedVideo and returns a reconstructed one."""
        pass

class BaselineRepeatStrategy(ReconstructionStrategy):
    """The strategy for using the 'repeat last known' baseline."""
    def __init__(self):
        super().__init__('BaselineRepeatStrategy')

    def reconstruct(self, masked_video: CaptionedVideo) -> CaptionedVideo | None:
        """
        Fills masked clips by repeating the data from the last known clip.
        If initial clips are masked, it back-fills them with the first valid data.
        """
        masked_clips = masked_video.clips
        if not masked_clips:
            return masked_video

        # First Pass: Find the first available data payload
        first_valid_data = None
        for clip in masked_clips:
            if clip.data != DATA_MISSING:
                first_valid_data = clip.data
                break

        # Second Pass: Reconstruct the transcript
        reconstructed_clips = []
        last_known_data = first_valid_data

        for clip in masked_clips:
            new_clip = clip.model_copy()
            if clip.data != DATA_MISSING:
                last_known_data = clip.data
                new_clip.data = clip.data
            else:
                # Fill the masked clip with the last known data
                new_clip.data = last_known_data
            reconstructed_clips.append(new_clip)

        return masked_video.model_copy(update={'clips': reconstructed_clips})

class LLMStrategy(ReconstructionStrategy):
    """The strategy for using an LLM for reconstruction."""
    def __init__(self, name: str, llm_model, prompt_builder: BasePromptBuilder):
        super().__init__(name)
        self.llm_model = llm_model
        self.prompt_builder = prompt_builder # Injected dependency

    def reconstruct(self, masked_video: CaptionedVideo) -> CaptionedVideo | None:
        try:
            prompt = self.prompt_builder.build_prompt(masked_video)
            llm_response_text = call_llm(self.llm_model, prompt)
            return parse_llm_response(llm_response_text)
        except Exception as e:
            logging.error(f"{e} for {masked_video.video_id=}")
            return None


def build_reconstruction_strategy(strategy_config: dict) -> ReconstructionStrategy:
    """
    Factory function that reads the strategy-specific config and builds
    the correct reconstruction strategy object.
    """
    strategy_type = strategy_config.get("type")
    if not strategy_type:
        raise ValueError("'type' must be specified in the strategy configuration.")

    if strategy_type == "llm":
        return LLMStrategy(config=strategy_config)
    elif strategy_type == "baseline_repeat_last":
        return BaselineStrategy(config=strategy_config)
    # Add other strategies here in the future
    else:
        raise NotImplementedError(f"Strategy type '{strategy_type}' is not implemented.")


class ReconstructionStrategyBuilder:
    """
    A builder class responsible for creating reconstruction strategy objects.
    It initializes and holds the LLM client, ensuring it's created only once.
    """
    def __init__(self, config: dict):
        self.llm_model = None
        self.config = config

    def get_strategy(self, strategy_config: dict) -> ReconstructionStrategy:
        """
        Builds and returns a specific strategy instance based on the config.
        """
        strategy_type = strategy_config.get("type")
        if not strategy_type:
            raise ConfigError("'type' must be specified in the strategy configuration.")

        if strategy_type == "llm":
            if self.llm_model is None:
                self.llm_model = initialize_llm(self.config)
            prompt_builder = get_prompt_builder(strategy_config)
            # Inject the pre-initialized LLM model into the strategy
            return LLMStrategy(
                name=strategy_config.get("name"),
                llm_model=self.llm_model,
                prompt_builder=prompt_builder
            )

        elif strategy_type == "baseline_repeat_last":
            return BaselineRepeatStrategy()

        else:
            raise NotImplementedError(f"Strategy type '{strategy_type}' is not implemented.")
