import logging
from abc import ABC, abstractmethod
from data_models import DATA_MISSING, CaptionedClip
from data_models import CaptionedVideo
from llm_interaction import LLM_Manager, build_llm_manager
from parsers import parse_llm_response
from prompting import BasePromptBuilder, JSONPromptBuilder
from exceptions import UserFacingError
from pydantic import BaseModel

class Reconstructed(BaseModel):
    video_id: str
    reconstructed_clips: dict[int, CaptionedClip]
    debug_data: dict|None = None
    skip_reason: str|None = None
    metrics: dict|None = None

    def align(self, orig_clips: list[CaptionedClip]) -> tuple[list[str], list[str]]:
        """
        Helper method to extract reference and candidate sentences.
        """
        references = []
        candidates = []

        for i, c in self.reconstructed_clips.items():
            candidates.append(c.data.caption)
            references.append(orig_clips[i].data.caption)

        return candidates, references

    def skip(self, reason: str):
        self.skip_reason = reason
        return self

    def with_metrics(self, metrics: dict):
        self.metrics = metrics
        return self

    def json_str(self):
        return self.model_dump_json(exclude_none=True)


class ReconstructionStrategy(ABC):
    """An abstract base class for all reconstruction methods."""
    def __init__(self, name: str):
        self.name = name

    def __repr__(self):
        return self.name

    @abstractmethod
    def reconstruct(self, masked_video: CaptionedVideo) -> Reconstructed | None:
        """Takes a masked CaptionedVideo and returns a reconstructed one."""
        pass

class BaselineRepeatStrategy(ReconstructionStrategy):
    """The strategy for using the 'repeat last known' baseline."""
    def __init__(self):
        super().__init__('BaselineRepeatStrategy')

    def reconstruct(self, masked_video: CaptionedVideo) -> Reconstructed | None:
        """
        Fills masked clips by repeating the data from the last known clip.
        If initial clips are masked, it back-fills them with the first valid data.
        """
        masked_clips = masked_video.clips

        # First Pass: Find the first available data payload
        first_valid_data = None
        for clip in masked_clips:
            if clip.data != DATA_MISSING:
                first_valid_data = clip.data
                break

        # Second Pass: Reconstruct the transcript
        reconstructed_clips = {}
        last_known_data = first_valid_data

        for i, clip in enumerate(masked_clips):
            # new_clip = clip.model_copy()
            if clip.data != DATA_MISSING:
                last_known_data = clip.data
                # new_clip.data = clip.data
            else:
                # Fill the masked clip with the last known data
                new_clip = clip.model_copy(update={'data': last_known_data})
                # new_clip.data = last_known_data
                reconstructed_clips[i]=new_clip

        # return masked_video.model_copy(update={'clips': reconstructed_clips})
        return Reconstructed(video_id=masked_video.video_id, reconstructed_clips=reconstructed_clips)


class LLMStrategy(ReconstructionStrategy):
    """The strategy for using an LLM for reconstruction."""
    def __init__(self, name: str, llm_model, prompt_builder: BasePromptBuilder):
        super().__init__(name)
        self.llm_model = llm_model
        self.prompt_builder = prompt_builder

    def reconstruct(self, masked_video: CaptionedVideo) -> Reconstructed | None:
        try:
            prompt = self.prompt_builder.build_prompt(masked_video)
            llm_response_text = self.llm_model.call(prompt)
            reconstructed_clips = parse_llm_response(llm_response_text)
            assert reconstructed_clips and len(reconstructed_clips)==len(masked_video.clips)
            ok = []
            failed = []
            changed_unmasked = []
            reconstructed_dict = {}
            for i, c in enumerate(masked_video.clips):
                if c.data == DATA_MISSING:
                    if hasattr(reconstructed_clips[i], 'data') and \
                        hasattr(reconstructed_clips[i].data, 'caption') \
                            and reconstructed_clips[i].data.caption\
                            and reconstructed_clips[i].data.caption != DATA_MISSING:
                        ok.append(i)
                        reconstructed_dict[i] = reconstructed_clips[i]
                    else:
                        failed.append(i)
                else:
                    if c != reconstructed_clips[i]:
                        changed_unmasked.append(i)
            # reconstructed_video = masked_video.model_copy(update={'clips': reconstructed_clips})
            debug_data=None
            if failed or changed_unmasked:
                debug_data = {
                    "ok": ok,
                    "failed": failed,
                    "changed_unmasked": changed_unmasked,
                    "llm_response_text": llm_response_text
                }
            return Reconstructed(
                video_id=masked_video.video_id,
                reconstructed_clips=reconstructed_dict,
                debug_data=debug_data
            )
        except Exception as e:
            logging.error(f"{e} for {masked_video.video_id=}")
            return None

class ReconstructionStrategyBuilder:
    """
    A builder class responsible for creating reconstruction strategy objects.
    It initializes and holds the LLM client, ensuring it's created only once.
    """
    def __init__(self, config: dict):
        self.llm_model: LLM_Manager|None = None
        self.config = config

    def get_strategy(self, strategy_config: dict) -> ReconstructionStrategy:
        """
        Builds and returns a specific strategy instance based on the config.
        """
        strategy_type = strategy_config.get("type")
        if not strategy_type:
            raise UserFacingError("'type' must be specified in the strategy configuration.")

        if strategy_type == "llm":
            if self.llm_model is None:
                self.llm_model = build_llm_manager(self.config)
            prompt_builder = JSONPromptBuilder.from_config(strategy_config)
            return LLMStrategy(
                name=strategy_config.get("name"),
                llm_model=self.llm_model,
                prompt_builder=prompt_builder
            )

        elif strategy_type == "baseline_repeat_last":
            return BaselineRepeatStrategy()

        else:
            raise NotImplementedError(f"Strategy type '{strategy_type}' is not implemented.")
