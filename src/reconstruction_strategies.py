import logging
from abc import ABC, abstractmethod
from data_models import CaptionedClip
from data_models import CaptionedVideo
from llm_interaction import build_llm_manager, init_llm
from parsers import parse_llm_response
from prompting import PromptBuilder, PromptBuilderIndexedData
from exceptions import UserFacingError
from pydantic import BaseModel

class Reconstructed(BaseModel):
    video_id: str
    reconstructed_captions: dict[int, str]
    debug_data: dict|None = None
    skip_reason: str|None = None
    metrics: dict|None = None

    def align(self, orig_clips: list[CaptionedClip]) -> tuple[list[str], list[str]]:
        """
        Helper method to extract reference and candidate sentences.
        """
        references = []
        candidates = []

        for i, c in self.reconstructed_captions.items():
            assert i == orig_clips[i].index
            candidates.append(c)
            references.append(orig_clips[i].caption)

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
        first_valid_caption = None
        for clip in masked_clips:
            if clip.caption is not None:
                first_valid_caption = clip.caption
                break

        # Second Pass: Reconstruct the captions
        reconstructed_captions = {}
        last_known_caption = first_valid_caption

        for clip in masked_clips:
            if clip.caption is not None:
                last_known_caption = clip.caption
            else:
                reconstructed_captions[clip.index]=last_known_caption
        try:
            return Reconstructed(video_id=masked_video.video_id, reconstructed_captions=reconstructed_captions)
        except Exception:
            logging.error(f"{masked_video=} {reconstructed_captions=}")
            raise


class LLMStrategy(ReconstructionStrategy):
    """The strategy for using an LLM for reconstruction."""
    def __init__(self, name: str, llm_model, prompt_builder: PromptBuilder):
        super().__init__(name)
        self.llm_model = llm_model
        self.prompt_builder: PromptBuilder = prompt_builder

    def reconstruct(self, masked_video: CaptionedVideo) -> Reconstructed:
        debug_data = None
        try:
            prompt = self.prompt_builder.build_prompt(masked_video)
            logging.debug(f"video_id={masked_video.video_id} {prompt=}")
            llm_response_text = self.llm_model.call(prompt)
            if not llm_response_text:
                return Reconstructed(video_id=masked_video.video_id, reconstructed_captions={}, debug_data={
                    "error": "LLM error - llm_response_text empty",
                    "raw_response": self.llm_model.last_raw_response
                })
            reconstructed_video = parse_llm_response(llm_response_text)
            if not reconstructed_video:
                return Reconstructed(video_id=masked_video.video_id, reconstructed_captions={}, debug_data={
                    "error": "LLM error - failed parsing",
                    "raw_response": self.llm_model.last_raw_response
                })

            recon_caps, dups = reconstructed_video.to_dict()

            if not recon_caps:
                return Reconstructed(video_id=masked_video.video_id, reconstructed_captions={}, debug_data={
                    "error": "LLM error - failed parsing to_dict",
                    "raw_response": self.llm_model.last_raw_response
                })

            if dups:
                return Reconstructed(video_id=masked_video.video_id, reconstructed_captions={}, debug_data={
                    "error": "LLM error - duplicate indices found",
                    "llm_response_text": llm_response_text,
                    "dups": dups
                })



            ok = []
            failed = []
            changed_unmasked = []
            reconstructed_dict:dict[int, str] = {}
            for c in masked_video.clips:
                if c.caption is None:
                    if new_cap := recon_caps.get(c.index):
                        ok.append(c.index)
                        reconstructed_dict[c.index] = new_cap
                    else:
                        failed.append(c.index)
                        reconstructed_dict[c.index] = "" #TODO check if needed, check BertScore is 0

                # original not masked but index is reconstructed
                elif c.index in recon_caps and c.caption != recon_caps.get(c.index):
                    changed_unmasked.append(c.index)

            if failed or changed_unmasked:
                debug_data = {
                    "ok": ok,
                    "failed": failed,
                    "changed_unmasked": changed_unmasked,
                    "llm_response_text": llm_response_text
                }
            return Reconstructed(
                video_id=masked_video.video_id,
                reconstructed_captions=reconstructed_dict,
                debug_data=debug_data
            )
        except Exception as e:
            logging.error(f"{e} for {masked_video.video_id=}")
            if not debug_data:
                debug_data = {}
            debug_data.update({
                "error": str(e),
                "raw_response": self.llm_model.last_raw_response
            })
            return Reconstructed(
                video_id=masked_video.video_id,
                reconstructed_captions={},
                debug_data=debug_data
            )


class ReconstructionStrategyBuilder:
    """
    A builder class responsible for creating reconstruction strategy objects.
    """
    def __init__(self, llm_cache):
        self.init_llm_api = False
        self.llm_cache = llm_cache

    def get_strategy(self, strategy_config: dict) -> ReconstructionStrategy:
        """
        Builds and returns a specific strategy instance based on the config.
        """
        strategy_type = strategy_config.get("type")
        if not strategy_type:
            raise UserFacingError("'type' must be specified in the strategy configuration.")

        if strategy_type == "llm":
            if not self.init_llm_api:
                init_llm()
                self.init_llm_api = True
            return LLMStrategy(
                name=strategy_config["name"],
                llm_model=build_llm_manager(strategy_config['llm'], self.llm_cache),
                prompt_builder=PromptBuilderIndexedData()
            )

        elif strategy_type == "baseline_repeat_last":
            return BaselineRepeatStrategy()

        else:
            raise NotImplementedError(f"Strategy type '{strategy_type}' is not implemented.")
