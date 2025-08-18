import logging
from abc import ABC, abstractmethod
from typing import Any

from google import genai
from pydantic import BaseModel

from data_models.captions_only import CaptionedClip
from data_models.captions_only import CaptionedVideo
from llm_interaction import LLM_Manager_Builder
from parsers import parse_llm_response
from prompting import PromptBuilder, JSONPromptBuilder
from utils import UserFacingError


class Reconstructed(BaseModel):
    video_id: str
    reconstructed_captions: dict[int, str]
    debug_data: dict[str, Any]|None = None
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

    # Error message constants
    EMPTY_RESPONSE_ERROR = "LLM error - llm_response_text empty"
    PARSING_ERROR = "LLM error - failed parsing"
    TO_DICT_ERROR = "LLM error - failed parsing to_dict"
    DUPLICATE_INDICES_ERROR = "LLM error - duplicate indices found"

    def __init__(self, name: str, llm_model, prompt_builder: PromptBuilder):
        super().__init__(name)
        self.llm_model = llm_model
        self.prompt_builder: PromptBuilder = prompt_builder

    def reconstruct(self, masked_video: CaptionedVideo) -> Reconstructed:
        try:
            llm_response_text = self._get_llm_response(masked_video)
            if not llm_response_text:
                return self._create_error_result(masked_video.video_id, self.EMPTY_RESPONSE_ERROR)

            recon_caps, dups = self._parse_and_validate_response(llm_response_text)
            if not recon_caps:
                return self._create_error_result(masked_video.video_id, self.PARSING_ERROR)

            if not recon_caps:
                return self._create_error_result(masked_video.video_id, self.TO_DICT_ERROR)

            if dups:
                return self._create_error_result(
                    masked_video.video_id,
                    self.DUPLICATE_INDICES_ERROR,
                    {"llm_response_text": llm_response_text, "dups": dups}
                )

            return self._process_reconstruction_results(masked_video, recon_caps, llm_response_text)

        except Exception as e:
            logging.error(f"{e} for {masked_video.video_id=}", exc_info=True)

            return self._create_error_result(
                masked_video.video_id,
                str(e),
                {"raw_response": self.llm_model.last_raw_response}
            )

    def _get_llm_response(self, masked_video: CaptionedVideo) -> str:
        """Generate prompt and get response from LLM."""
        prompt = self.prompt_builder.build_prompt(masked_video)
        logging.debug(f"video_id={masked_video.video_id} {prompt=}")
        return self.llm_model.call(prompt).text

    @staticmethod
    def _parse_and_validate_response(llm_response_text: str) -> tuple[dict[int, str], dict[int, int]]:
        """Parse LLM response and convert to dictionary format."""
        reconstructed_video = parse_llm_response(model=Reconstructed, response_text=llm_response_text)
        if not reconstructed_video:
            return {}, {}
        return reconstructed_video.to_dict()

    def _process_reconstruction_results(self, masked_video: CaptionedVideo, recon_caps: dict[int, str],
                                        llm_response_text: str) -> Reconstructed:
        """Process reconstruction results and categorize clips."""
        ok, failed, changed_unmasked, reconstructed_dict = self._categorize_clips(masked_video, recon_caps)

        debug_data = None
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

    @staticmethod
    def _categorize_clips(masked_video: CaptionedVideo, recon_caps: dict[int, str]) -> tuple[
        list[int], list[int], list[int], dict[int, str]]:
        """Categorize clips into successful, failed, and changed categories."""
        ok = []
        failed = []
        changed_unmasked = []
        reconstructed_dict: dict[int, str] = {}

        for c in masked_video.clips:
            if c.is_masked():
                if new_cap := recon_caps.get(c.index):
                    ok.append(c.index)
                    reconstructed_dict[c.index] = new_cap
                else:
                    failed.append(c.index)
                    reconstructed_dict[c.index] = ""  # TODO check if needed, check BertScore is 0
            elif c.index in recon_caps and c.caption != recon_caps.get(c.index):
                changed_unmasked.append(c.index)

        return ok, failed, changed_unmasked, reconstructed_dict

    def _create_error_result(self, video_id: str, error_message: str, extra_debug_data: dict = None) -> Reconstructed:
        """Create a Reconstructed result for error cases."""
        debug_data = {
            "error": error_message,
            "raw_response": self.llm_model.last_raw_response
        }
        if extra_debug_data:
            debug_data.update(extra_debug_data)

        return Reconstructed(
            video_id=video_id,
            reconstructed_captions={},
            debug_data=debug_data
        )

class ReconstructionStrategyBuilder:
    """
    A builder class responsible for creating reconstruction strategy objects.
    """
    def __init__(self, llm_cache, master_seed:int, llm_client:genai.Client):
        self.master_seed = master_seed
        self.llm_manager_builder = LLM_Manager_Builder(llm_client, llm_cache)

    def get_strategy(self, strategy_config: dict) -> ReconstructionStrategy:
        """
        Builds and returns a specific strategy instance based on the config.
        """
        strategy_type = strategy_config.get("type")
        if not strategy_type:
            raise UserFacingError("'type' must be specified in the strategy configuration.")

        if strategy_type == "llm":
            llm_conf = strategy_config['llm'].copy()
            llm_conf['seed'] = llm_conf.get('seed',0)+self.master_seed
            return LLMStrategy(
                name=strategy_config["name"],
                llm_model=self.llm_manager_builder.from_config(llm_conf),
                prompt_builder=JSONPromptBuilder.from_config(llm_conf)
            )

        elif strategy_type == "baseline_repeat_last":
            return BaselineRepeatStrategy()

        else:
            raise NotImplementedError(f"Strategy type '{strategy_type}' is not implemented.")
