# src/reconstruction_strategies.py
from abc import ABC, abstractmethod
import mlflow

from data_models import CaptionedClip
from llm_interaction import call_llm
from parsers import parse_llm_response
from baselines import repeat_last_known_baseline
from pipeline import build_prompt

def build_prompt(masked_captions: list[CaptionedClip], config: dict) -> str:
    """Builds the final JSON prompt to be sent to the LLM."""
    logging.info("Building LLM prompt as JSON...")
    caption_for_json = [clip.model_dump() for clip in masked_captions]
    json_prompt_data = json.dumps(caption_for_json, indent=2)

    instruction = (
        "You are an expert video analyst. Reconstruct the full data object for any "
        f"timestamp where the 'data' field is the token '{DATA_MISSING}'. "
        "Return the complete JSON list with all masks filled."
    )

    final_prompt = f"{instruction}\n\n---\n\n{json_prompt_data}"

    # Check if MLflow is active before logging, which is safer for tests
    if mlflow.active_run():
        mlflow.log_text(final_prompt, "prompt.txt")

    return final_prompt


class ReconstructionStrategy(ABC):
    """An abstract base class for all reconstruction methods."""
    @abstractmethod
    def reconstruct(self, masked_captions: list[CaptionedClip]) -> list[CaptionedClip] | None:
        """Takes a masked caption and returns a reconstructed one."""
        pass

class LLMStrategy(ReconstructionStrategy):
    """The strategy for using an LLM for reconstruction."""
    def __init__(self, llm_model, config):
        self.llm_model = llm_model
        self.config = config

    def reconstruct(self, masked_captions: list[CaptionedClip]) -> list[CaptionedClip] | None:
        prompt = build_prompt(masked_captions)
        llm_response_text = call_llm(self.llm_model, prompt)
        mlflow.log_text(llm_response_text, "llm_response.txt")
        return parse_llm_response(llm_response_text)

class BaselineStrategy(ReconstructionStrategy):
    """The strategy for using the 'repeat last known' baseline."""
    def reconstruct(self, masked_captions: list[CaptionedClip]) -> list[CaptionedClip] | None:
        return repeat_last_known_baseline(masked_captions)

