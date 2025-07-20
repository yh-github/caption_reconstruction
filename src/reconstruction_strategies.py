# src/reconstruction_strategies.py
from abc import ABC, abstractmethod
import mlflow

from data_models import CaptionedClip
from llm_interaction import call_llm
from parsers import parse_llm_response
from baselines import repeat_last_known_baseline
from pipeline import build_prompt

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

