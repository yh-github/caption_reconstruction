import json
from abc import ABC, abstractmethod
from data_models import CaptionedVideo
from constants import DATA_MISSING

class BasePromptBuilder(ABC):
    """An abstract base class for all prompt building strategies."""
    @abstractmethod
    def build_prompt(self, masked_video: CaptionedVideo) -> str:
        pass

class JSONPromptBuilder(BasePromptBuilder):
    """Builds a prompt that instructs the LLM to work with JSON."""
    def __init__(self, instruction_template: str):
        self.instruction_template = instruction_template

    def build_prompt(self, masked_video: CaptionedVideo) -> str:
        """Builds the final JSON prompt to be sent to the LLM."""
        instruction = self.instruction_template.format(DATA_MISSING=DATA_MISSING)
        
        captions_for_json = [clip.model_dump() for clip in masked_video.clips]
        json_prompt_data = json.dumps(captions_for_json, indent=2)

        return f"{instruction}\n\n---\n\n{json_prompt_data}"

    @staticmethod
    def from_config(config: dict):
        """Constructs the builder from a configuration dictionary."""
        template_path = config.get("strategy", {}).get("prompt_template")
        if not template_path:
            raise ValueError("Prompt template path not specified in config.")
        # This method now calls the 'from_path' method
        return JSONPromptBuilder.from_path(template_path)

    @staticmethod
    def from_path(template_path: str):
        """Constructs the builder from a file path."""
        with open(template_path, 'r') as f:
            template_string = f.read().strip()
        # This method now calls the 'from_string' method
        return JSONPromptBuilder.from_string(template_string)

    @staticmethod
    def from_string(template_string: str):
        """Constructs the builder directly from a string."""
        return JSONPromptBuilder(instruction_template=template_string)
