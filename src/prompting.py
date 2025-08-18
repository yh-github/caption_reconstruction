from abc import ABC, abstractmethod
from data_models.captions_only import CaptionedVideo


class PromptBuilder(ABC):
    """An abstract base class for all prompt building strategies."""
    @abstractmethod
    def build_prompt(self, masked_video: CaptionedVideo) -> str:
        pass


class PromptBuilderDataOnly(PromptBuilder):

    def build_prompt(self, masked_video: CaptionedVideo) -> str:
        return ("[\n" +
                ",\n".join([
                    '  '+c.model_dump_json()
                    for c in masked_video.clips
                ])
                + "\n]")


class FormatDict(dict):

    @staticmethod
    def from_dict(d):
        return FormatDict(d if d else {})

    def __missing__(self, key):
        return '{' + str(key) + '}'

class JSONPromptBuilder(PromptBuilder):
    """Builds a prompt that instructs the LLM to work with JSON."""

    def __init__(self, instruction_template: str, consts:dict[str,str]=None):
        self.instruction_template = instruction_template.format_map(FormatDict.from_dict(consts))
        self.data_prompter = PromptBuilderDataOnly()

    def build_prompt(self, masked_video: CaptionedVideo) -> str:
        """Builds the final JSON prompt to be sent to the LLM."""
        instruction = self.instruction_template #.format(DATA_MISSING=DATA_MISSING)

        # captions_for_json = [clip.model_dump() for clip in masked_video.clips]
        # json_prompt_data = json.dumps(captions_for_json, indent=2)
        json_prompt_data = self.data_prompter.build_prompt(masked_video)

        return f"{instruction}\n\n{json_prompt_data}"

    def set_consts(self, consts:dict[str,str]):
        self.instruction_template = self.instruction_template.format_map(FormatDict.from_dict(consts))
        return self

    def with_vars(self, values:dict[str,str]) -> str:
        return self.instruction_template.format(**values)

    @staticmethod
    def from_config(config: dict):
        """Constructs the builder from a configuration dictionary."""
        template_path = config.get("prompt_template")
        if not template_path:
            raise ValueError("Prompt template path not specified in config.")
        return JSONPromptBuilder.from_path(template_path)

    @staticmethod
    def from_path(template_path: str):
        """Constructs the builder from a file path."""
        with open(template_path, 'r') as f:
            template_string = f.read().strip()
        return JSONPromptBuilder.from_string(template_string)

    @staticmethod
    def from_string(template_string: str):
        """Constructs the builder directly from a string."""
        return JSONPromptBuilder(instruction_template=template_string)
