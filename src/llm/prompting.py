from abc import ABC, abstractmethod
from typing import Iterator

from data_models.captions_only import CaptionedVideo
import re


def simple_safe_format(template_string: str, data: dict[str, str]) -> tuple[str, list[str]]:
    """
    Replaces placeholders in a string with values from a dictionary.

    This function is a "safe" formatter. It finds all occurrences of "{key}"
    and replaces them with the corresponding value from the `data` dictionary.
    If a key found in the template does not exist in the dictionary, the
    placeholder is left unchanged, and no error is raised.

    Args:
        template_string: The string containing placeholders like "{key}".
        data: A dictionary mapping keys to their replacement values.

    Returns:
        A tuple containing:
        - The formatted string.
        - A list of unique keys that were in the template but not in the data dict.
    """
    missing_keys = set()

    def good_key(k:str):
        return isinstance(k,str) and k.isupper() and k.isidentifier()

    assert all(good_key(k) for k in data.keys()), f"Bad keys: {[k for k in data.keys() if not good_key(k)]}"
    assert all(isinstance(v,str) for v in data.values()), f"Bad value types {[f'{v}:{type(v)}' for v in data.values()]}"

    def replacer(match: re.Match) -> str:
        """
        This is the replacement function called by re.sub for each match.
        It also tracks any keys that are not found in the data dictionary.
        """
        # The key is the content inside the curly braces, which is capture group 1.
        key = match.group(1)

        # Check if the key exists in the data.
        if key in data:
            return data[key]
        else:
            # If the key is missing, add it to our set and return the original
            # placeholder (e.g., "{MISSING_KEY}").
            missing_keys.add(key)
            return match.group(0)

    # This regex finds a literal '{', captures a key (letters, numbers, or
    # underscores), and finds a literal '}'.
    # re.sub calls the `replacer` function for every match it finds.
    formatted_string = re.sub(r"\{([A-Z0-9_]+)\}", replacer, template_string)

    # Return the formatted string and the sorted list of unique missing keys.
    return formatted_string, sorted(list(missing_keys))


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


class JSONPromptBuilder(PromptBuilder):
    """Builds a prompt that instructs the LLM to work with JSON."""

    def __init__(self, instruction_template: str, consts:dict[str,str]|None=None):
        self._instruction_template = instruction_template
        self.set_consts(consts)
        self._data_prompter = PromptBuilderDataOnly()

    def build_prompt(self, masked_video: CaptionedVideo) -> str:
        """Builds the final JSON prompt to be sent to the LLM."""
        instruction = self._instruction_template #.format(DATA_MISSING=DATA_MISSING)

        # captions_for_json = [clip.model_dump() for clip in masked_video.clips]
        # json_prompt_data = json.dumps(captions_for_json, indent=2)
        json_prompt_data = self._data_prompter.build_prompt(masked_video)

        return f"{instruction}\n\n{json_prompt_data}"

    def set_consts(self, consts:dict[str,str]|None):
        if consts:
            self._instruction_template, missing_keys = simple_safe_format(self._instruction_template, consts)
        return self

    def with_vars(self, values:dict[str,str]) -> str:
        formatted_string, missing_keys = simple_safe_format(self._instruction_template, values)
        if missing_keys:
            raise ValueError(f"Missing keys in prompt template: {missing_keys}")
        return formatted_string

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


def numbered_list(xs:Iterator[str]) -> str:
    return "".join(f"{i}. {x}\n" for i, x in enumerate(xs, start=1))
