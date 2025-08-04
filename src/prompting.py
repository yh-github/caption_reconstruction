from abc import ABC, abstractmethod
from data_models import CaptionedVideo


class PromptBuilder(ABC):
    """An abstract base class for all prompt building strategies."""
    @abstractmethod
    def build_prompt(self, masked_video: CaptionedVideo) -> str:
        pass


class PromptBuilderIndexedData(PromptBuilder):

    def build_prompt(self, masked_video: CaptionedVideo) -> str:
        json_lines = [clip.model_dump_json()+"\n" for i, clip in enumerate(masked_video.clips)]
        return "".join(json_lines)
