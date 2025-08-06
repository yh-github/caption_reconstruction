from abc import ABC, abstractmethod
from data_models import CaptionedVideo


class PromptBuilder(ABC):
    """An abstract base class for all prompt building strategies."""
    @abstractmethod
    def build_prompt(self, masked_video: CaptionedVideo) -> str:
        pass


class PromptBuilderIndexedData(PromptBuilder):

    def build_prompt(self, masked_video: CaptionedVideo) -> str:
        return ("[\n" +
                ",\n".join([
                    '  '+c.model_dump_json()
                    for c in masked_video.clips
                ])
                + "\n]")

