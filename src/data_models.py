from pydantic import BaseModel, Field, field_validator, ConfigDict, RootModel


class TimestampRange(BaseModel):
    """Represents a time range with a start and end point."""
    model_config = ConfigDict(frozen=True)

    start: float = Field(..., ge=0, description="Start time of the clip in seconds.")
    duration: float = Field(..., ge=0, description="Duration of the clip in seconds.")

    # noinspection PyMethodParameters
    @field_validator('start', 'duration')
    def round_timestamp(cls, value):
        """Rounds the timestamp to 3 decimal places for reproducibility."""
        return round(value, 3)

class CaptionedClip(BaseModel):
    model_config = ConfigDict(frozen=True)

    index: int
    timestamp: TimestampRange
    caption: str|None

class CaptionedVideo(BaseModel):
    """
    Represents a complete sequence of clips for a single video
    """
    model_config = ConfigDict(frozen=True)

    video_id: str = Field(..., description="A unique identifier for the video.")
    clips: list[CaptionedClip] = Field(..., description="An ordered list of captioned clips.")

    @field_validator('clips')
    def check_indices_are_sequential(cls, clips: list[CaptionedClip]) -> list[CaptionedClip]:
        """
        Validates that the clip indices are set correctly and sequentially.
        """
        for i, clip in enumerate(clips):
            if clip.index != i:
                raise ValueError(f"Clip index mismatch at position {i}. Expected index {i}, but got {clip.index}.")
        return clips

# class ReconstructedOutput(BaseModel):
#     """
#     Represents the sparse reconstruction output from the LLM for a single video.
#     """
#     model_config = ConfigDict(frozen=True)
#
#     reconstructed_captions: dict[int, str] = Field(
#         ...,
#         description="A dictionary mapping the integer index of a masked clip to its reconstructed text caption."
#     )

class ReconstructedCaption(BaseModel):
    """Represents a single reconstructed caption with its original index."""
    index: int = Field(..., description="The original index of the clip that was reconstructed.")
    caption: str = Field(..., description="The newly generated caption for the clip.")

class ReconstructedCaptions(RootModel[list[ReconstructedCaption]]):
    def to_dict(self) -> dict[int, str]:
        """
        Converts the list of reconstructed captions into a dictionary
        mapping the clip index to its caption.
        """
        return {item.index: item.caption for item in self.root}
