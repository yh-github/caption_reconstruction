from pydantic import BaseModel, Field, field_validator, ConfigDict


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

class ReconstructedOutput(BaseModel):
    """
    Represents the sparse reconstruction output from the LLM for a single video.
    """
    model_config = ConfigDict(frozen=True)

    reconstructed_captions: dict[int, str] = Field(
        ...,
        description="A dictionary mapping the integer index of a masked clip to its reconstructed text caption."
    )