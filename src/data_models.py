from pydantic import BaseModel, Field, ConfigDict, field_validator
from typing import Union

class NarrativeOnlyPayload(BaseModel):
    caption: str

class StructuredPayload(BaseModel):
    caption: str
    objects: list[str] = Field(default_factory=list)
    verbs: list[str] = Field(default_factory=list)

class TimestampRange(BaseModel):
    """Represents a time range with a start and end point."""
    start: float = Field(..., ge=0, description="Start time of the clip in seconds.")
    end: float = Field(..., ge=0, description="End time of the clip in seconds.")

    # Using field_validator for Pydantic V2 compatibility
    @field_validator('end') # Use field_validator
    @classmethod # field_validator should be a class method
    def end_must_be_after_start(cls, end_time, info): # field_validator takes 'info' instead of 'values'
        """Ensures the end time is greater than the start time."""
        if 'start' in info.data and end_time <= info.data['start']: # Access data via info.data
            raise ValueError('End time must be after start time')
        return end_time

class CaptionedClip(BaseModel):
    """
    Represents a single clip in the caption using a time range.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)
    timestamp: TimestampRange
    data: Union[NarrativeOnlyPayload, StructuredPayload, str]

class CaptionedVideo(BaseModel):
    """
    Represents a complete transcript for a single video, including metadata
    and the sequence of clips.
    """
    video_id: str = Field(..., description="A unique identifier for the video.")
    clips: list[CaptionedClip] = Field(..., description="An ordered list of captioned clips.")
