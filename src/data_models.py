# src/data_models.py
from pydantic import BaseModel, Field, ConfigDict
from typing import Union
from constants import DATA_MISSING

# --- Data Payload Models ---
class NarrativeOnlyPayload(BaseModel):
    """The data payload for a simple, description-only clip."""
    description: str

class StructuredPayload(BaseModel):
    """The data payload for a clip with structured data."""
    description: str
    objects: list[str] = Field(default_factory=list)
    verbs: list[str] = Field(default_factory=list)


# --- Main Clip Model ---
class CaptionedClip(BaseModel):
    """
    Represents a single clip in the caption.
    The 'data' field can either hold a detailed payload object or our
    special DATA_MISSING token.
    """
    # This is the modern way to configure Pydantic models
    model_config = ConfigDict(arbitrary_types_allowed=True)

    timestamp: float = Field(..., gt=0)
    data: Union[NarrativeOnlyPayload, StructuredPayload, str]


class CaptionedVideo(BaseModel):
    """
    Represents a complete transcript for a single video, including metadata
    and the sequence of clips.
    """
    video_id: str = Field(..., description="A unique identifier for the video.")
    clips: list[CaptionedClip] = Field(..., description="An ordered list of captioned clips.")
