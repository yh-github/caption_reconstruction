# src/data_models.py
from pydantic import BaseModel, Field
from typing import Union
from constants import DATA_MISSING

# --- Data Payload Models ---
# These models define the structure of the data when it's *not* masked.

class NarrativeOnlyPayload(BaseModel):
    """The data payload for a simple, description-only clip."""
    description: str

class StructuredPayload(BaseModel):
    """The data payload for a clip with structured data."""
    description: str
    objects: list[str] = Field(default_factory=list)
    verbs: list[str] = Field(default_factory=list)


# --- Main Clip Model ---
# This is the primary object for each timestamp in our transcript.

class TranscriptClip(BaseModel):
    """
    Represents a single clip in the transcript.
    The 'data' field can either hold a detailed payload object or our
    special DATA_MISSING token.
    """
    timestamp: float = Field(..., gt=0)
    data: Union[NarrativeOnlyPayload, StructuredPayload, str]

    class Config:
        # Pydantic configuration to allow custom types like our payload models
        arbitrary_types_allowed = True
