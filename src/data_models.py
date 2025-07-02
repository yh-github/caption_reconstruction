from pydantic import BaseModel, Field

# =================================================================
# == Data Models - Pure Data Carriers
# =================================================================

class BaseTranscriptClip(BaseModel):
    """A base model ensuring all clips have a timestamp."""
    timestamp: float = Field(
        ...,
        gt=0,
        description="The end time of the clip in seconds."
    )

class NarrativeOnlyClip(BaseTranscriptClip):
    """A simple clip with only a narrative description."""
    description: str

class StructuredClip(BaseTranscriptClip):
    """A more detailed clip including objects and verbs."""
    description: str
    objects: list[str] = Field(default_factory=list)
    verbs: list[str] = Field(default_factory=list)

# =================================================================
# == Model Factory
# =================================================================

def get_clip_model(richness_level: str):
    """Factory to select the appropriate Pydantic model from a config string."""
    models = {
        "narrative_only": NarrativeOnlyClip,
        "structured": StructuredClip,
    }
    model = models.get(richness_level)
    if model is None:
        raise ValueError(f"Unknown richness_level: '{richness_level}'")
    return model
