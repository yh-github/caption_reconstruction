from data_models import NarrativeOnlyClip, StructuredClip

# =================================================================
# == Formatter Functions
# =================================================================

def format_narrative_clip(clip: NarrativeOnlyClip) -> str:
    """Formats a NarrativeOnlyClip as a simple time-stamped sentence."""
    return f"Time: {clip.timestamp:.2f}s - Description: {clip.description}"

def format_structured_clip(clip: StructuredClip) -> str:
    """Formats a StructuredClip with its object and verb lists."""
    parts = [f"Time: {clip.timestamp:.2f}s", f"Description: {clip.description}"]
    if clip.objects:
        parts.append(f"Objects: {', '.join(clip.objects)}")
    if clip.verbs:
        parts.append(f"Verbs: {', '.join(clip.verbs)}")
    return " - ".join(parts)

# =================================================================
# == Formatter Factory
# =================================================================

def get_clip_formatter(richness_level: str):
    """Factory to select the appropriate formatter function from a config string."""
    formatters = {
        "narrative_only": format_narrative_clip,
        "structured": format_structured_clip,
    }
    formatter = formatters.get(richness_level)
    if formatter is None:
        raise ValueError(f"No formatter found for richness_level: '{richness_level}'")
    return formatter
