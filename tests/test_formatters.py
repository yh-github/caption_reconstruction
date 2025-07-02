# tests/test_formatters.py
from data_models import NarrativeOnlyClip, StructuredClip
from formatters import format_narrative_clip, format_structured_clip

def test_format_narrative_only_clip():
    """
    Tests that the narrative-only formatter produces the correct string.
    """
    # Arrange
    clip = NarrativeOnlyClip(timestamp=3.0, description="The person picks up a red book.")
    expected_string = "Time: 3.00s - Description: The person picks up a red book."
    
    # Act
    actual_string = format_narrative_clip(clip)
    
    # Assert
    assert actual_string == expected_string

def test_format_structured_clip():
    """
    Tests that the structured formatter correctly includes all data.
    """
    # Arrange
    clip = StructuredClip(
        timestamp=4.5,
        description="A woman is kicking a soccer ball.",
        objects=["woman", "soccer ball"],
        verbs=["kicking"]
    )
    expected_string = "Time: 4.50s - Description: A woman is kicking a soccer ball. - Objects: woman, soccer ball - Verbs: kicking"
    
    # Act
    actual_string = format_structured_clip(clip)
    
    # Assert
    assert actual_string == expected_string

def test_format_structured_clip_with_missing_data():
    """
    Tests that the structured formatter handles missing optional data gracefully.
    """
    # Arrange
    clip = StructuredClip(
        timestamp=5.0,
        description="A car drives by."
        # No objects or verbs are provided
    )
    expected_string = "Time: 5.00s - Description: A car drives by."

    # Act
    actual_string = format_structured_clip(clip)
    
    # Assert
    assert actual_string == expected_string
