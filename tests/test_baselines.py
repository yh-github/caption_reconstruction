# tests/test_baselines.py
from baselines import repeat_last_known_baseline
from data_models import CaptionedClip, NarrativeOnlyPayload
from constants import DATA_MISSING

def test_repeat_last_known_baseline():
    """Tests that the baseline correctly fills masked clips."""
    # Arrange
    masked_captions = [
        CaptionedClip(timestamp=1.0, data=NarrativeOnlyPayload(description="first")),
        CaptionedClip(timestamp=2.0, data=DATA_MISSING),
        CaptionedClip(timestamp=3.0, data=DATA_MISSING),
        CaptionedClip(timestamp=4.0, data=NarrativeOnlyPayload(description="fourth")),
        CaptionedClip(timestamp=5.0, data=DATA_MISSING),
    ]

    # Act
    result = repeat_last_known_baseline(masked_captions)

    # Assert
    assert result[1].data.description == "first"
    assert result[2].data.description == "first"
    assert result[4].data.description == "fourth"

def test_repeat_last_known_baseline_starts_with_mask():
    """Tests the edge case where the first clip is masked."""
    masked_captions = [
        CaptionedClip(timestamp=1.0, data=DATA_MISSING),
        CaptionedClip(timestamp=2.0, data=NarrativeOnlyPayload(description="second")),
    ]
    result = repeat_last_known_baseline(masked_captions)
    assert result[0].data == DATA_MISSING # Should remain masked

def test_repeat_last_known_baseline_starts_with_mask():
    """
    Tests that the baseline correctly back-fills when the first clip is masked.
    """
    # Arrange
    masked_captions = [
        CaptionedClip(timestamp=1.0, data=DATA_MISSING),
        CaptionedClip(timestamp=2.0, data=DATA_MISSING),
        CaptionedClip(timestamp=3.0, data=NarrativeOnlyPayload(description="third")),
    ]
    
    # Act
    result = repeat_last_known_baseline(masked_captions)

    # Assert
    # The first two clips should be filled with the data from the third clip.
    assert result[0].data.description == "third"
    assert result[1].data.description == "third"
    # The third clip should be untouched.
    assert result[2].data.description == "third"
