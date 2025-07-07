# tests/test_baselines.py
from baselines import repeat_last_known_baseline
from data_models import TranscriptClip, NarrativeOnlyPayload
from constants import DATA_MISSING

def test_repeat_last_known_baseline():
    """Tests that the baseline correctly fills masked clips."""
    # Arrange
    masked_transcript = [
        TranscriptClip(timestamp=1.0, data=NarrativeOnlyPayload(description="first")),
        TranscriptClip(timestamp=2.0, data=DATA_MISSING),
        TranscriptClip(timestamp=3.0, data=DATA_MISSING),
        TranscriptClip(timestamp=4.0, data=NarrativeOnlyPayload(description="fourth")),
        TranscriptClip(timestamp=5.0, data=DATA_MISSING),
    ]

    # Act
    result = repeat_last_known_baseline(masked_transcript)

    # Assert
    assert result[1].data.description == "first"
    assert result[2].data.description == "first"
    assert result[4].data.description == "fourth"

def test_repeat_last_known_baseline_starts_with_mask():
    """Tests the edge case where the first clip is masked."""
    masked_transcript = [
        TranscriptClip(timestamp=1.0, data=DATA_MISSING),
        TranscriptClip(timestamp=2.0, data=NarrativeOnlyPayload(description="second")),
    ]
    result = repeat_last_known_baseline(masked_transcript)
    assert result[0].data == DATA_MISSING # Should remain masked

