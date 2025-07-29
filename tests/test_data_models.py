import pytest
from data_models import CaptionedClip, NarrativeOnlyPayload, TimestampRange
from reconstruction_strategies import Reconstructed

# --- Test Data Fixture ---
@pytest.fixture
def sample_clips() -> tuple[list[CaptionedClip], dict[int, CaptionedClip]]:
    """Provides a sample list of original clips and a dict of reconstructed clips."""
    
    # Original clips that serve as the ground truth
    orig_clips = [
        CaptionedClip(
            timestamp=TimestampRange(start=0.0, end=5.0),
            data=NarrativeOnlyPayload(caption="The original first sentence.")
        ),
        CaptionedClip(
            timestamp=TimestampRange(start=5.0, end=10.0),
            data=NarrativeOnlyPayload(caption="The original second sentence.")
        ),
        CaptionedClip(
            timestamp=TimestampRange(start=10.0, end=15.0),
            data=NarrativeOnlyPayload(caption="The original third sentence.")
        ),
    ]

    # Reconstructed clips that our system generated
    reconstructed_clips = {
        0: CaptionedClip(
            timestamp=TimestampRange(start=0.5, end=4.8),
            data=NarrativeOnlyPayload(caption="A reconstructed first sentence.")
        ),
        2: CaptionedClip(
            timestamp=TimestampRange(start=10.2, end=14.9),
            data=NarrativeOnlyPayload(caption="A reconstructed third sentence.")
        ),
    }
    
    return orig_clips, reconstructed_clips

# --- Tests for the Reconstructed Class ---

def test_reconstructed_initialization(sample_clips):
    """🧪 Tests if a Reconstructed object is created correctly with initial data."""
    _, reconstructed_clips = sample_clips
    
    recon = Reconstructed(video_id="vid_001", reconstructed_clips=reconstructed_clips)

    assert recon.video_id == "vid_001"
    assert len(recon.reconstructed_clips) == 2
    assert recon.reconstructed_clips[0].data.caption == "A reconstructed first sentence."
    assert recon.skip_reason is None

def test_align_method(sample_clips):
    """🧪 Tests the `align` method to ensure it correctly extracts captions."""
    orig_clips, reconstructed_clips = sample_clips
    recon = Reconstructed(video_id="vid_001", reconstructed_clips=reconstructed_clips)

    candidates, references = recon.align(orig_clips)

    expected_candidates = [
        "A reconstructed first sentence.",
        "A reconstructed third sentence."
    ]
    expected_references = [
        "The original first sentence.",
        "The original third sentence."
    ]

    assert candidates == expected_candidates
    assert references == expected_references

def test_skip_method(sample_clips):
    """🧪 Tests the `skip` method to ensure it sets the reason and supports chaining."""
    _, reconstructed_clips = sample_clips
    recon = Reconstructed(video_id="vid_001", reconstructed_clips=reconstructed_clips)
    
    reason = "Video was too blurry to process."
    result = recon.skip(reason)

    assert recon.skip_reason == reason
    assert result is recon

def test_with_metrics_method(sample_clips):
    """🧪 Tests the `with_metrics` method to ensure it attaches metrics."""
    _, reconstructed_clips = sample_clips
    recon = Reconstructed(video_id="vid_001", reconstructed_clips=reconstructed_clips)
    
    metrics = {"bleu_score": 0.88, "rougeL": 0.91}
    result = recon.with_metrics(metrics)

    assert recon.metrics == metrics
    assert result is recon

def test_method_chaining(sample_clips):
    """🧪 Tests if `skip` and `with_metrics` can be chained together fluently."""
    _, reconstructed_clips = sample_clips
    recon = Reconstructed(video_id="vid_001", reconstructed_clips=reconstructed_clips)

    reason = "Processing failed."
    metrics = {"error_code": 500}

    result = recon.skip(reason).with_metrics(metrics)

    assert result.skip_reason == reason
    assert result.metrics == metrics
    assert result is recon
