import pytest
from reconstruction.text_reconstruction import LLMStrategy
from data_models.captions_only import CaptionedVideo, CaptionedClip, TimestampRange

def create_mock_video(clips_data):
    """
    Helper to create a CaptionedVideo from a list of (index, caption|None) tuples.
    None caption implies masked.
    """
    clips = []
    for idx, (i, cap) in enumerate(clips_data):
        # simple timestamp mock
        ts = TimestampRange(start=idx*10, duration=10)
        clips.append(CaptionedClip(index=i, caption=cap, timestamp=ts))
    return CaptionedVideo(video_id="test_vid", clips=clips)

def test_categorize_clips_basic():
    """Test basic successful reconstruction of masked clips."""
    # Clip 0: Masked
    # Clip 1: Unmasked ("hello")
    # Clip 2: Masked
    video = create_mock_video([
        (0, None),
        (1, "hello"),
        (2, None)
    ])
    
    # LLM returns:
    # 0 -> "reconstructed 0"
    # 2 -> "reconstructed 2"
    recon_caps = {
        0: "reconstructed 0",
        2: "reconstructed 2"
    }
    
    result = LLMStrategy._categorize_clips(video, recon_caps)
    
    # Assertions using the new Pydantic model
    assert 0 in result.ok
    assert 2 in result.ok
    assert result.failed == []
    assert result.changed_unmasked == []
    assert result.reconstructed_dict[0] == "reconstructed 0"
    assert result.reconstructed_dict[2] == "reconstructed 2"

def test_categorize_clips_failures():
    """Test partial failure (missing key)."""
    video = create_mock_video([
        (0, None),
        (1, None)
    ])
    
    # Only 0 is returned
    recon_caps = {0: "rec 0"}
    
    result = LLMStrategy._categorize_clips(video, recon_caps)
    
    assert 0 in result.ok
    assert 1 in result.failed
    assert result.reconstructed_dict[1] == "" # Empty string for failed

def test_categorize_clips_changed_unmasked():
    """Test detection of changes to unmasked clips (hallucination overlap)."""
    video = create_mock_video([
        (0, "original")
    ])
    
    # LLM tries to rewrite clip 0
    recon_caps = {0: "hallucinated rewrite"}
    
    result = LLMStrategy._categorize_clips(video, recon_caps)
    
    assert result.changed_unmasked == [0]
    # Note: changed unmasked are NOT added to reconstructed_dict in current logic
    assert 0 not in result.reconstructed_dict 
