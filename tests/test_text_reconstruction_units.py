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


def test_align_recon_indices_1_indexed_shift():
    """Test auto-alignment when LLM returns 1-indexed keys [1, 2, 3] for masked [0, 1, 2]."""
    video = create_mock_video([
        (0, None),
        (1, None),
        (2, None),
        (3, "fourth"),
    ])
    recon_caps = {1: "rec 0", 2: "rec 1", 3: "rec 2"}
    aligned = LLMStrategy._align_recon_indices(video, recon_caps)
    assert aligned == {0: "rec 0", 1: "rec 1", 2: "rec 2"}


def test_align_recon_indices_no_shift_needed():
    """Test that correctly 0-indexed reconstructions are not shifted."""
    video = create_mock_video([
        (0, None),
        (1, None),
        (2, "third"),
    ])
    recon_caps = {0: "rec 0", 1: "rec 1"}
    aligned = LLMStrategy._align_recon_indices(video, recon_caps)
    assert aligned == {0: "rec 0", 1: "rec 1"}


def test_align_recon_indices_middle_unaffected():
    """Test that middle gap (e.g. index > 0) is not modified by alignment."""
    video = create_mock_video([
        (0, "first"),
        (1, None),
        (2, None),
        (3, "fourth"),
    ])
    recon_caps = {1: "rec 1", 2: "rec 2"}
    aligned = LLMStrategy._align_recon_indices(video, recon_caps)
    assert aligned == {1: "rec 1", 2: "rec 2"}


def test_json_prompt_builder_positional_templates(tmp_path):
    """Test that JSONPromptBuilder routes start, default, and end templates and formats variables."""
    from llm.prompting import JSONPromptBuilder

    prompt_dir = tmp_path / "prompts"
    prompt_dir.mkdir()
    (prompt_dir / "default.txt").write_text("DEFAULT GAP {MISSING_INDICES}")
    (prompt_dir / "start.txt").write_text("START SCENE {START_INDEX}-{END_INDEX}")
    (prompt_dir / "end.txt").write_text("END SCENE {COUNT}")

    builder = JSONPromptBuilder.from_path(prompt_dir)

    # Start masked
    video_start = create_mock_video([(0, None), (1, None), (2, "present")])
    prompt_start = builder.build_prompt(video_start)
    assert "START SCENE 0-1" in prompt_start

    # Middle masked
    video_mid = create_mock_video([(0, "present"), (1, None), (2, "present")])
    prompt_mid = builder.build_prompt(video_mid)
    assert "DEFAULT GAP [1]" in prompt_mid

    # End masked
    video_end = create_mock_video([(0, "present"), (1, "present"), (2, None)])
    prompt_end = builder.build_prompt(video_end)
    assert "END SCENE 1" in prompt_end 
