# tests/test_data_loaders.py
import pytest
from data_loaders import VatexLoader, get_data_loader
from data_models import TranscriptClip
from data_loaders import VideoStorytellingLoader

def test_video_storytelling_loader():
    """
    Tests that the VideoStorytellingLoader correctly parses the mock TXT data,
    converts timestamps to seconds, and extracts descriptions.
    """
    # Arrange
    mock_dir_path = "tests/fixtures/storytelling_mock"
    loader = VideoStorytellingLoader(mock_dir_path)

    # Act
    clips = loader.load()

    # Assert
    assert len(clips) == 3

    # Check the first clip
    assert isinstance(clips[0], TranscriptClip)
    assert clips[0].data.description == "seating of wedding party and family"
    assert clips[0].timestamp == 120.0 # 2 minutes = 120 seconds

    # Check the last clip
    assert clips[2].data.description == "the bride arrives and is escort by her father to the pavilion"
    assert clips[2].timestamp == 393.0 # 6 minutes and 33 seconds
def test_vatex_loader():
    """
    Tests that the VatexLoader correctly parses the mock JSON data,
    takes only the first 5 captions, and creates placeholder timestamps.
    """
    # Arrange
    mock_file_path = "tests/fixtures/vatex_mock/mock_data.json"
    loader = VatexLoader(mock_file_path)

    # Act
    clips = loader.load()

    # Assert
    # 2 videos * 5 captions each = 10 clips total
    assert len(clips) == 10
    
    # Check the first video's clips
    assert isinstance(clips[0], TranscriptClip)
    assert clips[0].data.description == "caption 1a"
    assert clips[0].timestamp == 1.0
    
    # Check that the 6th caption from the first video was correctly ignored
    assert "caption 1f" not in [c.data.description for c in clips]
    
    # Check the last clip of the second video
    assert clips[9].data.description == "caption 2e"
    assert clips[9].timestamp == 5.0

def test_get_data_loader_factory():
    """
    Tests that the factory function returns the correct loader instance
    based on the configuration.
    """
    # Arrange
    vatex_config = {"data": {"name": "vatex", "path": "some/path.json"}}
    story_config = {"data": {"name": "video_storytelling", "path": "some/dir"}}
    bad_config = {"data": {"name": "unknown_dataset", "path": "..."}}

    # Act
    vatex_loader = get_data_loader(vatex_config)
    story_loader = get_data_loader(story_config)

    # Assert
    assert isinstance(vatex_loader, VatexLoader)
    assert isinstance(story_loader, VideoStorytellingLoader)
    with pytest.raises(NotImplementedError):
        get_data_loader(bad_config)
