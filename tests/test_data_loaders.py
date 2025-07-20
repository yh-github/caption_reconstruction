# tests/test_data_loaders.py
import pytest
from data_loaders import VatexLoader, VideoStorytellingLoader, get_data_loader
from data_models import CaptionedVideo, CaptionedClip

def test_video_storytelling_loader():
    """
    Tests that the VideoStorytellingLoader returns a list of CaptionedVideo objects.
    """
    # Arrange
    mock_dir_path = "tests/fixtures/storytelling_mock"
    loader = VideoStorytellingLoader(mock_dir_path)

    # Act
    videos = loader.load()

    # Assert
    assert len(videos) == 1 # We have one video file in our mock data
    assert isinstance(videos[0], CaptionedVideo)
    
    # Check the contents of the video object
    assert videos[0].video_id == "4nAse9fEOww.mp4"
    assert len(videos[0].clips) == 3

def test_vatex_loader():
    """
    Tests that the VatexLoader returns a list of CaptionedVideo objects.
    """
    # Arrange
    mock_file_path = "tests/fixtures/vatex_mock/mock_data.json"
    loader = VatexLoader(mock_file_path)

    # Act
    videos = loader.load()

    # Assert
    assert len(videos) == 2 # We have two videos in our mock data
    assert isinstance(videos[0], CaptionedVideo)
    
    # Check the first video
    assert videos[0].video_id == "video1"
    assert len(videos[0].clips) == 5 # Should only take the first 5 captions

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
