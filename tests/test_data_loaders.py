import pytest
from data_loaders import ToyDataLoader, VatexLoader, VideoStorytellingLoader, get_data_loader
from data_models import CaptionedVideo, CaptionedClip

def test_toy_data_loader_from_file():
    """
    Tests that the ToyDataLoader correctly loads and parses the mock
    JSON data file into a CaptionedVideo object.
    """
    # Arrange
    # Point the loader to our new standard toy data file
    mock_file_path = "datasets/toy_dataset/data.json"
    loader = ToyDataLoader(mock_file_path)

    # Act
    videos = loader.load()

    # Assert
    # 1. Check the high-level structure
    assert len(videos) == 2
    assert isinstance(videos[0], CaptionedVideo)

    # 2. Check the details of the loaded video
    video = videos[0]
    assert video.video_id == "toy_video_1"
    assert len(video.clips) == 10

    # 3. Spot-check a specific clip to ensure data is correct
    assert video.clips[2].data.description == "The person picks up a red book from the table."
    assert video.clips[2].timestamp == 3.0


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
    loader = VatexLoader(mock_file_path, limit=5)

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
    vatex_config = {"name": "vatex", "path": "some/path.json"}
    story_config = {"name": "video_storytelling", "path": "some/dir"}
    bad_config = {"name": "unknown_dataset", "path": "..."}

    # Act
    vatex_loader = get_data_loader(vatex_config)
    story_loader = get_data_loader(story_config)

    # Assert
    assert isinstance(vatex_loader, VatexLoader)
    assert isinstance(story_loader, VideoStorytellingLoader)
    with pytest.raises(NotImplementedError):
        get_data_loader(bad_config)

