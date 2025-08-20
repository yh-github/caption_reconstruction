import pytest
from data_models.video_link import VideoLinkData


def test_mask_with_valid_percentages():
    """
    Test the mask method with valid percentage values splitting the VideoLinkData instance.
    """
    # Arrange
    video_data = VideoLinkData(
        video_id="test_id",
        uri="http://example.com/video.mp4",
        start_offset=0.0,
        end_offset=100.0,
    )

    start_percentage = 0.2
    end_percentage = 0.8

    # Act
    result = video_data.mask(start_percentage=start_percentage, end_percentage=end_percentage)

    # Assert
    assert len(result) == 2
    assert result[0].start_offset == 0.0
    assert result[0].end_offset == 20.0
    assert result[1].start_offset == 80.0
    assert result[1].end_offset == 100.0


def test_mask_with_invalid_percentage_values():
    """
    Test the mask method raises ValueError when invalid percentages are provided.
    """
    # Arrange
    video_data = VideoLinkData(
        video_id="test_id",
        uri="http://example.com/video.mp4",
        start_offset=0.0,
        end_offset=100.0,
    )

    # Act & Assert
    with pytest.raises(ValueError, match="Percentages must be between 0 and 1"):
        video_data.mask(start_percentage=-0.1, end_percentage=0.5)

    with pytest.raises(ValueError, match="Percentages must be between 0 and 1"):
        video_data.mask(start_percentage=0.5, end_percentage=1.1)

    with pytest.raises(ValueError, match="end_percentage must be greater than start_percentage"):
        video_data.mask(start_percentage=0.6, end_percentage=0.4)


def test_mask_with_edge_case_percentages():
    """
    Test the mask method with edge cases, such as start_percentage=0 and end_percentage=1, where no split occurs.
    """
    # Arrange
    video_data = VideoLinkData(
        video_id="test_id",
        uri="http://example.com/video.mp4",
        start_offset=0.0,
        end_offset=100.0,
    )

    start_percentage = 0.0
    end_percentage = 1.0

    # Act
    result = video_data.mask(start_percentage=start_percentage, end_percentage=end_percentage)

    # Assert
    assert len(result) == 0


def test_mask_with_non_zero_start():
    """
    Test the mask method with a video that has non-zero start offset.
    """
    # Arrange
    video_data = VideoLinkData(
        video_id="test_id",
        uri="http://example.com/video.mp4",
        start_offset=50.0,
        end_offset=150.0,
    )

    # Act
    result = video_data.mask(start_percentage=0.2, end_percentage=0.8)

    # Assert
    assert len(result) == 2
    assert result[0].start_offset == 50.0
    assert result[0].end_offset == 70.0
    assert result[1].start_offset == 130.0
    assert result[1].end_offset == 150.0


def test_mask_with_small_percentage_difference():
    """
    Test the mask method with very small difference between percentages.
    """
    # Arrange
    video_data = VideoLinkData(
        video_id="test_id",
        uri="http://example.com/video.mp4",
        start_offset=0.0,
        end_offset=100.0,
    )

    # Act
    result = video_data.mask(start_percentage=0.495, end_percentage=0.505)

    # Assert
    assert len(result) == 2
    assert result[0].start_offset == 0.0
    assert result[0].end_offset == 49.5
    assert result[1].start_offset == 50.5
    assert result[1].end_offset == 100.0


def test_mask_with_identical_percentages():
    """
    Test the mask method when start and end percentages are identical.
    """
    # Arrange
    video_data = VideoLinkData(
        video_id="test_id",
        uri="http://example.com/video.mp4",
        start_offset=0.0,
        end_offset=100.0,
    )

    # Act & Assert
    with pytest.raises(ValueError, match="end_percentage must be greater than start_percentage"):
        video_data.mask(start_percentage=0.5, end_percentage=0.5)
