import pytest
from reconstruction.masking import PartitionMasking, get_masking_strategies, FixedFillMasking
from data_models.captions_only import CaptionedClip,  TimestampRange

# --- The Fixture (no changes needed) ---
@pytest.fixture
def captions_of_length():
    """
    A factory fixture that creates a list of CaptionedClip objects
    of a specified length.
    """
    def _create_captions(num_clips):
        return [
            CaptionedClip(index=i, timestamp=TimestampRange(start=i, duration=1), caption=f"Clip {i+1}")
            for i in range(num_clips)
        ]
    return _create_captions


# --- Corrected Tests ---

@pytest.mark.parametrize(
    "num_clips, num_partitions, start_partition, expected_indices",
    [
        (10, 5, 1, {2, 3}),
        (7, 3, 2, {5, 6}),  # Corrected from {4, 5} to {5, 6} because 7/3 = {0:{0, 1, 2}, 1:{3, 4}, 2:{5, 6}}
        (20, 10, 8, {16, 17}),
    ]
)
def test_partition_masking_scenarios(captions_of_length, num_clips, num_partitions, start_partition, expected_indices):
    """
    Tests specific partition masking scenarios on videos of various lengths.
    """
    # Arrange
    captions = captions_of_length(num_clips)
    strategy = PartitionMasking(
        num_partitions=num_partitions,
        start_partition=start_partition,
        num_parts_to_mask=1
    )

    # Act
    # Unpack the tuple returned by the apply method
    masked_clips, returned_indices = strategy.apply(captions)

    # Assert
    # Directly compare the returned indices with the expected ones
    assert returned_indices == expected_indices

    # Optional: A sanity check that the correct clips were indeed masked
    for clip in masked_clips:
        assert clip.index in expected_indices and clip.is_masked() or \
            clip.index not in expected_indices and not clip.is_masked()


def test_partition_masking_on_5_clips(captions_of_length):
    """
    A specific, standalone test for the 5-clip edge case.
    """
    # Arrange
    captions = captions_of_length(5)
    strategy = PartitionMasking(num_partitions=5, start_partition=2, num_parts_to_mask=1)

    # Act
    masked_clips, returned_indices = strategy.apply(captions)

    # Assert
    assert returned_indices == {2}


# --- Passing tests (no changes needed) ---

def test_factory_generates_correct_number_of_strategies_1_2():
    """
    Tests that get_masking_strategies correctly generates the total number
    of strategy instances from a grid search configuration.
    """
    masking_configs = [{"scheme": "partition", "num_partitions": 5, "num_parts_to_mask": [1, 2]}]
    strategies = get_masking_strategies(masking_configs=masking_configs, master_seed=42)
    assert len(strategies) == 9
    assert all(isinstance(s, PartitionMasking) for s in strategies)


def test_factory_generates_correct_number_of_strategies_1_2_3_4():
    """
    Tests that the factory correctly generates the total number of
    strategy instances from a grid search over masks of size 1, 2, 3, and 4.
    """
    masking_configs = [{"scheme": "partition", "num_partitions": 5, "num_parts_to_mask": [1, 2, 3, 4]}]
    strategies = get_masking_strategies(masking_configs, master_seed=42)
    assert len(strategies) == 14
    assert all(isinstance(s, PartitionMasking) for s in strategies)

from reconstruction.masking import ContiguousMasking

def test_contiguous_masking_correctly_masks_indices():
    """
    Tests that the ContiguousMasking strategy works as expected.
    """
    
    # We expect a block of 3 clips to be masked
    strategy = ContiguousMasking(seed=42, width=3)
    
    # Act
    # With a seed of 42 and 10 clips, the random start index will be 1
    masked_indices = strategy.get_indices_to_mask(num_clips=10)
    
    # Assert
    # The masked indices should be a contiguous block of 3, starting at 1
    assert masked_indices == {1, 2, 3}


###################

# This test file is dedicated to testing the FixedFillMasking class.

@pytest.mark.parametrize("width, start_ind, num_clips, expected", [
    # Standard cases
    (5, 3, 10, {1, 2, 3, 4, 5}),
    (4, 5, 10, {4, 5, 6, 7}),

    # Edge cases near the start
    (5, 0, 10, {0, 1, 2, 3, 4}),
    (5, 1, 10, {0, 1, 2, 3, 4}),

    # Edge cases near the end
    (5, 9, 10, {5, 6, 7, 8, 9}),
    (3, 8, 10, {7, 8, 9}),

    # Single item width
    (1, 5, 10, {5}),

    # Full width cases
    (10, 5, 10, set(range(10))),
    (12, 5, 10, set(range(10))),  # Width larger than num_clips
])
def test_get_indices_to_mask(width, start_ind, num_clips, expected):
    """
    Tests the get_indices_to_mask method with various valid inputs.
    """
    strategy = FixedFillMasking(width=width, start_ind=start_ind)
    result = strategy.get_indices_to_mask(num_clips)
    assert result == expected


def test_get_indices_to_mask_invalid_start():
    """
    Tests that the method raises a ValueError if the start_ind is out of bounds.
    """
    strategy = FixedFillMasking(width=5, start_ind=10)
    with pytest.raises(ValueError, match="start_ind (10) must be less than num_clips (10)"):
        strategy.get_indices_to_mask(num_clips=10)


def test_get_indices_to_mask_with_zero_clips():
    """
    Tests behavior when there are no clips available.
    """
    strategy = FixedFillMasking(width=5, start_ind=0)
    with pytest.raises(ValueError):
        strategy.get_indices_to_mask(num_clips=0)  # start_ind=0 is not < num_clips=0

    # A valid case where the result should be empty
    strategy_valid = FixedFillMasking(width=5, start_ind=0)
    assert strategy_valid.get_indices_to_mask(num_clips=1) == {0}
