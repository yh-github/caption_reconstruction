import pytest
from masking import PartitionMasking, get_masking_strategies
from data_models import CaptionedClip, NarrativeOnlyPayload
from constants import DATA_MISSING

# --- The Fixture (no changes needed) ---
@pytest.fixture
def captions_of_length():
    """
    A factory fixture that creates a list of CaptionedClip objects
    of a specified length.
    """
    def _create_captions(num_clips):
        return [
            CaptionedClip(timestamp=i+1, data=NarrativeOnlyPayload(description=f"Clip {i+1}"))
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
    for i, clip in enumerate(masked_clips):
        if i in expected_indices:
            assert clip.data == DATA_MISSING
        else:
            assert clip.data != DATA_MISSING


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
