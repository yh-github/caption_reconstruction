import pytest
from masking import PartitionMasking, get_masking_strategies
from data_models import CaptionedClip, NarrativeOnlyPayload
from constants import DATA_MISSING

# --- The Fixture (no longer parameterized) ---
@pytest.fixture
def captions_of_length(request):
    """
    A factory fixture that creates a list of CaptionedClip objects
    of a specified length.
    """
    def _create_captions(num_clips):
        return [CaptionedClip(timestamp=i+1, data=NarrativeOnlyPayload(description=f"Clip {i+1}")) for i in range(num_clips)]
    return _create_captions

# --- The New, Specific Tests ---

@pytest.mark.parametrize(
    "num_clips, num_partitions, start_partition, expected_indices",
    [
        (10, 5, 1, {2, 3}),      # Test 1: 10 clips, 5 parts, mask 2nd part
        (7, 3, 2, {5, 6}),       # Test 2: 7 clips, 3 parts, mask 3rd part
        (20, 10, 8, {16, 17}),   # Test 3: 20 clips, 10 parts, mask 9th part
    ]
)
def test_partition_masking_scenarios(captions_of_length, num_clips, num_partitions, start_partition, expected_indices):
    """
    Tests specific partition masking scenarios on videos of various lengths.
    Each row in the @pytest.mark.parametrize decorator will run as a separate test.
    """
    # Arrange
    # "Call" the fixture with the specific parameter for this test run
    captions = captions_of_length(num_clips)
    strategy = PartitionMasking(
        num_partitions=num_partitions,
        start_partition=start_partition,
        num_parts_to_mask=1
    )

    # Act
    masked = strategy.apply(captions)
    masked_indices = {i for i, c in enumerate(masked) if c.data == DATA_MISSING}

    # Assert
    assert masked_indices == expected_indices

def test_partition_masking_on_5_clips(captions_of_length):
    """
    A specific, standalone test for the 5-clip edge case.
    """
    # Arrange
    captions = captions_of_length(5)
    strategy = PartitionMasking(num_partitions=5, start_partition=2, num_parts_to_mask=1)

    # Act
    masked = strategy.apply(captions)
    masked_indices = {i for i, c in enumerate(masked) if c.data == DATA_MISSING}

    # Assert
    assert masked_indices == {2}

def test_factory_generates_correct_number_of_strategies_1_2():
    """
    Tests that get_masking_strategies correctly generates the total number
    of strategy instances from a grid search configuration.
    """
    # Arrange: This config defines a grid search over two partition sizes (1 and 2)
    masking_configs = [{
        "scheme": "partition",
        "num_partitions": 5,
        "num_parts_to_mask": [1, 2] # Test two different mask sizes
    }]

    # Act
    # We pass a sample video length, as the factory is data-aware.
    strategies = get_masking_strategies(
        masking_configs=masking_configs,
        master_seed=42
    )

    # Assert
    # For a 5-partition system:
    # - Masks of size 1 have 5 possible start positions (0-4).
    # - Masks of size 2 have 4 possible start positions (0-3).
    # Total = 5 + 4 = 9 strategies should be generated.
    assert len(strategies) == 9
    assert all(isinstance(s, PartitionMasking) for s in strategies)


def test_factory_generates_correct_number_of_strategies_1_2_3_4():
    """
    Tests that the factory correctly generates the total number of
    strategy instances from a grid search over masks of size 1, 2, 3, and 4.
    """
    # Arrange
    masking_configs = [{
        "scheme": "partition",
        "num_partitions": 5,
        "num_parts_to_mask": [1, 2, 3, 4]
    }]

    # Act
    strategies = get_masking_strategies(masking_configs, master_seed=42)

    # Assert
    # For a 5-partition system:
    # - Masks of size 1 have 5 possible starts.
    # - Masks of size 2 have 4 possible starts.
    # - Masks of size 3 have 3 possible starts.
    # - Masks of size 4 have 2 possible starts.
    # Total = 5 + 4 + 3 + 2 = 14 strategies should be generated.
    assert len(strategies) == 14
    assert all(isinstance(s, PartitionMasking) for s in strategies)

