# tests/test_masking.py
import pytest
import random
from masking import get_masking_strategy, RandomMasking, ContiguousControlledMasking, ContiguousRandomMasking, SystematicMasking
from data_models import CaptionedClip, NarrativeOnlyPayload
from constants import DATA_MISSING
from data_loaders import ToyDataLoader

@pytest.fixture(scope="module")
def toy_captions():
    loader = ToyDataLoader("datasets/toy_dataset/data.json")
    return loader.load()[0].clips

def test_factory_creates_correct_strategy_types():
    """Tests that the factory function returns the correct strategy objects."""
    # Arrange
    configs = {
        "random": {"scheme": "random", "ratio": 0.5},
        "contiguous_controlled": {"scheme": "contiguous_controlled", "mask_index": 1, "num_to_mask": 1},
        "contiguous_random": {"scheme": "contiguous_random", "ratio": 0.5},
        "systematic": {"scheme": "systematic", "ratio": 0.3}
    }

    # Act & Assert
    # We now pass the specific masking_config dictionary to the factory.
    strategy_r = get_masking_strategy(configs["random"], seed=42)
    assert isinstance(strategy_r, RandomMasking)

    strategy_cc = get_masking_strategy(configs["contiguous_controlled"], seed=42)
    assert isinstance(strategy_cc, ContiguousControlledMasking)

    strategy_cr = get_masking_strategy(configs["contiguous_random"], seed=42)
    assert isinstance(strategy_cr, ContiguousRandomMasking)

    strategy_s = get_masking_strategy(configs["systematic"], seed=42)
    assert isinstance(strategy_s, SystematicMasking)

def test_contiguous_controlled_masking(toy_captions):
    masking_config = {'scheme': 'contiguous_controlled', 'mask_index': 3, 'num_to_mask': 4}
    strategy = get_masking_strategy(masking_config, seed=1)
    masked = strategy.apply(toy_captions)
    masked_indices = {i for i, c in enumerate(masked) if c.data == DATA_MISSING}
    assert masked_indices == {3, 4, 5, 6}

def test_systematic_masking_is_deterministic(toy_captions):
    masking_config = {'scheme': 'systematic', 'ratio': 0.3}
    strategy = get_masking_strategy(masking_config, seed=1)
    masked = strategy.apply(toy_captions)
    masked_indices = {i for i, c in enumerate(masked) if c.data == DATA_MISSING}
    assert masked_indices == {0, 3, 6, 9}

def test_masking_zero_ratio(toy_captions):
    masking_config = {'scheme': 'random', 'ratio': 0.0}
    strategy = get_masking_strategy(masking_config, seed=42)
    masked = strategy.apply(toy_captions)
    mask_count = sum(1 for c in masked if c.data == DATA_MISSING)
    assert mask_count == 0

def test_masking_full_ratio(toy_captions):
    masking_config = {'scheme': 'random', 'ratio': 1.0}
    strategy = get_masking_strategy(masking_config, seed=42)
    masked = strategy.apply(toy_captions)
    mask_count = sum(1 for c in masked if c.data == DATA_MISSING)
    assert mask_count == len(toy_captions)

def test_masking_empty_caption_list():
    masking_config = {'scheme': 'random', 'ratio': 0.5}
    strategy = get_masking_strategy(masking_config, seed=42)
    masked = strategy.apply([])
    assert masked == []
