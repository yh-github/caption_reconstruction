# tests/test_pipeline.py
import json
from pipeline import apply_masking, build_prompt
from toy_data import create_toy_transcript
from constants import DATA_MISSING

def test_apply_masking_replaces_data_payload():
    """
    Tests that the masking logic correctly replaces the 'data' field.
    """
    # Arrange
    ground_truth = create_toy_transcript()
    config = {'masking': {'ratio': 0.5, 'scheme': 'random'}, 'random_seed': 42}

    # Act
    masked_transcript = apply_masking(ground_truth, config)

    # Assert
    mask_count = sum(1 for clip in masked_transcript if clip.data == DATA_MISSING)
    assert mask_count == 5

def test_build_prompt_creates_valid_json():
    """
    Tests that the prompt builder creates a valid JSON string.
    """
    # Arrange
    ground_truth = create_toy_transcript()
    config = {'masking': {'ratio': 0.3, 'scheme': 'random'}, 'random_seed': 10}
    masked_transcript = apply_masking(ground_truth, config)

    # Act
    # The config argument was missing here. Now it's added.
    prompt = build_prompt(masked_transcript, config)
    
    # Assert
    json_part = prompt.split("---\n\n")[1]
    data = json.loads(json_part)
    assert isinstance(data, list)
    assert len(data) == 10
    mask_count_in_json = sum(1 for item in data if item['data'] == DATA_MISSING)
    assert mask_count_in_json == 3

def test_apply_masking_contiguous():
    """Tests that contiguous masking creates a single block of masked clips."""
    ground_truth = create_toy_transcript()
    config = {'masking': {'ratio': 0.3, 'scheme': 'contiguous'}, 'random_seed': 5}
    
    masked_transcript = apply_masking(ground_truth, config)
    
    # Find the indices of the masked clips
    masked_indices = [i for i, clip in enumerate(masked_transcript) if clip.data == DATA_MISSING]
    
    assert len(masked_indices) == 3
    # Check if the indices are a continuous block (e.g., [4, 5, 6])
    assert masked_indices[-1] - masked_indices[0] == len(masked_indices) - 1

def test_apply_masking_systematic():
    """Tests that systematic masking creates a regular pattern."""
    ground_truth = create_toy_transcript()
    # Ratio of 0.3 on 10 clips means 3 masks, so step should be 10 // 3 = 3
    config = {'masking': {'ratio': 0.3, 'scheme': 'systematic'}, 'random_seed': 1}
    
    masked_transcript = apply_masking(ground_truth, config)
    
    masked_indices = [i for i, clip in enumerate(masked_transcript) if clip.data == DATA_MISSING]
    
    assert len(masked_indices) > 0
    # Check that the difference between consecutive masked indices is constant
    steps = [masked_indices[i] - masked_indices[i-1] for i in range(1, len(masked_indices))]
    assert all(s == steps[0] for s in steps)

# Edge cases

def test_apply_masking_zero_ratio():
    """Tests that a masking ratio of 0.0 results in no masked clips."""
    ground_truth = create_toy_transcript()
    config = {'masking': {'ratio': 0.0, 'scheme': 'random'}, 'random_seed': 42}
    
    masked_transcript = apply_masking(ground_truth, config)
    
    mask_count = sum(1 for clip in masked_transcript if clip.data == DATA_MISSING)
    assert mask_count == 0
    assert len(masked_transcript) == len(ground_truth)

def test_apply_masking_full_ratio():
    """Tests that a masking ratio of 1.0 results in all clips being masked."""
    ground_truth = create_toy_transcript()
    config = {'masking': {'ratio': 1.0, 'scheme': 'random'}, 'random_seed': 42}
    
    masked_transcript = apply_masking(ground_truth, config)
    
    mask_count = sum(1 for clip in masked_transcript if clip.data == DATA_MISSING)
    assert mask_count == len(ground_truth)

def test_apply_masking_empty_transcript():
    """Tests that the function handles an empty transcript gracefully."""
    ground_truth = []
    config = {'masking': {'ratio': 0.5, 'scheme': 'random'}, 'random_seed': 42}
    
    masked_transcript = apply_masking(ground_truth, config)
    
    assert masked_transcript == []

