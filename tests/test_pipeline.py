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
