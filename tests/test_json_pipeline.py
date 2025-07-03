# tests/test_json_pipeline.py
import json
from main import apply_masking, build_prompt
from toy_data import create_toy_transcript
from constants import DATA_MISSING

def test_apply_masking_replaces_data_payload():
    """
    Tests that the new masking logic correctly replaces the entire 'data'
    field with the DATA_MISSING token.
    """
    # Arrange
    ground_truth = create_toy_transcript()
    config = {'masking': {'ratio': 0.5, 'scheme': 'random'}, 'random_seed': 42}

    # Act
    masked_transcript = apply_masking(ground_truth, config)

    # Assert
    mask_count = sum(1 for clip in masked_transcript if clip.data == DATA_MISSING)
    data_object_count = sum(1 for clip in masked_transcript if clip.data != DATA_MISSING)

    assert mask_count == 5
    assert data_object_count == 5
    assert len(masked_transcript) == len(ground_truth)

def test_build_prompt_creates_valid_json():
    """
    Tests that the prompt builder creates a valid JSON string with the
    correct structure and includes the masked tokens.
    """
    # Arrange
    ground_truth = create_toy_transcript()
    config = {'masking': {'ratio': 0.3, 'scheme': 'random'}, 'random_seed': 10}
    masked_transcript = apply_masking(ground_truth, config)

    # Act
    prompt = build_prompt(masked_transcript, config)
    
    # Find the JSON part of the prompt
    json_part = prompt.split("---\n\n")[1]

    # Assert
    # 1. Check that the output is valid JSON
    data = json.loads(json_part)
    assert isinstance(data, list)
    assert len(data) == 10

    # 2. Check that the masked items are present as the special string
    mask_count_in_json = sum(1 for item in data if item['data'] == DATA_MISSING)
    assert mask_count_in_json == 3 # 10 clips * 0.3 ratio = 3 masks
