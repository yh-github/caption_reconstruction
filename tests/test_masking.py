from main import apply_masking
from toy_data import create_toy_transcript

def test_apply_masking_random_half():
    """
    Tests the 'random' masking scheme with a 50% ratio.
    Verifies that exactly half of the clips are masked.
    """
    # Arrange: Get our ground truth data and set up a config
    ground_truth = create_toy_transcript()
    num_clips = len(ground_truth)
    
    config = {
        'masking': {
            'ratio': 0.5,
            'scheme': 'random'
        },
        'random_seed': 42 # Use a fixed seed for a predictable outcome
    }

    # Act: Run the function we want to test
    masked_transcript = apply_masking(ground_truth, config)

    # Assert: Check if the outcome is what we expect
    assert len(masked_transcript) == num_clips # Ensure no clips were added or removed

    # Count the number of masked descriptions
    mask_count = sum(1 for clip in masked_transcript if clip.description == "[MASK]")
    
    assert mask_count == num_clips // 2 # Check if exactly half are masked
