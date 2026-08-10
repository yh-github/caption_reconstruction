from unittest.mock import patch
import pytest
from common_utils.device_setup import calculate_optimal_batch_size

def test_dynamic_batching_non_cuda():
    with patch("common_utils.device_setup.is_cuda", return_value=False):
        # Non-CUDA returns requested_batch_size
        assert calculate_optimal_batch_size(8) == 8
        assert calculate_optimal_batch_size(1) == 1

def test_dynamic_batching_cuda_scaling():
    with patch("common_utils.device_setup.is_cuda", return_value=True):
        # Mock 16GB free VRAM -> 16 // 1.5 = 10 safe batch size
        with patch("torch.cuda.mem_get_info", return_value=(16 * 1024**3, 16 * 1024**3)):
            assert calculate_optimal_batch_size(16) == 10
            assert calculate_optimal_batch_size(5) == 5

        # Mock 4GB free VRAM -> 4 // 1.5 = 2 safe batch size
        with patch("torch.cuda.mem_get_info", return_value=(4 * 1024**3, 16 * 1024**3)):
            assert calculate_optimal_batch_size(16) == 2
            assert calculate_optimal_batch_size(1) == 1

        # Mock < 1.5GB free VRAM -> default_min = 1
        with patch("torch.cuda.mem_get_info", return_value=(1 * 1024**3, 16 * 1024**3)):
            assert calculate_optimal_batch_size(16) == 1
