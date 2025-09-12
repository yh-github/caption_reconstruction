import pytest
import numpy as np
from numpy.typing import NDArray
from reconstruction.vector_reconstruction import MeanClosestVectors, RepeatClosestVector


@pytest.fixture
def sample_vectors_with_nans() -> NDArray[np.float64]:
    """Provides a sample matrix with NaNs for testing reconstruction."""
    return np.array([
        [np.nan, np.nan],  # 0: Missing
        [1.0, 1.0],  # 1: Known
        [2.0, 2.0],  # 2: Known
        [3.0, 3.0],  # 3: Known
        [np.nan, np.nan],  # 4: Missing
        [np.nan, np.nan],  # 5: Missing
        [6.0, 6.0],  # 6: Known
        [7.0, 7.0],  # 7: Known
        [np.nan, np.nan],  # 8: Missing
        [9.0, 9.0],  # 9: Known
    ], dtype=np.float64)


@pytest.fixture
def ground_truth_vectors() -> NDArray[np.float64]:
    """Provides the complete matrix without any NaNs for ground truth comparison."""
    return np.array([
        [0.0, 1.0],  # 0: Ground truth for missing
        [1.0, 1.0],  # 1: Known
        [2.0, 2.0],  # 2: Known
        [3.0, 3.0],  # 3: Known
        [4.0, 4.0],  # 4: Ground truth for missing
        [5.0, 5.0],  # 5: Ground truth for missing
        [6.0, 6.0],  # 6: Known
        [7.0, 7.0],  # 7: Known
        [8.0, 8.0],  # 8: Ground truth for missing
        [9.0, 9.0],  # 9: Known
    ], dtype=np.float64)


def test_repeat_closest_vector_reconstruction(sample_vectors_with_nans):
    """Tests the RepeatClosestVector strategy returns a correct dense matrix."""
    strategy = RepeatClosestVector()
    result = strategy.reconstruct(sample_vectors_with_nans)
    assert result.shape == (4, 2)
    expected = np.array([
        [1.0, 1.0],  # For index 0, uses vector at index 1
        [3.0, 3.0],  # For index 4, uses vector at index 3
        [6.0, 6.0],  # For index 5, uses vector at index 6
        [7.0, 7.0],  # For index 8, uses vector at index 7 (tie-break)
    ])
    np.testing.assert_array_equal(result, expected)


def test_mean_closest_vectors_reconstruction(sample_vectors_with_nans):
    """Tests the MeanClosestVectors strategy returns a correct dense matrix."""
    strategy = MeanClosestVectors()
    result = strategy.reconstruct(sample_vectors_with_nans)
    assert result.shape == (4, 2)
    expected = np.array([
        [1.0, 1.0],  # Edge case: only has 'after' neighbor at index 1
        [4.5, 4.5],  # Mean of vectors at index 3 and 6
        [4.5, 4.5],  # Mean of vectors at index 3 and 6
        [8.0, 8.0],  # Edge case: only has 'after' neighbor at index 9
    ])
    np.testing.assert_array_equal(result, expected)


def test_reconstruction_fails_with_no_known_vectors():
    """Tests that a ValueError is raised if the input matrix is all NaNs."""
    all_nan_vectors = np.full((5, 2), np.nan, dtype=np.float64)
    strategy = MeanClosestVectors()
    with pytest.raises(ValueError, match="Cannot reconstruct; the input matrix has no known vectors."):
        strategy.reconstruct(all_nan_vectors)


# def test_reconstruction_similarity_calculation(sample_vectors_with_nans, ground_truth_vectors):
#     """
#     Tests the similarity calculation by comparing reconstructed vectors against
#     a ground truth matrix.
#     """
#     strategy = MeanClosestVectors()
#     masked_indices = np.where(np.isnan(sample_vectors_with_nans).all(axis=1))[0]
#
#     reconstructed_matrix = strategy.reconstruct(sample_vectors_with_nans)
#
#     similarity_scores = calculate_reconstruction_similarity(
#         reconstructed_matrix,
#         ground_truth_vectors,
#         masked_indices
#     )
#
#     assert similarity_scores.shape == (4,)
#
#     # Manually calculate expected similarities for MeanClosestVectors:
#     # Reconstructed: [1,1], [4.5,4.5], [4.5,4.5], [9,9]
#     # Ground Truth: [0,1], [4,4], [5,5], [8,8]
#     expected_vec1_sim = elementwise_cosine_similarity(np.array([[1.0, 1.0]]), np.array([[0.0, 1.0]]))[0]  # ~0.707
#     expected_vec2_sim = elementwise_cosine_similarity(np.array([[4.5, 4.5]]), np.array([[4.0, 4.0]]))[0]  # 1.0
#     expected_vec3_sim = elementwise_cosine_similarity(np.array([[4.5, 4.5]]), np.array([[5.0, 5.0]]))[0]  # 1.0
#     expected_vec4_sim = elementwise_cosine_similarity(np.array([[9.0, 9.0]]), np.array([[8.0, 8.0]]))[0]  # 1.0
#
#     expected_scores = np.array([expected_vec1_sim, expected_vec2_sim, expected_vec3_sim, expected_vec4_sim])
#     np.testing.assert_allclose(similarity_scores, expected_scores, atol=1e-7)
#
