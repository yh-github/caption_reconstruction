import pytest
import numpy as np
from numpy.typing import NDArray
from vectors.efficient_startegies import MeanClosestVectors, RepeatClosestVector


@pytest.mark.parametrize("strategy_class, expected_indices", [
    (MeanClosestVectors, np.array([1, 3, 6, 7, 9])),
    (RepeatClosestVector, np.array([1, 3, 6, 7]))
])
def test_get_required_indices(strategy_class, expected_indices):
    strategy = strategy_class()
    total, missing = 10, np.array([0, 4, 5, 8])
    actual = strategy.get_required_indices(total, missing)
    np.testing.assert_array_equal(actual, expected_indices)


@pytest.mark.parametrize("strategy_class, expected_results", [
    (MeanClosestVectors, {0: [1.0, 1.0], 4: [4.5, 4.5], 5: [4.5, 4.5], 8: [8.0, 8.0]}),
    (RepeatClosestVector, {0: [1.0, 1.0], 4: [3.0, 3.0], 5: [6.0, 6.0], 8: [7.0, 7.0]})
])
def test_reconstruct_from_provided(strategy_class, expected_results):
    strategy = strategy_class()
    total, missing = 10, np.array([0, 4, 5, 8])
    provided_vectors = {
        1: np.array([1.0, 1.0]),
        3: np.array([3.0, 3.0]),
        6: np.array([6.0, 6.0]),
        7: np.array([7.0, 7.0]),
        9: np.array([9.0, 9.0]),
    }
    result = strategy.reconstruct_from_provided(provided_vectors, total, missing)

    assert set(result.keys()) == set(expected_results.keys())
    for key in expected_results:
        np.testing.assert_array_equal(result[key], expected_results[key])


@pytest.mark.parametrize("strategy_class, expected_results", [
    (MeanClosestVectors, {0: [1.0, 1.0], 4: [4.0, 4.0], 8: [8.0, 8.0]}),
    (RepeatClosestVector, {0: [1.0, 1.0], 4: [3.0, 3.0], 8: [7.0, 7.0]})
])
def test_orchestration_flow(strategy_class, expected_results):
    """A full integration test of the decoupled workflow for both strategies."""
    strategy = strategy_class()

    def mock_embedding_function(indices: NDArray[np.int_]) -> dict[int, NDArray[np.float64]]:
        all_vectors = {i: np.array([float(i), float(i)]) for i in range(10)}
        return {i: all_vectors[i] for i in indices}

    reconstructed_vectors = strategy.reconstruct_efficiently(
        total_items=10,
        missing_indices_list=[0, 4, 8],
        embedding_function=mock_embedding_function
    )

    assert set(reconstructed_vectors.keys()) == set(expected_results.keys())
    for key in expected_results:
        np.testing.assert_array_equal(reconstructed_vectors[key], expected_results[key])


@pytest.mark.parametrize("strategy_class, expected_results", [
    (MeanClosestVectors, {3: [4.0, 4.0], 4: [4.0, 4.0], 5: [4.0, 4.0]}),
    (RepeatClosestVector, {3: [2.0, 2.0], 4: [2.0, 2.0], 5: [6.0, 6.0]})
])
def test_orchestration_flow_consecutive(strategy_class, expected_results):
    """Tests the workflow with a block of three consecutively missing vectors."""
    strategy = strategy_class()

    def mock_embedding_function(indices: NDArray[np.int_]) -> dict[int, NDArray[np.float64]]:
        all_vectors = {i: np.array([float(i), float(i)]) for i in range(10)}
        return {i: all_vectors[i] for i in indices}

    reconstructed_vectors = strategy.reconstruct_efficiently(
        total_items=10,
        missing_indices_list=[3, 4, 5],
        embedding_function=mock_embedding_function
    )

    assert set(reconstructed_vectors.keys()) == set(expected_results.keys())
    for key in expected_results:
        np.testing.assert_array_equal(reconstructed_vectors[key], expected_results[key])
