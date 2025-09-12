from common_utils.error_handling import UserFacingError
import numpy as np
from abc import ABC, abstractmethod
from numpy.typing import NDArray


class VectorReconstructionStrategy(ABC):
    @abstractmethod
    def reconstruct(self, vectors: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Takes a 2D matrix with missing vectors (represented by all-NaN rows),
        reconstructs them, and returns a dense matrix of only the new vectors.
        """
        pass

    def __str__(self):
        return self.__class__.__name__

    def __repr__(self):
        return self.__str__()


class RepeatClosestVector(VectorReconstructionStrategy):
    """
    Reconstructs a missing vector by finding the single closest known vector
    (by index) and repeating its value.
    """

    @staticmethod
    def _get_closest_indices(vectors: NDArray[np.float64]) -> dict[int, int]:
        """Helper to find the closest known index for each missing index."""
        closest_map = {}
        missing_indices = np.where(np.isnan(vectors).all(axis=1))[0]
        known_indices = np.where(~np.isnan(vectors).all(axis=1))[0]

        if known_indices.size == 0:
            raise ValueError("Cannot reconstruct; the input matrix has no known vectors.")

        for i in missing_indices:
            closest_known_idx_pos = np.argmin(np.abs(known_indices - i))
            closest_map[i] = known_indices[closest_known_idx_pos]
        return closest_map

    def reconstruct(self, vectors: NDArray[np.float64]) -> NDArray[np.float64]:
        closest_indices_map = self._get_closest_indices(vectors)
        if not closest_indices_map:
            return np.array([], dtype=vectors.dtype).reshape(0, vectors.shape[1])

        sorted_missing_indices = sorted(closest_indices_map.keys())
        known_indices_to_use = [closest_indices_map[i] for i in sorted_missing_indices]
        return vectors[known_indices_to_use]

class MeanClosestVectors(VectorReconstructionStrategy):
    """
    Reconstructs a missing vector by finding the closest known "before" and "after"
    vectors and calculating their mean. Handles edge cases gracefully.
    """

    def reconstruct(self, vectors: NDArray[np.float64]) -> NDArray[np.float64]:
        reference_vectors_list = []
        missing_indices = np.where(np.isnan(vectors).all(axis=1))[0]
        known_indices = np.where(~np.isnan(vectors).all(axis=1))[0]

        if known_indices.size == 0:
            raise ValueError("Cannot reconstruct; the input matrix has no known vectors.")

        for i in missing_indices:
            before_indices = known_indices[known_indices < i]
            closest_before = before_indices[-1] if before_indices.size > 0 else None
            after_indices = known_indices[known_indices > i]
            closest_after = after_indices[0] if after_indices.size > 0 else None

            if closest_before is not None and closest_after is not None:
                reference_vectors_list.append((vectors[closest_before] + vectors[closest_after]) / 2.0)
            elif closest_before is not None:
                reference_vectors_list.append(vectors[closest_before])
            elif closest_after is not None:
                reference_vectors_list.append(vectors[closest_after])

        if not reference_vectors_list:
            return np.array([], dtype=vectors.dtype).reshape(0, vectors.shape[1])
        return np.array(reference_vectors_list)


# --- New Similarity Calculation Function ---

# def calculate_reconstruction_similarity(
#     reconstructed_vectors: NDArray[np.float64],
#     ground_truth_vectors: NDArray[np.float64],
#     masked_indices: NDArray[np.int_]
# ) -> NDArray[np.float64]:
#     """
#     Calculates the cosine similarity of reconstructed vectors against their
#     original ground truth counterparts.
#
#     Args:
#         reconstructed_vectors: A dense matrix of the newly created vectors.
#         ground_truth_vectors: The original, complete matrix (without NaNs).
#         masked_indices: The indices of the vectors that were reconstructed.
#
#     Returns:
#         A 1D NumPy array of similarity scores.
#     """
#     if reconstructed_vectors.shape[0] == 0:
#         return np.array([], dtype=np.float64)
#
#     # Select the original vectors that correspond to the reconstructed ones
#     original_subset = ground_truth_vectors[masked_indices]
#
#     return elementwise_cosine_similarity(reconstructed_vectors, original_subset)


class VectorReconstructionStrategyBuilder:

    def get_strategy(self, strategy_config: dict) -> VectorReconstructionStrategy:
        """
        Builds and returns a specific strategy instance based on the config.
        """
        strategy_type = strategy_config.get("type")
        if not strategy_type:
            raise UserFacingError("'type' must be specified in the strategy configuration.")

        if strategy_type == "repeat_closest":
            return RepeatClosestVector()
        elif strategy_type == "mean_closest":
            return MeanClosestVectors()

        else:
            raise NotImplementedError(f"Strategy type '{strategy_type}' is not implemented.")

