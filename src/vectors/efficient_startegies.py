from typing import Callable
from common_utils.error_handling import UserFacingError
import numpy as np
from abc import ABC, abstractmethod
from numpy.typing import NDArray


class VectorReconstructionStrategy(ABC):
    """Abstract base class for all vector reconstruction strategies."""

    def __str__(self) -> str:
        return self.__class__.__name__

    def __repr__(self) -> str:
        return self.__str__()

    @abstractmethod
    def get_required_indices(self, total_vectors: int, missing_indices: NDArray[np.int_]) -> NDArray[np.int_]:
        """
        Calculates the set of known vector indices required to perform a reconstruction.

        Args:
            total_vectors: The total number of vectors in the conceptual matrix.
            missing_indices: A sorted array of indices that are missing.

        Returns:
            A sorted, unique array of known indices that need to be fetched.
        """
        pass

    @abstractmethod
    def reconstruct_from_provided(
            self,
            provided_vectors: dict[int, NDArray[np.float64]],
            total_vectors: int,
            missing_indices: NDArray[np.int_]
    ) -> dict[int, NDArray[np.float64]]:
        """
        Reconstructs missing vectors using only the provided known vectors.

        Args:
            provided_vectors: A dictionary mapping known indices to their vector data.
            total_vectors: The total number of vectors.
            missing_indices: The indices to reconstruct.

        Returns:
            A dictionary mapping the reconstructed indices to their new vectors.
        """
        pass

    def reconstruct_efficiently(
            self,
            total_items: int,
            missing_indices_list: list[int],
            embedding_function: Callable[[NDArray[np.int_]], dict[int, NDArray[np.float64]]]
    ) -> dict[int, NDArray[np.float64]]:
        """Orchestrates the efficient, decoupled reconstruction process."""
        missing_indices = np.sort(np.array(missing_indices_list, dtype=int))
        required_indices = self.get_required_indices(total_items, missing_indices)
        provided_vectors = embedding_function(required_indices)
        reconstructed = self.reconstruct_from_provided(
            provided_vectors, total_items, missing_indices
        )
        return reconstructed


class MeanClosestVectors(VectorReconstructionStrategy):
    """
    Implements the two-phase reconstruction using the mean of closest neighbors.
    """

    def get_required_indices(self, total_vectors: int, missing_indices: NDArray[np.int_]) -> NDArray[np.int_]:
        required: set[int] = set()
        all_indices = np.arange(total_vectors)
        known_indices = np.setdiff1d(all_indices, missing_indices, assume_unique=True)

        if known_indices.size == 0 and missing_indices.size > 0:
            raise ValueError("Cannot reconstruct; no known vectors exist.")

        for i in missing_indices:
            before_indices = known_indices[known_indices < i]
            if before_indices.size > 0:
                required.add(before_indices[-1])

            after_indices = known_indices[known_indices > i]
            if after_indices.size > 0:
                required.add(after_indices[0])

        return np.sort(np.array(list(required), dtype=int))

    def reconstruct_from_provided(
            self,
            provided_vectors: dict[int, NDArray[np.float64]],
            total_vectors: int,
            missing_indices: NDArray[np.int_]
    ) -> dict[int, NDArray[np.float64]]:
        reconstructed_vectors = {}
        known_provided_indices = np.sort(np.array(list(provided_vectors.keys())))

        for i in missing_indices:
            before_indices = known_provided_indices[known_provided_indices < i]
            closest_before = before_indices[-1] if before_indices.size > 0 else None

            after_indices = known_provided_indices[known_provided_indices > i]
            closest_after = after_indices[0] if after_indices.size > 0 else None

            if closest_before is not None and closest_after is not None:
                reconstructed_vectors[i] = (provided_vectors[closest_before] + provided_vectors[closest_after]) / 2.0
            elif closest_before is not None:
                reconstructed_vectors[i] = provided_vectors[closest_before]
            elif closest_after is not None:
                reconstructed_vectors[i] = provided_vectors[closest_after]

        return reconstructed_vectors


class RepeatClosestVector(VectorReconstructionStrategy):
    """
    Implements the two-phase reconstruction using the single closest neighbor.
    """

    @staticmethod
    def _find_closest_known(known_indices: NDArray[np.int_], target_index: int) -> int:
        """Finds the single closest index in a sorted array of known indices."""
        closest_pos = np.argmin(np.abs(known_indices - target_index))
        return known_indices[closest_pos]

    def get_required_indices(self, total_vectors: int, missing_indices: NDArray[np.int_]) -> NDArray[np.int_]:
        required: set[int] = set()
        all_indices = np.arange(total_vectors)
        known_indices = np.setdiff1d(all_indices, missing_indices, assume_unique=True)

        if known_indices.size == 0 and missing_indices.size > 0:
            raise ValueError("Cannot reconstruct; no known vectors exist.")

        for i in missing_indices:
            closest_known = self._find_closest_known(known_indices, i)
            required.add(closest_known)

        return np.sort(np.array(list(required), dtype=int))

    def reconstruct_from_provided(
            self,
            provided_vectors: dict[int, NDArray[np.float64]],
            total_vectors: int,
            missing_indices: NDArray[np.int_]
    ) -> dict[int, NDArray[np.float64]]:
        reconstructed_vectors = {}
        known_provided_indices = np.sort(np.array(list(provided_vectors.keys())))

        for i in missing_indices:
            closest_known = self._find_closest_known(known_provided_indices, i)
            reconstructed_vectors[i] = provided_vectors[closest_known]

        return reconstructed_vectors

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

