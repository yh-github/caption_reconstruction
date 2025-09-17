import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from typing import List, Tuple
from abc import ABC, abstractmethod

# Define a type alias for clarity
RawData = List[Tuple[np.ndarray, str]]


# --- Feature Extraction Strategies ---

class FeatureExtractionStrategy(ABC):
    """Abstract base class for all matrix-to-vector feature extraction strategies."""

    def __str__(self) -> str:
        # Provides a clean default name for printing, e.g., "MeanStrategy"
        return self.__class__.__name__

    @abstractmethod
    def extract(self, matrix: np.ndarray) -> np.ndarray:
        """Takes a 2D matrix and returns a single 1D feature vector."""
        pass


class MeanStrategy(FeatureExtractionStrategy):
    """Represents the matrix by its mean vector (semantic centroid)."""

    def extract(self, matrix: np.ndarray) -> np.ndarray:
        return matrix.mean(axis=0)


class MaxPoolingStrategy(FeatureExtractionStrategy):
    """Represents the matrix by its component-wise maximum values."""

    def extract(self, matrix: np.ndarray) -> np.ndarray:
        return matrix.max(axis=0)


class TemporalMeanDiffStrategy(FeatureExtractionStrategy):
    """Represents the matrix by the mean of its temporal differences (velocity)."""

    def extract(self, matrix: np.ndarray) -> np.ndarray:
        if len(matrix) < 2:
            return np.zeros(matrix.shape[1])
        diffs = np.diff(matrix, axis=0)
        return diffs.mean(axis=0)


class TemporalMaxPoolingStrategy(FeatureExtractionStrategy):
    """Represents the matrix by the max of its temporal differences (peak change)."""

    def extract(self, matrix: np.ndarray) -> np.ndarray:
        if len(matrix) < 2:
            return np.zeros(matrix.shape[1])
        diffs = np.diff(matrix, axis=0)
        return diffs.max(axis=0)


class TemporalAccelerationMeanStrategy(FeatureExtractionStrategy):
    """Represents the matrix by the mean of its second-order differences (acceleration)."""

    def extract(self, matrix: np.ndarray) -> np.ndarray:
        if len(matrix) < 3:
            return np.zeros(matrix.shape[1])
        # First difference (velocity)
        velocity = np.diff(matrix, axis=0)
        # Second difference (acceleration)
        acceleration = np.diff(velocity, axis=0)
        return acceleration.mean(axis=0)


class ConcatKeyFramesStrategy(FeatureExtractionStrategy):
    """
    Represents the matrix by concatenating vectors from specified key indices.
    """

    def __init__(self, key_indices: List[int] = [0, -1]):
        # Default to first and last frames/sentences
        self.key_indices = key_indices

    def __str__(self) -> str:
        # Override the default __str__ to include the parameters
        return f"{self.__class__.__name__}(key_indices={self.key_indices})"

    def extract(self, matrix: np.ndarray) -> np.ndarray:
        num_rows = len(matrix)
        # Resolve negative indices (like -1 for the last element)
        resolved_indices = [idx if idx >= 0 else num_rows + idx for idx in self.key_indices]
        # Filter out any indices that might be out of bounds after resolution
        valid_indices = [idx for idx in resolved_indices if 0 <= idx < num_rows]

        if not valid_indices:
            # Return a zero vector of the expected flattened size if no valid frames are found
            return np.zeros(matrix.shape[1] * len(self.key_indices))

        key_vectors = [matrix[idx] for idx in valid_indices]
        return np.concatenate(key_vectors)


# --- Main Processing Pipeline ---

def get_all_strategies() -> List[FeatureExtractionStrategy]:
    """
    Automatically finds and instantiates all concrete FeatureExtractionStrategy
    subclasses defined in this file.
    """
    strategies = []
    # This line finds all classes that inherit from our base strategy
    for subclass in FeatureExtractionStrategy.__subclasses__():
        if subclass == ConcatKeyFramesStrategy:
            # For configurable strategies, we can add multiple interesting instances
            strategies.append(ConcatKeyFramesStrategy(key_indices=[0, -1]))
            strategies.append(ConcatKeyFramesStrategy(key_indices=[0, 60 // 2, -1]))
        else:
            # Instantiate all other strategies with their default settings
            strategies.append(subclass())
    return strategies


def generate_mock_data(num_samples: int = 20) -> RawData:
    """Generates a list of tuples, each containing a random matrix and a unique ID."""
    data = []
    for i in range(num_samples):
        matrix = np.random.rand(60, 512)
        sample_id = f"data_point_{i:03d}"
        data.append((matrix, sample_id))
    return data


def process_and_cluster_with_ids(
        raw_data: RawData,
        strategy: FeatureExtractionStrategy,
        n_components: int = 2,
        n_clusters: int = 4
) -> pd.DataFrame:
    """
    Performs the full pipeline using a specified feature extraction strategy.
    """
    print(f"--- Starting data processing pipeline with strategy: {strategy} ---")

    # 1. Feature Extraction and ID Tracking
    feature_vectors = []
    original_ids = []

    # This print statement can be verbose, so it's commented out.
    # print(f"Extracting features from {len(raw_data)} data points...")
    for matrix, sample_id in raw_data:
        feature_vector = strategy.extract(matrix)
        feature_vectors.append(feature_vector)
        original_ids.append(sample_id)

    # 2. Data Preparation
    feature_matrix = np.array(feature_vectors)
    # print(f"Created feature matrix with shape: {feature_matrix.shape}")

    # 3. Dimensionality Reduction
    # print(f"Reducing dimensions from {feature_matrix.shape[1]} to {n_components} using PCA...")
    pca = PCA(n_components=n_components)
    reduced_vectors = pca.fit_transform(feature_matrix)
    # print(f"Created reduced matrix with shape: {reduced_vectors.shape}")

    # 4. Clustering
    # print(f"Clustering the {reduced_vectors.shape[0]} points into {n_clusters} clusters using K-Means...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
    cluster_labels = kmeans.fit_predict(reduced_vectors)

    # 5. Association and Final Output
    # print("Associating results with original IDs...")
    results_df = pd.DataFrame({
        'id': original_ids,
        'x_reduced': reduced_vectors[:, 0],
        'y_reduced': reduced_vectors[:, 1],
        'cluster_label': cluster_labels
    })

    print("--- Pipeline complete ---")
    return results_df


if __name__ == "__main__":
    # Generate some sample data
    my_raw_data = generate_mock_data(num_samples=50)

    # Automatically get all defined strategies
    all_strategies = get_all_strategies()

    # --- Run the pipeline for every strategy to compare results ---
    for strategy_instance in all_strategies:
        results = process_and_cluster_with_ids(
            my_raw_data,
            strategy=strategy_instance,
            n_components=2,
            n_clusters=5
        )
        print(f"\n--- Results using {strategy_instance} ---")
        print(results.head(5))
        print("\n" + "=" * 70)
