import sys
from pathlib import Path
import re
import yaml

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from abc import ABC, abstractmethod
from data.vector_dataloaders import VectorDataLoader
from experiment_executor.config_loader import load_config

# Define a type alias for clarity
RawData = list[tuple[np.ndarray, str]]

SEED = 0x5EED

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

    def __init__(self, key_indices: list[int]|None):
        # Default to first and last frames/sentences
        self.key_indices = key_indices or [0, -1]

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


# --- Dimensionality Reduction Strategies ---

class DimReductionStrategy(ABC):
    """Abstract base class for all dimensionality reduction strategies."""

    def __str__(self) -> str:
        return self.__class__.__name__

    @abstractmethod
    def reduce(self, matrix: np.ndarray) -> np.ndarray:
        """Takes a 2D feature matrix and returns a new 2D matrix with fewer columns."""
        pass


class PCAStrategy(DimReductionStrategy):
    """Reduces dimensions using Principal Component Analysis (linear)."""

    def __init__(self, n_components: int = 2):
        self.n_components = n_components

    def __str__(self) -> str:
        return f"PCA(n_components={self.n_components})"

    def reduce(self, matrix: np.ndarray) -> np.ndarray:
        pca = PCA(n_components=self.n_components)
        return pca.fit_transform(matrix)


class TSNEStrategy(DimReductionStrategy):
    """Reduces dimensions using t-SNE (non-linear)."""

    def __init__(self, n_components: int = 2, perplexity: int = 30):
        self.n_components = n_components
        self.perplexity = perplexity

    def __str__(self) -> str:
        return f"TSNE(n_components={self.n_components}, perplexity={self.perplexity})"

    def reduce(self, matrix: np.ndarray) -> np.ndarray:
        # For small datasets, perplexity must be less than the number of samples
        effective_perplexity = min(self.perplexity, len(matrix) - 1)
        tsne = TSNE(n_components=self.n_components, perplexity=effective_perplexity, random_state=SEED)
        return tsne.fit_transform(matrix)


# --- Main Processing Pipeline ---

def get_all_feature_strategies() -> list[FeatureExtractionStrategy]:
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


def sanitize_for_filename(text: str) -> str:
    """
    Sanitizes a string to be a valid and readable filename by removing or
    replacing characters that are problematic for file systems.
    """
    # Replace common separators and problematic characters with underscores
    s = str(text)
    s = re.sub(r'[=,\s]', '_', s)
    # Remove any character that is not a letter, number, underscore, hyphen, or parenthesis
    s = re.sub(r'[^\w\(\)_\[\]-]', '', s)
    # Tidy up by replacing multiple underscores with a single one
    s = re.sub(r'__+', '_', s)
    return s


def extract_and_reduce(
    raw_data: RawData,
    feature_strategy: FeatureExtractionStrategy,
    reduction_strategy: DimReductionStrategy
) -> pd.DataFrame:
    """
    Performs feature extraction and dimensionality reduction, returning a DataFrame
    with the original IDs and the new low-dimensional coordinates.
    """
    print(f"--- Processing with: {feature_strategy} -> {reduction_strategy} ---")

    # 1. Feature Extraction and ID Tracking
    feature_vectors = [feature_strategy.extract(matrix) for matrix, _ in raw_data]
    original_ids = [sample_id for _, sample_id in raw_data]

    # 2. Data Preparation
    feature_matrix = np.array(feature_vectors)

    # 3. Dimensionality Reduction
    reduced_vectors = reduction_strategy.reduce(feature_matrix)

    # 4. Association and Final Output
    return pd.DataFrame({
        'id': original_ids,
        'x': reduced_vectors[:, 0],
        'y': reduced_vectors[:, 1]
    })


if __name__ == "__main__":
    # Setup
    config = load_config(sys.argv[1])
    data_config = config["data_config"]
    SEED = config["base_params"]["master_seed"] + data_config.get("seed", 0)

    dataloader = VectorDataLoader.from_config(data_config)
    dataset_name = Path(data_config["path"]).stem

    # my_raw_data = generate_mock_data(num_samples=50)
    my_raw_data = list(dataloader.load())

    results_dir = Path("results/pooling/")/dataset_name
    results_dir.mkdir(parents=True, exist_ok=True)

    # Get all strategy instances to iterate over
    feature_strategies = get_all_feature_strategies()
    dim_reduction_strategies = [PCAStrategy(n_components=2), TSNEStrategy(n_components=2)]

    # --- Run the full pipeline for every combination of strategies ---
    for feat_strat in feature_strategies:
        for dim_red_strat in dim_reduction_strategies:
            # 1. Generate the data
            results_df = extract_and_reduce(
                my_raw_data,
                feature_strategy=feat_strat,
                reduction_strategy=dim_red_strat
            )

            # 2. Create descriptive filenames and metadata using the new helper
            feat_name = sanitize_for_filename(str(feat_strat))
            dim_red_name = sanitize_for_filename(str(dim_red_strat))

            base_filename = f"{feat_name}_{dim_red_name}"
            csv_path = results_dir / f"{base_filename}.csv"
            yaml_path = results_dir / f"{base_filename}_meta.yaml"

            metadata = {
                "dataset_name": dataset_name,
                "feature_extraction_strategy": str(feat_strat),
                "dimensionality_reduction_strategy": str(dim_red_strat),
                "num_samples": len(my_raw_data),
                "feature_dimension": results_df.shape[1] - 1
            }

            # 3. Save the results to disk
            results_df.to_csv(csv_path, index=False)
            with yaml_path.open('w') as f:
                yaml.dump(metadata, f, indent=2, sort_keys=False)

            print(f"    -> Saved results to {csv_path} and {yaml_path}")
            print("-" * 50)
