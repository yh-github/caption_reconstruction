import pandas as pd
from pydantic import BaseModel
from typing import Iterator, Self
import numpy as np
from numpy.typing import NDArray
from pathlib import Path

NPY_FILE_PATTERN = "*.npy"

def find_numpy_files(directory: Path, file_pattern: str = NPY_FILE_PATTERN) -> list[Path]:
    return list(directory.rglob(file_pattern))

def load_numpy_files(
    npy_files: list[Path],
    max_rows:int|None=None
) -> Iterator[tuple[NDArray[np.float64], str]]:
    max_rows = max_rows or 60
    for file_path in npy_files:
        yield np.load(file_path)[:max_rows], file_path.stem

class VectorStats(BaseModel):
    mean:float
    std:float
    min:float
    max:float

    @classmethod
    def from_vector(cls, vector: list[float] | np.ndarray) -> Self:
        if len(vector) == 0:
            return cls(mean=0.0, std=0.0, min=0.0, max=0.0)

        return cls(
            mean=float(np.mean(vector)),
            std=float(np.std(vector)),
            min=float(np.min(vector)),
            max=float(np.max(vector))
        )

def calculate_elementwise_cosine(
    vectors_a: list[list[float]]|NDArray[np.float64],
    vectors_b: list[list[float]]|NDArray[np.float64]
) -> np.ndarray:
    """
    Takes two lists of vectors of the same length, validates them, and computes
    the element-wise cosine similarity between corresponding vectors.

    Args:
        vectors_a: A list of M vectors, where each vector is a list of floats.
        vectors_b: A list of M vectors, where each vector is a list of floats.

    Returns:
        A NumPy ndarray of shape (M,) where the element at index (i) is the
        cosine similarity between the i-th vector in `vectors_a` and the i-th
        vector in `vectors_b`.

    Raises:
        ValueError: If the input lists are empty, have a different number of vectors,
                    are not 2D, or if the inner vectors do not have matching dimensions.
    """
    # --- Step 1: Convert lists to ndarrays ---
    # Using np.float64 for better precision.
    if isinstance(vectors_a, list):
        matrix_a = np.array(vectors_a, dtype=np.float64)
    else:
        matrix_a = vectors_a
    if isinstance(vectors_b, list):
        matrix_b = np.array(vectors_b, dtype=np.float64)
    else:
        matrix_b = vectors_b

    # --- Step 2: Validate dimensions ---
    if matrix_a.ndim != 2 or matrix_b.ndim != 2:
        raise ValueError("Inputs must be convertible to 2D matrices (list of lists).")

    if matrix_a.shape[0] == 0 or matrix_b.shape[0] == 0:
        raise ValueError("Input lists cannot be empty.")

    # New check: ensure both lists have the same number of vectors.
    if matrix_a.shape[0] != matrix_b.shape[0]:
        raise ValueError(
            f"Input lists must have the same number of vectors. Got {matrix_a.shape[0]} and {matrix_b.shape[0]}."
        )

    # The crucial check: the length of the inner vectors must be the same.
    if matrix_a.shape[1] != matrix_b.shape[1]:
        raise ValueError(
            f"Vector dimensions do not match. Got {matrix_a.shape[1]} and {matrix_b.shape[1]}."
        )

    # --- Step 3: Calculate element-wise cosine similarity ---
    # The formula for cosine similarity is: (A · B) / (||A|| * ||B||)
    # We can compute this for all corresponding vector pairs at once.

    # Element-wise product and sum along rows to get dot products
    dot_products = np.sum(matrix_a * matrix_b, axis=1)

    # Calculate L2 norms (magnitudes) for each vector in both matrices
    norms_a = np.linalg.norm(matrix_a, axis=1)
    norms_b = np.linalg.norm(matrix_b, axis=1)

    # Denominator with a small epsilon to prevent division by zero
    epsilon = 1e-8
    denominator = norms_a * norms_b + epsilon

    similarity_vector = dot_products / denominator

    return similarity_vector


def calculate_similarity_stats(
        m: NDArray[np.float64],
        start_index: int,
        end_index: int
) -> VectorStats:
    """Calculate similarity statistics between vectors in a matrix.

    This function computes the cosine similarity between a reference vector and a set of
    comparison vectors. The reference vector is the mean of the vectors at `start_index`
    and `end_index`. The comparison vectors are all vectors in the matrix `m` strictly
    between `start_index` and `end_index`.

    Args:
        m: A 2D numpy array (matrix) where each row is a vector.
        start_index: The index of the first vector for creating the reference mean.
        end_index: The index of the second vector for creating the reference mean.

    Returns:
        A tuple containing four float values:
        (mean similarity, standard deviation, min similarity, max similarity).
        If there are no vectors between the start and end indices, it returns
        (0.0, 0.0, 0.0, 0.0).
    """
    # --- 1. Input Validation and Edge Case Handling ---
    if not (0 <= start_index < end_index - 1 < end_index < len(m)):
        raise IndexError(
            "Indices are out of bounds or invalid. "
            "Ensure 0 <= start_index < end_index < len(m), and end_index-start_index>1"
        )

    comparison_vectors = m[start_index + 1: end_index]

    if comparison_vectors.shape[0] == 0:
        raise Exception('No vectors between start and end')

    # --- 2. Calculate the Reference Vector ---
    # The reference is the mean of the vectors at the start and end indices.
    vec_start = m[start_index]
    vec_end = m[end_index]
    reference_vector = (vec_start + vec_end) / 2.0

    # --- 3. Vectorized Cosine Similarity Calculation ---
    # The formula for cosine similarity is: (A · B) / (||A|| * ||B||)
    # We can compute this for all comparison vectors at once.

    # Calculate dot products of the reference vector against all comparison vectors
    dot_products = np.dot(comparison_vectors, reference_vector)

    # Calculate the L2 norm (magnitude) of the reference vector
    norm_reference = np.linalg.norm(reference_vector)

    # Calculate the L2 norm for each of the comparison vectors (row-wise)
    norm_comparisons = np.linalg.norm(comparison_vectors, axis=1)

    # Calculate the denominator of the cosine similarity formula
    # Add a small epsilon to prevent division by zero if any vector has zero magnitude
    epsilon = 1e-8
    denominator = norm_reference * norm_comparisons + epsilon

    # Compute the array of similarity scores
    similarities = dot_products / denominator

    # --- 4. Calculate and Return Statistics ---

    return VectorStats.from_vector(similarities)

def get_indices(max_ind:int=60) -> list[tuple[int, int]]:
    def inner():
        for start in [0,10,20,30,40,50]:
            for offset in [2, 3, 5, 7, 9]:
                yield start, start + offset
        for start in [10,20,30]:
            for offset in [15, 20, 25]:
                yield start, start + offset

    res = {(start, end) for start, end in inner() if end < max_ind}
    return list(res)

def modular(directory:Path):
    npy_files = find_numpy_files(directory)
    if not npy_files:
        raise Exception(f"No .npy files found in {directory}")

    data = []
    for start, end in get_indices():
        for m, m_id in load_numpy_files(npy_files):
            try:
                d = {'vid_id':m_id, 'start':start, 'end':end}
                stats = calculate_similarity_stats(m, start, end)
                d.update(stats)
                data.append(d)
            except IndexError as e:
                print(f"Error: {e}")

    df = pd.DataFrame(data)
    df['ptp'] = df['max'] - df['min']
    df['width'] = df['end'] - df['start'] - 1
    print(df.head())
    print(len(data), len(get_indices()), len(list(load_numpy_files(npy_files))))
    df.to_csv('results/eval_sim_'+directory.name+'.csv')


def monolith(directory:Path):
    npy_files = find_numpy_files(directory)
    if not npy_files:
        raise Exception(f"No .npy files found in {directory}")

    data = []
    for start, end in get_indices():
        for m, m_id in load_numpy_files(npy_files):
            try:
                d = {'vid_id':m_id, 'start':start, 'end':end}
                stats = calculate_similarity_stats(m, start, end)
                d.update(stats)
                data.append(d)
            except IndexError as e:
                print(f"Error: {e}")

    df = pd.DataFrame(data)
    df['ptp'] = df['max'] - df['min']
    df['width'] = df['end'] - df['start'] - 1
    print(df.head())
    print(len(data), len(get_indices()), len(list(load_numpy_files(npy_files))))
    df.to_csv('results/eval_sim_'+directory.name+'.csv')

Matrix = list[list[float]]|NDArray[np.float64]

class VectorReconstructionEvaluator:
    def evaluate(self, pred_vecs:Matrix, true_vecs:Matrix) -> dict:
        return VectorStats.from_vector(calculate_elementwise_cosine(pred_vecs, true_vecs)).model_dump()

    @staticmethod
    def agg_metrics(all_metrics):
        vs = [VectorStats.model_validate(m) for m in all_metrics]
        means = VectorStats.from_vector([v.mean for v in vs])

        return {
            "num_of_instances": len(all_metrics),
            "mean_mean": means.mean,
            "mean_std": means.std,
            "mean_min": means.min,
            "mean_max": means.max
        }


# import time
# if __name__ == "__main__":
#     start_time = time.time()
#     main(Path("local/wild_videos_embs"))
#     end_time = time.time()
#     elapsed_time = end_time - start_time
#     print(f"Execution time: {elapsed_time:.2f} seconds")


# a=np.array([[11.,2.,3.],[21.,2.,3.],[31.,2.,3.],[41.,2.,3.],[51.,2.,3.],[61.,2.,3.],[71.,2.,3.]])
#
# inds=np.array([0,2,6])
# print(len(a))
# b=a[inds]
# print(b, type(b))