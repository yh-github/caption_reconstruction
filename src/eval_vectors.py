import pandas as pd
from pydantic import BaseModel
from typing import Iterator
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

class SimStats(BaseModel):
    mean:float
    std:float
    min:float
    max:float

def calculate_similarity_stats(
        m: NDArray[np.float64],
        start_index: int,
        end_index: int
) -> SimStats:
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

    return SimStats(
        mean=float(np.mean(similarities)),
        std=float(np.std(similarities)),
        min=float(np.min(similarities)),
        max=float(np.max(similarities))
    )

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

def main(directory:Path):
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

import time
if __name__ == "__main__":
    start_time = time.time()
    main(Path("local/wild_videos_embs"))
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Execution time: {elapsed_time:.2f} seconds")

