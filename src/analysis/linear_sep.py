import numpy as np
from sklearn.svm import SVC
from numpy.typing import NDArray


def calculate_separability_score(
        all_vectors: NDArray[np.float64],
        group1_indices: list[int] | NDArray[np.int_],
        group2_indices: list[int] | NDArray[np.int_]
) -> float:
    """
    Calculates a "signed margin" score for the linear separability of two groups
    of vectors within a larger matrix.

    Args:
        all_vectors: The full 2D array of feature data.
        group1_indices: A list or array of indices for rows belonging to the first group.
        group2_indices: A list or array of indices for rows belonging to the second group.

    Returns:
        A float representing the separability score (positive for separable, negative for overlap).
    """
    # 1. Assemble the subset of data we care about for this comparison.
    indices = np.concatenate([group1_indices, group2_indices])
    X = all_vectors[indices]

    # Create the corresponding labels array (0 for group 1, 1 for group 2)
    y = np.array([0] * len(group1_indices) + [1] * len(group2_indices), dtype=int)

    if X.shape[0] < 2:
        # Not enough data to train a separator
        return 0.0

    # 2. Train a linear SVM. C=1e9 approximates a hard-margin SVM.
    svm = SVC(kernel='linear', C=1e9)
    svm.fit(X, y)

    # 3. Get the signed distance of each point from the hyperplane.
    scores = svm.decision_function(X)

    # 4. Check for misclassifications.
    positive_class_label = 1
    misclassified_mask = np.zeros_like(y, dtype=bool)
    misclassified_mask[y == positive_class_label] = scores[y == positive_class_label] < 0
    misclassified_mask[y != positive_class_label] = scores[y != positive_class_label] > 0

    # 5. Calculate the score based on whether the data is separable.
    if np.any(misclassified_mask):
        # Case B: Inseparable. Score is the negative distance of the worst offender.
        misclassified_scores = scores[misclassified_mask]
        separability_score = -np.max(np.abs(misclassified_scores))
    else:
        # Case A: Separable. Score is the margin.
        separability_score = np.min(np.abs(scores))

    return separability_score


