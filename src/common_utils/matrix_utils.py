import base64
import numpy as np
import io

def matrix_to_b64(matrix: np.ndarray) -> str:
    """
    Encodes a numpy matrix (float32/64) to a Base64 string.
    Uses .npy format internally to preserve shape and dtype.
    """
    if not isinstance(matrix, np.ndarray):
        matrix = np.array(matrix)
        
    # Use BytesIO to capture the npy binary data
    with io.BytesIO() as f:
        np.save(f, matrix)
        f.seek(0)
        return base64.b64encode(f.read()).decode('utf-8')

def b64_to_matrix(b64_str: str) -> np.ndarray:
    """
    Decodes a Base64 string back to a numpy matrix.
    Expects the string to be encoded via matrix_to_b64 (npy format).
    """
    if not b64_str:
        return np.array([])
        
    bytes_data = base64.b64decode(b64_str)
    with io.BytesIO(bytes_data) as f:
        return np.load(f)
