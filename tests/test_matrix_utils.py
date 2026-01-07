import sys
import numpy as np
from common_utils.matrix_utils import matrix_to_b64, b64_to_matrix

def test_matrix_encoding():
    print("Testing Matrix Encoding...")
    
    # Random matrix
    original = np.random.rand(5, 5).astype(np.float32)
    
    # Encode
    b64 = matrix_to_b64(original)
    print(f"Base64 length: {len(b64)}")
    
    # Decode
    decoded = b64_to_matrix(b64)
    
    # Verify
    assert np.allclose(original, decoded), "Decoded matrix does not match original"
    assert original.dtype == decoded.dtype, f"Dtypes mismatched: {original.dtype} vs {decoded.dtype}"
    
    print("✅ Matrix Encoding Test Passed")

def test_empty_matrix():
    print("Testing Empty Matrix...")
    b64 = matrix_to_b64(np.array([]))
    decoded = b64_to_matrix(b64)
    assert decoded.size == 0
    print("✅ Empty Matrix Test Passed")

if __name__ == "__main__":
    test_matrix_encoding()
    test_empty_matrix()
