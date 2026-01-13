
import json
import numpy as np
import base64
import sys

# Copying matrix_from_b64 logic from common_utils.matrix_utils (assuming it uses standard numpy packing)
def matrix_from_b64(b64_str: str) -> np.ndarray:
    try:
        # It seems the codebase uses a custom encoding or just pickle/numpy?
        # Let's try standard decoding if it's just raw bytes of a numpy array
        # or check how eval_vectors.py imports it.
        # eval_vectors.py imports matrix_to_b64 from common_utils.matrix_utils
        # Let's peek at that file? 
        # Or just try to decode assuming it's a pickled numpy array or buffer.
        
        # Step 449 output says:
        # "similarity_matrix_b64": "k05VTVBZAQ..." (starts with numpy header)
        import io
        bytes_data = base64.b64decode(b64_str)
        with io.BytesIO(bytes_data) as f:
            return np.load(f, allow_pickle=True)
    except Exception as e:
        print(f"Decode error: {e}")
        return None

path = "results/recon/manual_download/reconstruction/wild_dev_sim_text/phi-3__t=1.5_rp=1.2__fixed_fill(w=3, i=0)/Bertram-Craft_2-clip-3.json"

with open(path, 'r') as f:
    data = json.load(f)

# Keys are indices from 'reconstructed_captions'
keys = sorted([int(k) for k in data.get('reconstructed_captions', {}).keys() if k.isdigit()])
print(f"Indices: {keys}")

b64 = data['metrics']['similarity_matrix_b64']
mat = matrix_from_b64(b64)

if mat is not None:
    print(f"Matrix shape: {mat.shape}")
    print("Matrix snippet:")
    print(mat)
    
    # Calculate R@1 normally to verify
    # Row i is predicted, Col j is truth (or vice versa? Logic says distractor pool matches cols)
    # Reconstructed vectors (preds) are rows?
    # In eval_vectors.py: sim_dist = np.dot(pred, truth.T) -> Shape (M, M)
    # Row i = Pred i measures against Truth 0, Truth 1, Truth 2...
    # So element (i, j) is score of Pred i vs Truth j.
    # Truth for Pred i is Truth i.
    # So we want (i, i) to be max in row i.
    
    preds = np.argmax(mat, axis=1)
    print(f"Predicted best indices per row: {preds}")
    
    # Temporal Recal @ 1 (Window 1)
    # |True - Pred| <= 1
    # Actually, we don't look at indices of argmax.
    # The 'columns' are the truth pool indices.
    # If keys are [0, 1, 2], then col 0 is index 0.
    # If Pred 0 matches best with Col 1, it thinks it's index 1.
    # |0 - 1| = 1 <= 1 -> Success.
    
    success = 0
    for i, p_idx in enumerate(preds):
        true_idx = i # Since we assume 1-to-1 alignment in the subset
        dist = abs(true_idx - p_idx)
        print(f"query {i} (index {keys[i]}): best match col {p_idx} (index {keys[p_idx]}), dist={dist}")
        if dist <= 1:
            success += 1
            
    print(f"Temporal R@1 (w=1): {success/len(preds)}")
