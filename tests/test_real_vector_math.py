import unittest
import numpy as np
from evaluations.eval_vectors import calculate_elementwise_cosine, context_projection, calculate_similarity_stats

class TestRealVectorMath(unittest.TestCase):
    def test_calculate_elementwise_cosine(self):
        # 1. Identical vectors -> 1.0
        vecs_a = np.array([[1.0, 0.0], [0.0, 1.0]])
        vecs_b = np.array([[1.0, 0.0], [0.0, 1.0]])
        sims = calculate_elementwise_cosine(vecs_a, vecs_b)
        np.testing.assert_array_almost_equal(sims, np.array([1.0, 1.0]))

        # 2. Opposite vectors -> -1.0
        vecs_c = np.array([[-1.0, 0.0]])
        vecs_d = np.array([[1.0, 0.0]])
        sims = calculate_elementwise_cosine(vecs_c, vecs_d)
        np.testing.assert_array_almost_equal(sims, np.array([-1.0]))

        # 3. Orthogonal vectors -> 0.0
        vecs_e = np.array([[1.0, 0.0]])
        vecs_f = np.array([[0.0, 1.0]])
        sims = calculate_elementwise_cosine(vecs_e, vecs_f)
        np.testing.assert_array_almost_equal(sims, np.array([0.0]))
        
        # 4. Dimension mismatch check
        with self.assertRaises(ValueError):
            calculate_elementwise_cosine(np.array([[1,2]]), np.array([[1,2,3]]))
            
        # 5. Length mismatch check
        with self.assertRaises(ValueError):
            calculate_elementwise_cosine(np.array([[1,2]]), np.array([[1,2], [3,4]]))

    def test_context_projection(self):
        # 1. Project vector onto itself -> Residual should be 0 (or close)
        matrix = np.array([[2.0, 2.0]])
        mean_vec = np.array([1.0, 1.0]) # Same direction
        residual = context_projection(matrix, mean_vec)
        # Projection of [2,2] onto [1,1] is [2,2]. Residual = [0,0]
        np.testing.assert_array_almost_equal(residual, np.array([[0.0, 0.0]]))
        
        # 2. Project orthogonal vector -> Residual should be identical to input
        matrix = np.array([[1.0, -1.0]]) # Orthogonal to [1,1]
        mean_vec = np.array([1.0, 1.0])
        residual = context_projection(matrix, mean_vec)
        np.testing.assert_array_almost_equal(residual, matrix)
        
        # 3. Mixed case
        # Vec = [3, 2]. Context = [1, 0].
        # Proj = (3*1 + 2*0)/(1*1 + 0*0) * [1,0] = 3 * [1,0] = [3, 0]
        # Residual = [3, 2] - [3, 0] = [0, 2]
        matrix = np.array([[3.0, 2.0]])
        mean_vec = np.array([1.0, 0.0])
        residual = context_projection(matrix, mean_vec)
        np.testing.assert_array_almost_equal(residual, np.array([[0.0, 2.0]]))

    def test_calculate_similarity_stats(self):
        # Create a simple matrix
        # Index 0: Reference Start [1,0]
        # Index 1: Comparison A [1,0] (Sim 1.0)
        # Index 2: Comparison B [0,1] (Sim 0.0)
        # Index 3: Reference End [1,0]
        # Mean Reference = ([1,0]+[1,0])/2 = [1,0]
        
        m = np.array([
            [1.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0]
        ])
        
        stats = calculate_similarity_stats(m, start_index=0, end_index=3)
        
        # We expect similarities: [1.0, 0.0]
        # Mean = 0.5, Std = 0.5, Min = 0.0, Max = 1.0
        self.assertAlmostEqual(stats.mean, 0.5)
        self.assertAlmostEqual(stats.std, 0.5)
        self.assertAlmostEqual(stats.min, 0.0)
        self.assertAlmostEqual(stats.max, 1.0)
        
        # Error case: Invalid indices
        with self.assertRaises(IndexError):
            calculate_similarity_stats(m, 0, 1) # Not enough gap

if __name__ == "__main__":
    unittest.main()
