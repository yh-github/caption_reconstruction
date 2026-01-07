import unittest
import numpy as np
from evaluations.eval_vectors import calculate_retrieval_metrics

class TestRetrievalMetrics(unittest.TestCase):
    def test_perfect_match(self):
        # GT and Recon are identical
        # Pool is GT
        gt = np.array([[1.0, 0.0], [0.0, 1.0]])
        recon = np.array([[1.0, 0.0], [0.0, 1.0]])
        
        metrics = calculate_retrieval_metrics(recon, gt, gt)
        
        self.assertEqual(metrics['mean_rank'], 1.0)
        self.assertEqual(metrics['recall_at_1'], 1.0)
        self.assertEqual(metrics['mrr'], 1.0)

    def test_swapped_predictions(self):
        # 2 vectors. Recon is swapped.
        # gt[0] = [1,0], gt[1] = [0,1]
        # recon[0] = [0,1] (matches gt[1]), recon[1] = [1,0] (matches gt[0])
        
        # Sim Matrix (Recon x Pool):
        # R0 x G0 (1,0) = 0
        # R0 x G1 (0,1) = 1 (Better!)
        
        # Sim GT (R0 x G0) = 0
        
        # "Better than GT":
        # For R0: G1(1.0) > G0(0.0)? Yes. Count = 1. Rank = 2.
        
        gt = np.array([[1.0, 0.0], [0.0, 1.0]])
        recon = np.array([[0.0, 1.0], [1.0, 0.0]])
        
        metrics = calculate_retrieval_metrics(recon, gt, gt)
        
        self.assertEqual(metrics['mean_rank'], 2.0)
        self.assertEqual(metrics['recall_at_1'], 0.0)
        self.assertEqual(metrics['mrr'], 0.5)

    def test_duplicates_in_pool(self):
        # GT: A, A, B. 
        # Recon: A, A, B.
        # R0(A) should match G0(A) AND G1(A).
        # Sim(R0, G0) = 1.0.
        # Sim(R0, G1) = 1.0.
        # Sim(R0, G2) = 0.0.
        
        # Strict inequality > :
        # G1 > G0 ? 1.0 > 1.0 False.
        # G2 > G0 ? 0.0 > 1.0 False.
        # Count = 0. Rank = 1.
        
        gt = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        recon = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        
        metrics = calculate_retrieval_metrics(recon, gt, gt)
        
        self.assertEqual(metrics['mean_rank'], 1.0)
        self.assertEqual(metrics['recall_at_1'], 1.0)

    def test_inexact_but_good_match(self):
        # GT: [1, 0]
        # Pool: [1, 0], [0, 1]
        # Recon: [0.9, 0.1]
        
        # Sim(R, G0) = 0.9
        # Sim(R, G1) = 0.1
        
        # G1 > G0? 0.1 > 0.9 False.
        # Rank 1.
        
        gt = np.array([[1.0, 0.0]])
        pool = np.array([[1.0, 0.0], [0.0, 1.0]])
        recon = np.array([[0.9, 0.1]])
        
        metrics = calculate_retrieval_metrics(recon, gt, pool)
        self.assertEqual(metrics['mean_rank'], 1.0)

if __name__ == '__main__':
    unittest.main()
