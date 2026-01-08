import unittest
import numpy as np
from evaluations.evaluation import ReconstructionEvaluator
from evaluations.metrics import MetricsRecordRaw, MetricsMetadata
from evaluations.eval_vectors import calculate_retrieval_metrics

class TestMetricAggregation(unittest.TestCase):
    def test_agg_metrics_min_max_std(self):
        # Create fake metrics
        meta = MetricsMetadata(data_type="test", recon_strategy="test", video_id="v1", size=1, masked=[])
        
        # Video 1: mean_rank = 10
        m1 = MetricsRecordRaw(metadata=meta, raw_metrics={"mean_rank": 10.0})
        # Video 2: mean_rank = 20
        m2 = MetricsRecordRaw(metadata=meta, raw_metrics={"mean_rank": 20.0})
        # Video 3: mean_rank = 30
        m3 = MetricsRecordRaw(metadata=meta, raw_metrics={"mean_rank": 30.0})
        
        agg = ReconstructionEvaluator.agg_metrics([m1, m2, m3])
        
        # Check standard fields
        self.assertEqual(agg["num_of_instances"], 3)
        self.assertIn("mean_mean_rank_mean", agg)
        
        # Check new stats
        self.assertEqual(agg["mean_mean_rank_mean"], 20.0) # (10+20+30)/3
        self.assertEqual(agg["min_mean_rank_mean"], 10.0)
        self.assertEqual(agg["max_mean_rank_mean"], 30.0)
        # std of [10, 20, 30] is 8.16...
        self.assertAlmostEqual(agg["std_mean_rank_mean"], 8.1649, places=3)

    def test_retrieval_metrics_ranks_output(self):
        # Fake vectors: 3 items
        # Reconstructed is close to GT
        recon = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]])
        gt = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 1.0]])
        # Distractors same as GT
        distractors = gt
        
        metrics = calculate_retrieval_metrics(
            reconstructed_vectors=recon, 
            ground_truth_vectors=gt, 
            distractor_pool=distractors
        )
        
        self.assertIn("ranks", metrics)
        ranks = metrics["ranks"]
        self.assertTrue(isinstance(ranks, (list, np.ndarray)))
        self.assertEqual(len(ranks), 3)
        # Item 1: matches GT[0] (perfect) -> rank 1
        # Item 2: matches GT[1] (perfect) -> rank 1
        # Item 3: recon[2] is [1,0], GT[2] is [0,1]. 
        # Sim(recon2, GT2) = 0. Sim(recon2, dist[0]) = 1.0 > 0.
        # So rank should be > 1.
        
        self.assertEqual(ranks[0], 1)
        
    def test_agg_metrics_with_ranks_array(self):
        # Test that array metrics (like 'ranks') get aggregated correctly into per-video stats
        # and then into global stats
        meta = MetricsMetadata(data_type="test", recon_strategy="test", video_id="v1", size=1, masked=[])
        
        # Video 1: ranks = [1, 2, 3] -> mean=2, min=1
        m1 = MetricsRecordRaw(metadata=meta, raw_metrics={"ranks": [1, 2, 3]})
        # Video 2: ranks = [4, 5, 6] -> mean=5, min=4
        m2 = MetricsRecordRaw(metadata=meta, raw_metrics={"ranks": [4, 5, 6]})
        
        agg = ReconstructionEvaluator.agg_metrics([m1, m2])
        
        # Check "mean" aggregation (mean of means)
        # Video 1 mean: 2. Video 2 mean: 5. Global Mean of Mean: 3.5
        self.assertEqual(agg.get("mean_ranks_mean"), 3.5)
        
        # Check "min" aggregation (mean of mins)
        # Video 1 min: 1. Video 2 min: 4. Global Mean of Min: 2.5
        self.assertEqual(agg.get("mean_ranks_min"), 2.5)

        # Check "min_ranks_mean" (best average rank achieved by a video)
        # V1 mean=2, V2 mean=5. Min is 2.
        self.assertEqual(agg.get("min_ranks_mean"), 2.0)

if __name__ == '__main__':
    unittest.main()
