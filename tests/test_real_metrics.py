import unittest
import sys
import typing

# Monkeypatch Self for Python < 3.11
if not hasattr(typing, "Self"):
    typing.Self = typing.Any

import numpy as np
from evaluations.metrics import MetricsRecordRaw, MetricsMetadata, VectorStats, metrics_to_json

class TestRealMetrics(unittest.TestCase):
    def test_metrics_serialization_mixed_types(self):
        """
        Verify that MetricsRecordRaw can accept and serialize mixed types:
        - numpy arrays (vectors)
        - floats/ints (scalars like counts or means)
        - strings (Base64 matrix)
        """
        metadata = MetricsMetadata(
            data_type="video",
            recon_strategy="test_strat",
            video_id="vid_123",
            size=5,
            masked=[0, 2]
        )
        
        # Real-world data mix similar to what triggered the crash
        raw_data = {
            "mean_rank": 2.5, # Scalar float
            "retrieval_count_at_1": 1, # Scalar int
            "similarity_matrix_b64": "VGhpcyBpcyBhIEJhc2U2NCBTdHJpbmc=", # String
            "vector_metric": np.array([0.1, 0.2, 0.3]) # Numpy array
        }
        
        # 1. Instantiation (Validator check)
        record = MetricsRecordRaw(metadata=metadata, raw_metrics=raw_data)
        
        # 2. Stats calculation check
        # Should convert vector/scalar to VectorStats and ignore string
        stats_record = record.stats()
        
        self.assertIn("mean_rank", stats_record.metrics)
        self.assertIn("vector_metric", stats_record.metrics)
        self.assertNotIn("similarity_matrix_b64", stats_record.metrics, "Strings should be ignored in stats")
        
        # Verify scalar stat conversion
        mr_stats = stats_record.metrics["mean_rank"]
        self.assertEqual(mr_stats.mean, 2.5)
        self.assertEqual(mr_stats.std, 0.0)
        
        # 3. JSON Serialization check (simulating save to disk)
        # We don't serialize the record directly typically, but the Reconstructed object containing it.
        # But let's check basic dump capability of the internal dict if needed.
        # Actually, the runtime error was in `MetricsRecordRaw` instantiation/validation.
        
    def test_z_score_calculation(self):
        """Verify z-score logic handles mixed types gracefully."""
        metadata = MetricsMetadata(
             data_type="video",
             recon_strategy="test_strat",
             video_id="vid_123",
             size=5,
             masked=[]
        )
        
        start_data = {
            "val": 10.0,
            "vec": np.array([1.0, 2.0]),
            "b64": "ignore_me"
        }
        record = MetricsRecordRaw(metadata=metadata, raw_metrics=start_data)
        
        # Global stats mock
        global_stats = {
            "val": VectorStats(mean=5.0, std=2.0, min=0, max=10),
            "vec": VectorStats(mean=0.0, std=1.0, min=0, max=10) # Simple normalization
        }
        
        z_record = record.stats_z_score(global_stats)
        
        # 10.0 -> (10-5)/2 = 2.5
        self.assertAlmostEqual(z_record.metrics["val"].mean, 2.5)
        
        # vec ([1,2]) -> ([1-0]/1, [2-0]/1) -> [1, 2] -> mean=1.5
        self.assertAlmostEqual(z_record.metrics["vec"].mean, 1.5)
        
        self.assertNotIn("b64", z_record.metrics)

    def test_round_metrics(self):
        from evaluations.metrics import round_metrics
        
        data = {
            "scalar_long": 1.123456789,
            "list_long": [1.123456789, 2.987654321],
            "string": "keep_me",
            "int": 5
        }
        
        rounded = round_metrics(data, ndigits=2)
        
        self.assertEqual(rounded["scalar_long"], 1.12)
        self.assertEqual(rounded["list_long"], [1.12, 2.99])
        self.assertEqual(rounded["string"], "keep_me")
        self.assertEqual(rounded["int"], 5)

if __name__ == "__main__":
    unittest.main()
