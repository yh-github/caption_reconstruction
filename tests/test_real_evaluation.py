import unittest
import sys
import typing
import numpy as np
from unittest.mock import MagicMock

# Monkeypatch Self for Python < 3.11
if not hasattr(typing, "Self"):
    typing.Self = typing.Any

from evaluations.evaluation import ReconstructionEvaluator_Retrieval, Reconstructed
from data_models.captions_only import CaptionedVideo, CaptionedClip, TimestampRange

class TestRealEvaluation(unittest.TestCase):
    def setUp(self):
        # Create a mock embedder that returns predictable vectors
        self.mock_embedder = MagicMock()
        
        # Behavior: 
        # If text is "orig_1", return [1, 0]
        # If text is "recon_1", return [1, 0] (Perfect match)
        # If text is "orig_2", return [0, 1]
        # If text is "recon_2", return [0, -1] (Opposite)
        # If text is "context_1", return [0.1, 0.1] (Some background vector)
        
        def side_effect(video_id, texts, use_cache=True):
            # ... debug prints ...
            res = []
            for t in texts:
                if isinstance(t, str):
                    if "1" in t and "context" not in t: res.append([1.0, 0.0])
                    elif "2" in t: res.append([0.0, 1.0])
                    elif "opp" in t: res.append([0.0, -1.0])
                    elif "context" in t: res.append([1.0, 1.0]) # Context along diagonal
                    else: res.append([0.5, 0.5])
                else: res.append([0.1, 0.1])
            return np.array(res)
            
        self.mock_embedder.get_embeddings.side_effect = side_effect
        
        self.evaluator = ReconstructionEvaluator_Retrieval(self.mock_embedder)

    def test_evaluate_flow(self):
        """
        Tests the full evaluate() flow with:
        - 2 clips
        - Reconstruction perfectly matching clip 1
        - Reconstruction completely opposite to clip 2
        """
        # 1. Setup Input Data
        video = CaptionedVideo(
            video_id="test_vid",
            clips=[
                CaptionedClip(index=0, timestamp=TimestampRange(start=0.0, duration=1.0), caption="orig_1"),
                CaptionedClip(index=1, timestamp=TimestampRange(start=1.0, duration=1.0), caption="orig_2"),
                CaptionedClip(index=2, timestamp=TimestampRange(start=2.0, duration=1.0), caption="context_1") # Unmasked
            ]
        )
        
        # Reconstructed - Only indices 0 and 1 are reconstructed. Index 2 is context.
        reconstructed = Reconstructed(
            video_id="test_vid",
            reconstructed_captions={
                0: "recon_1",   # Should match orig_1 ([1,0] == [1,0])
                1: "recon_opp"  # Should mismatch orig_2 ([0,-1] vs [0,1])
            }
        )
        
        # 2. Run Evaluation
        metrics = self.evaluator.evaluate(reconstructed, video)
        
        # 3. Assertions
        print("Metrics:", metrics)
        
        # Check presence of keys
        self.assertIn("mean_rank", metrics)
        self.assertIn("similarity_matrix_b64", metrics)
        self.assertIn("retrieval_count_at_1", metrics)
        
        # Check retrieval logic correctness
        # Item 1: Recon=[1,0], GT=[1,0]. Distractor/Pool has [1,0] and [0,1].
        # Sim(Recon1, GT1) = 1.0
        # Sim(Recon1, Pool1) = 1.0, Sim(Recon1, Pool2) = 0.0. Rank = 1.
        
        # Item 2: Recon=[0,-1], GT=[0,1].
        # Sim(Recon2, GT2) = -1.0
        # Sim(Recon2, Pool1) = 0.0, Sim(Recon2, Pool2) = -1.0.
        # Wait, pool logic compares against ALL GTs.
        # Pool = [[1,0], [0,1]]
        # Query 2 is [0, -1]
        # Sim vs Pool[0] (1,0) = 0.0
        # Sim vs Pool[1] (0,1) = -1.0
        # Sim vs GT (0,1) = -1.0
        # Is 0.0 > -1.0? Yes. So Pool[0] is strictly better. Rank = 2.
        
        # So ranks should be [1, 2]. Mean rank = 1.5.
        self.assertEqual(metrics["mean_rank"], 1.5)
        
        # Retrieval at 1: Only item 1. Count = 1.
        self.assertEqual(metrics["retrieval_count_at_1"], 1)
        self.assertEqual(metrics["retrieval_total_queries"], 2)
        
        # Base64 string check
        self.assertIsInstance(metrics["similarity_matrix_b64"], str)
        self.assertTrue(len(metrics["similarity_matrix_b64"]) > 10)

if __name__ == "__main__":
    unittest.main()
