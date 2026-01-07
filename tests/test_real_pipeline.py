import unittest
from unittest.mock import MagicMock, patch
from pathlib import Path
import shutil
import tempfile
import numpy as np

from experiment_executor.experiment_runner import ExperimentRunner
from data.data_loaders import BaseDataLoader
from data_models.captions_only import CaptionedVideo, CaptionedClip, TimestampRange
from evaluations.evaluation import ReconstructionEvaluator
from reconstruction.masking import MaskingStrategy
from reconstruction.text_reconstruction import ReconstructionStrategy, Reconstructed

class TestRealPipeline(unittest.TestCase):
    def setUp(self):
        self.test_dir = Path(tempfile.mkdtemp())
        self.save_path = self.test_dir / "results"
        self.save_path.mkdir()

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_experiment_runner_end_to_end(self):
        """
        Tests the full run loop of ExperimentRunner with mocked strategies 
        but real data transfer and file saving.
        """
        # 1. Setup Data
        video = CaptionedVideo(
            video_id="test_vid_001",
            clips=[
                CaptionedClip(index=0, timestamp=TimestampRange(start=0, duration=5), caption="Hello world"),
                CaptionedClip(index=1, timestamp=TimestampRange(start=5, duration=5), caption="Goodbye moon")
            ]
        )
        
        # 2. Setup Mocks
        mock_loader = MagicMock(spec=BaseDataLoader)
        mock_loader.load.return_value = [video]
        mock_loader.get_data_type_name.return_value = "test_data"
        
        mock_masking = MagicMock(spec=MaskingStrategy)
        # Return video + set of indices masked
        mock_masking.mask_video.return_value = (video, {0}) 
        
        mock_recon_strat = MagicMock(spec=ReconstructionStrategy)
        reconstructed_obj = Reconstructed(
            video_id="test_vid_001",
            reconstructed_captions={0: "Hello universe"}, # Changed caption
            metrics=None 
        )
        mock_recon_strat.reconstruct.return_value = reconstructed_obj
        
        mock_evaluator = MagicMock(spec=ReconstructionEvaluator)
        # Evaluator returns raw metrics dict
        mock_evaluator.evaluate.return_value = {
            "bert_score": 0.95, 
            "vector_sim": np.array([0.8, 0.9])
        }
        
        # 3. Initialize Runner
        runner = ExperimentRunner(
            run_name="test_integration_run",
            data_loader=mock_loader,
            masking_strategy=mock_masking,
            reconstruction_strategy=mock_recon_strat,
            evaluator=mock_evaluator,
            save_path=self.save_path,
            conf_for_log={"test": True}
        )
        
        # 4. Run
        metrics = runner.run()
        
        # 5. Assertions
        
        # Check Metrics Returned
        self.assertEqual(len(metrics), 1)
        record = metrics[0]
        self.assertEqual(record.metadata.video_id, "test_vid_001")
        self.assertEqual(record.raw_metrics["bert_score"], 0.95)
        np.testing.assert_array_equal(record.raw_metrics["vector_sim"], np.array([0.8, 0.9]))
        
        # Check File Saved
        expected_file = self.save_path / "test_integration_run" / "test_vid_001.json"
        self.assertTrue(expected_file.exists())
        
        with open(expected_file, 'r') as f:
            import json
            content = json.load(f)
            # Ensure "metrics" were injected into the saved file
            self.assertEqual(content["metrics"]["bert_score"], 0.95)
            # Numpy array in JSON should be list
            self.assertEqual(content["metrics"]["vector_sim"], [0.8, 0.9])

if __name__ == "__main__":
    unittest.main()
