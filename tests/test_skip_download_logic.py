
import unittest
from unittest.mock import MagicMock, patch, Mock
from pathlib import Path
import sys

# Adjust path to include src
sys.path.append(str(Path(__file__).parent.parent / "src"))

from experiment_executor.experiment_runner import ExperimentRunner
from data.hf_sync import HFFileManager
from data_models.captions_only import CaptionedVideo, CaptionedClip, TimestampRange

class TestExperimentRunnerSkipDownload(unittest.TestCase):
    
    def setUp(self):
        self.video = CaptionedVideo(
            video_id="vid_skip_test", 
            clips=[CaptionedClip(index=0, timestamp=TimestampRange(start=0, duration=1), caption="hello")]
        )
        
        self.mock_loader = Mock()
        self.mock_loader.get_data_type_name.return_value = "mock"
        
        self.mock_hf = Mock(spec=HFFileManager)
        self.mock_masking = Mock()
        self.mock_recon = Mock()
        self.mock_eval = Mock()
        
    def test_skip_download_flag_true(self):
        """Test that with no_download_existing=True, we don't download even if file exists remotely."""
        runner = ExperimentRunner(
            run_name="test_run",
            data_loader=self.mock_loader,
            masking_strategy=self.mock_masking,
            reconstruction_strategy=self.mock_recon,
            evaluator=self.mock_eval,
            save_path=Path("/tmp/mock_results"),
            conf_for_log={},
            hf_manager=self.mock_hf,
            no_download_existing=True
        )
        
        # Setup state: File exists remotely, but not locally
        runner.remote_files = {"vid_skip_test.json"}
        # Ensure it checks local file and finds it missing
        with patch.object(Path, "exists", return_value=False):
            
            result = runner._process_single_video(self.video)
            
            # Assertions
            self.assertIsNone(result, "Should return None when skipping download")
            self.mock_hf.download_file.assert_not_called()
            self.mock_recon.reconstruct.assert_not_called()

    def test_skip_download_flag_false(self):
        """Test that with no_download_existing=False, we DO download."""
        runner = ExperimentRunner(
            run_name="test_run",
            data_loader=self.mock_loader,
            masking_strategy=self.mock_masking,
            reconstruction_strategy=self.mock_recon,
            evaluator=self.mock_eval,
            save_path=Path("/tmp/mock_results"),
            conf_for_log={},
            hf_manager=self.mock_hf,
            no_download_existing=False
        )
        
        # Setup state: File exists remotely, but not locally
        runner.remote_files = {"vid_skip_test.json"}
        
        with patch.object(Path, "exists", return_value=False):
            # Mock download success to stop further processing
            self.mock_hf.download_file.return_value = True
            
            # Mock loading existing result
            with patch.object(runner, "_load_existing_result") as mock_load:
                mock_load.return_value = Mock()
                
                runner._process_single_video(self.video)
                
                self.mock_hf.download_file.assert_called_once()
                mock_load.assert_called_once()

if __name__ == "__main__":
    unittest.main()
