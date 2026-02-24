import unittest
from unittest.mock import Mock, patch, mock_open, MagicMock
from pathlib import Path
import argparse
import sys
import os

# Ensure src is in pythonpath
sys.path.append(str(Path(__file__).parent.parent / "src"))

from experiment_executor import yt_video_processing
from data_models.video_link import VideoLinkData
from llm.llm_interaction import LLM_Response

class TestYTVideoProcessing(unittest.TestCase):
    def setUp(self):
        self.mock_config = {
            "base_params": {"master_seed": 1234},
            "paths": {"log_dir": "/tmp/logs", "disk_cache": "/tmp/cache"},
            "data_config": {
                "name": "test_dataset",
                "path": "test_path.json", 
                "limit": 10,
                "out_path": "/tmp/out",
                "duration_limit": 60
            },
            "llm": {
                "model_name": "test-model",
                "temperature": 0.0,
                "seed": 42,
                "response_schema": "list[CaptionedInterval]",
                "prompt_template": "dummy_prompt.txt",
                "fps": 1
            },
            "__parent_run_name__": "test_run"
        }

    @patch('experiment_executor.yt_video_processing.load_config')
    @patch('experiment_executor.yt_video_processing.read_prompt')
    @patch('experiment_executor.yt_video_processing.load_wild_links')
    @patch('experiment_executor.yt_video_processing.setup_logging')
    @patch('experiment_executor.yt_video_processing.LLM_Manager_Builder')
    @patch('diskcache.Cache')
    @patch('google.genai.Client')
    @patch('os.path.exists')
    @patch('os.path.getsize')
    @patch('builtins.open', new_callable=mock_open) # For save_to_file
    def test_main_processing_flow(self, mock_file_open, mock_getsize, mock_exists, mock_genai_client, 
                                  mock_cache, mock_builder_cls, mock_setup_logging, mock_load_links, 
                                  mock_read_prompt, mock_load_config):
        
        # 1. Setup Mocks
        mock_load_config.return_value = self.mock_config
        mock_read_prompt.return_value = "This is a prompt template"
        
        # Mock Video Links
        mock_video = MagicMock(spec=VideoLinkData)
        mock_video.video_id = "test_vid_1"
        mock_video.duration.return_value = 70.0 # Valid duration > 60
        mock_video.start_offset = 0
        mock_video.end_offset = 50
        mock_video.uri = "gs://video.uri"
        mock_load_links.return_value = [mock_video]
        
        # Mock Logging
        mock_logger = Mock()
        mock_setup_logging.return_value = ("/tmp/log.txt", mock_logger)
        
        # Mock LLM Builder and Manager
        mock_builder = mock_builder_cls.return_value
        
        # FIX: define a proper Pydantic model for the list items
        from pydantic import BaseModel
        class MockItem(BaseModel):
            start: str
            end: str
            caption: str
            
        mock_builder.config_response_schema.return_value = list[MockItem]  # Use paramterized list
        mock_llm = Mock()
        mock_builder.from_config.return_value = mock_llm
        
        # Mock LLM Response
        mock_response = Mock(spec=LLM_Response)
        mock_response.text = '[{"start": "0s", "end": "10s", "caption": "test"}]'
        mock_response.thoughts = "Some thoughts"
        mock_response.exception = None
        mock_llm.call.return_value = mock_response
        
        # Mock File System (Video not processed)
        mock_exists.return_value = False
        
        # 2. Run Main
        args = argparse.Namespace(config_path="dummy.yaml", dry_run=False)
        yt_video_processing.main(args)
        
        # 3. Assertions
        
        # Verify valid video was processed
        mock_llm.call.assert_called_once()
        
        # Verify result was saved
        # We expect open to be called for saving the result
        # Note: open is also called by read_prompt in real execution, but we mocked read_prompt
        # yt_video_processing.save_to_file calls open
        # We need to verify that save_to_file logic was triggered, which writes JSON
        
        # Inspect the content written
        handle = mock_file_open()
        
        # Check if any write call contained the video_id and captions
        written_content = "".join(call.args[0] for call in handle.write.call_args_list)
        self.assertIn("test_vid_1", written_content)
        self.assertIn("Some thoughts", written_content)

    @patch('experiment_executor.yt_video_processing.load_config')
    @patch('experiment_executor.yt_video_processing.read_prompt')
    @patch('experiment_executor.yt_video_processing.load_wild_links')
    @patch('experiment_executor.yt_video_processing.setup_logging')
    def test_dry_run_logic(self, mock_setup_logging, mock_load_links, mock_read_prompt, mock_load_config):
        # 1. Setup
        mock_load_config.return_value = self.mock_config
        mock_read_prompt.return_value = "prompt"
        mock_setup_logging.return_value = ("/tmp/log", Mock())
        
        mock_video = MagicMock(spec=VideoLinkData)
        mock_video.video_id = "test_vid_1"
        mock_video.duration.return_value = 50.0
        mock_load_links.return_value = [mock_video]
        
        # 2. Run with dry_run=True
        args = argparse.Namespace(config_path="dummy.yaml", dry_run=True)
        
        # We capture stdout to allow verification if needed, or just ensure it doesn't crash
        with patch('sys.stdout', new_callable=Mock) as mock_stdout:
            yt_video_processing.main(args)
            
            # 3. Assertions
            # Should NOT try to open cache or build LLM
            # (We didn't mock them, so if it tried, it would likely fail or we'd see it)
            # Verify Dry Run output logic (printing)
            print_calls = [c[0][0] for c in mock_stdout.write.call_args_list if isinstance(c[0][0], str)]
            # Note: sys.stdout.write is low level. print() calls write.
            # Easier to verify flow didn't crash.

from experiment_executor.yt_video_processing import load_local_links, gen_content_prompt_local, _normalize_name, _build_local_cache
from data_models.video_link import VideoLocalData
import tempfile

class TestLocalVideoProcessing(unittest.TestCase):
    def test_normalize_name(self):
        self.assertEqual(_normalize_name("Olly's-Farm_6-clip-1"), "olly_s-farm_6-clip-1")
        self.assertEqual(_normalize_name("Test-Video_1-clip-0"), "test-video_1-clip-0")
    
    @patch('experiment_executor.yt_video_processing._build_local_cache')
    @patch('experiment_executor.yt_video_processing.load_wild_dataset')
    def test_load_local_links(self, mock_load_wild_dataset, mock_build_cache):
        # Mock the dataset
        mock_wild_metadata = MagicMock()
        mock_wild_metadata.video_id = "test_vid_1"
        mock_wild_metadata.duration = 20.0
        mock_load_wild_dataset.return_value = [mock_wild_metadata]
        
        # Mock the cache (what _build_local_cache would return)
        mock_build_cache.return_value = {
            "test_vid_1": Path("/fake/test_vid_1.mp4"),
        }
        
        # Act
        links = load_local_links(
            path="dummy_path.json",
            local_dirs=[Path("/fake/dir")],
            duration_limit=10.0,
            max_size=None
        )
        
        # Assert
        self.assertEqual(len(links), 1)
        self.assertIsInstance(links[0], VideoLocalData)
        self.assertEqual(links[0].video_id, "test_vid_1")
        self.assertAlmostEqual(links[0].clip_duration, 10.5)  # Because limit_duration adds 0.5 buffer

    @patch('experiment_executor.yt_video_processing._build_local_cache')
    @patch('experiment_executor.yt_video_processing.load_wild_dataset')
    def test_load_local_links_apostrophe_matching(self, mock_load_wild_dataset, mock_build_cache):
        # JSON has apostrophe, filesystem has underscore
        mock_wild_metadata = MagicMock()
        mock_wild_metadata.video_id = "Olly's-Farm_6-clip-1"
        mock_wild_metadata.duration = 65.0
        mock_load_wild_dataset.return_value = [mock_wild_metadata]
        
        mock_build_cache.return_value = {
            "olly_s-farm_6-clip-1": Path("/fake/Olly_s-Farm_6-clip-1.mp4"),
        }
        
        links = load_local_links(
            path="dummy_path.json",
            local_dirs=[Path("/fake/dir")],
            duration_limit=60.0,
            max_size=None
        )
        
        self.assertEqual(len(links), 1)
        self.assertEqual(links[0].video_id, "Olly's-Farm_6-clip-1")

    @patch('experiment_executor.yt_video_processing.subprocess.run')
    @patch('builtins.open', new_callable=mock_open, read_data=b"fake_video_bytes")
    @patch('experiment_executor.yt_video_processing.os.remove')
    @patch('experiment_executor.yt_video_processing.os.path.exists', return_value=True)
    def test_gen_content_prompt_local(self, mock_exists, mock_remove, mock_file_open, mock_subprocess_run):
        # Setup
        vl = VideoLocalData(video_id="test_vid_1", path=Path("/tmp/test.mp4"), clip_duration=5.0)
        
        # Act
        prompt = "Describe video"
        content = gen_content_prompt_local(vl, prompt, fps=2)
        
        # Assert
        # Check subprocess.run called with correct ffmpeg args
        mock_subprocess_run.assert_called_once()
        cmd_args = mock_subprocess_run.call_args[0][0]
        self.assertEqual(cmd_args[0], 'ffmpeg')
        self.assertIn('-i', cmd_args)
        self.assertIn('/tmp/test.mp4', cmd_args)
        self.assertIn('-t', cmd_args)
        self.assertIn('5.0', cmd_args)
        
        # Check returned content structure
        self.assertEqual(len(content.parts), 2)
        
        video_part = content.parts[0]
        self.assertEqual(video_part.inline_data.mime_type, 'video/mp4')
        self.assertEqual(video_part.inline_data.data, b"fake_video_bytes")
        self.assertEqual(video_part.video_metadata.fps, 2)
        
        text_part = content.parts[1]
        self.assertEqual(text_part.text, "Describe video")
