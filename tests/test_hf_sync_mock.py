import unittest
from unittest.mock import MagicMock, patch, ANY
import yaml
from pathlib import Path
from tempfile import TemporaryDirectory
import time

# Import the class to test (adjust path if needed)
# Assuming src is in python path or we append it
import sys
sys.path.append(str(Path(__file__).parent.parent / "src"))

from data.hf_sync import HFFileManager
from huggingface_hub.utils import EntryNotFoundError

class TestHFFileManager(unittest.TestCase):
    
    def setUp(self):
        self.repo_id = "test/repo"
        self.mock_api_patcher = patch('data.hf_sync.HfApi')
        self.mock_download_patcher = patch('data.hf_sync.hf_hub_download')
        
        self.mock_api = self.mock_api_patcher.start()
        self.mock_download = self.mock_download_patcher.start()
        
        self.mock_api_instance = MagicMock()
        self.mock_api.return_value = self.mock_api_instance
        
        self.manager = HFFileManager(self.repo_id)

    def tearDown(self):
        self.manager.shutdown()
        self.mock_api_patcher.stop()
        self.mock_download_patcher.stop()

    def test_ensure_config_match_success(self):
        """Test match when remote config exists and matches."""
        local_config = {"a": 1, "b": 2}
        
        # Mock download returning a path to a file with same config
        with TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "metadata.yaml"
            with open(config_path, "w") as f:
                yaml.dump(local_config, f)
            
            self.mock_download.return_value = str(config_path)
            
            # Should not raise exception
            self.manager.ensure_config_match("remote_folder", local_config)
            
            self.mock_download.assert_called_once()
            # Should NOT upload anything if match
            self.mock_api_instance.upload_file.assert_not_called()

    def test_ensure_config_match_mismatch(self):
        """Test error when remote config exists but differs."""
        local_config = {"a": 1}
        remote_config = {"a": 2}
        
        with TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "metadata.yaml"
            with open(config_path, "w") as f:
                yaml.dump(remote_config, f)
            
            self.mock_download.return_value = str(config_path)
            
            with self.assertRaisesRegex(RuntimeError, "Configuration MISMATCH"):
                self.manager.ensure_config_match("remote_folder", local_config)

    def test_ensure_config_match_missing_create(self):
        """Test upload when remote config missing."""
        local_config = {"new": "config"}
        self.mock_download.side_effect = EntryNotFoundError("Not found")
        
        self.manager.ensure_config_match("remote_folder", local_config)
        
        # Wait for async upload
        self.manager.shutdown(wait=True)
        
        # Should have called upload_file
        self.mock_api_instance.upload_file.assert_called_once()
        # Verify uploaded content
        call_args = self.mock_api_instance.upload_file.call_args
        self.assertIn("path_or_fileobj", call_args.kwargs)
        uploaded_bytes = call_args.kwargs["path_or_fileobj"]
        self.assertEqual(yaml.safe_load(uploaded_bytes), local_config)

    def test_list_files(self):
        """Test filtering of file list."""
        self.mock_api_instance.list_repo_files.return_value = [
            "folder/file1.json",
            "folder/file2.json",
            "folder/sub/file3.json", # Should be ignored for 'folder'
            "other/file4.json"
        ]
        
        files = self.manager.list_files("folder")
        
        self.assertEqual(files, {"file1.json", "file2.json"})
        self.assertNotIn("sub/file3.json", files)

    def test_upload_file_async(self):
        with TemporaryDirectory() as tmpdir:
            local_path = Path(tmpdir) / "file.json"
            local_path.touch()
            remote_path = "remote/file.json"
            
            self.manager.upload_file_async(local_path, remote_path)
            
            # Shutdown to ensure tasks execute
            self.manager.shutdown(wait=True)
            
            # Check create_commit was called
            self.mock_api_instance.create_commit.assert_called_once()
            
            # Retrieve arguments to verify details
            call_args = self.mock_api_instance.create_commit.call_args
            self.assertEqual(call_args.kwargs['repo_id'], self.repo_id)
            self.assertEqual(call_args.kwargs['repo_type'], "dataset")
            
            operations = call_args.kwargs['operations']
            self.assertEqual(len(operations), 1)
            op = operations[0]
            # We can't easily check class type without importing it, but we can check attributes if it's not a Mock
            # Or if it is real CommitOperationAdd. 
            self.assertEqual(op.path_in_repo, str(remote_path))
            self.assertEqual(str(op.path_or_fileobj), str(local_path))

if __name__ == "__main__":
    unittest.main()
