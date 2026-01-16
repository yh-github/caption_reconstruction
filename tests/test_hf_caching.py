
import unittest
from unittest.mock import MagicMock, patch
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent / "src"))
from data.hf_sync import HFFileManager

class TestHFCaching(unittest.TestCase):
    def setUp(self):
        self.patcher = patch('data.hf_sync.HfApi')
        self.mock_api_cls = self.patcher.start()
        self.mock_api = self.mock_api_cls.return_value
        self.manager = HFFileManager("test/repo", read_only=True)
        
    def tearDown(self):
        self.manager.shutdown()
        self.patcher.stop()
        
    def test_list_files_caches_result(self):
        # Setup mock to return a list
        self.mock_api.list_repo_files.return_value = ["a/1.txt", "a/2.txt", "b/3.txt"]
        
        # First call
        files_a = self.manager.list_files("a")
        self.assertEqual(files_a, {"1.txt", "2.txt"})
        self.mock_api.list_repo_files.assert_called_once()
        
        # Second call (different folder)
        files_b = self.manager.list_files("b")
        self.assertEqual(files_b, {"3.txt"})
        
        # Should NOT have called API again
        self.mock_api.list_repo_files.assert_called_once()

if __name__ == "__main__":
    unittest.main()
