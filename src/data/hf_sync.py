import json
import logging
import time
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Optional

import yaml
from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.utils import EntryNotFoundError

logger = logging.getLogger(__name__)

class HFFileManager:
    """
    Manages synchronization of experiment results with a HuggingFace Dataset.
    Acts as a 'Shared Cache' source of truth.
    """
    def __init__(self, repo_id: str, repo_type: str = "dataset"):
        self.repo_id = repo_id
        self.repo_type = repo_type
        self.api = HfApi()
        
        # Background uploader setup
        self._upload_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="hf_upload")
        self._pending_uploads = 0
        self._lock = threading.Lock() # For pending count
        
    def ensure_config_match(self, remote_dir: str, local_config: dict[str, Any]) -> None:
        """
        Checks if a metadata.yaml exists in the remote directory.
        If YES: Validates that it matches `local_config`. Raises error if mismatch.
        If NO: Uploads `local_config` as metadata.yaml.
        """
        metadata_filename = "metadata.yaml"
        remote_path = f"{remote_dir}/{metadata_filename}"
        
        logger.info(f"Checking config compatibility at {self.repo_id}/{remote_path}...")
        
        try:
            # 1. Try to download existing metadata
            cached_path = hf_hub_download(
                repo_id=self.repo_id,
                filename=remote_path,
                repo_type=self.repo_type
            )
            
            with open(cached_path, 'r') as f:
                remote_config = yaml.safe_load(f)
            
            # Compare critical fields (you might want to allow some drift, but strict is safer)
            # For now, let's assume strict equality on the dumped dicts or key params.
            # A simple comparison:
            if remote_config != local_config:
                # TODO: Implement smarter diff visibility if needed
                raise RuntimeError(
                    f"Configuration MISMATCH! The remote folder '{remote_path}' already exists "
                    "with a DIFFERENT configuration. \n"
                    "Aborting to prevent dataset corruption. Please change your run name/config or fix the config."
                )
            logger.info("Remote config matches local. Proceeding.")
            
        except EntryNotFoundError:
            # 2. Upload our config if it doesn't exist
            logger.info("No existing metadata found. Claiming this folder.")
            self._upload_bytes_async(
                data=yaml.dump(local_config).encode('utf-8'),
                remote_path=remote_path
            )

    def list_files(self, folder_path: str) -> set[str]:
        """
        Returns a set of filenames (not full paths) present in the remote folder.
        Uses the HF API to list files efficiently.
        """
        try:
            # list_repo_files returns ALL files. We filter by prefix.
            all_files = self.api.list_repo_files(repo_id=self.repo_id, repo_type=self.repo_type)
            
            folder_prefix = folder_path.rstrip('/') + '/'
            files_in_folder = set()
            
            for f in all_files:
                if f.startswith(folder_prefix):
                    # Extract just the filename (e.g. "video_id.json")
                    rel_name = f[len(folder_prefix):]
                    if '/' not in rel_name: # Ensure we are not picking up sub-sub-folders
                        files_in_folder.add(rel_name)
                        
            return files_in_folder
        except Exception as e:
            logger.warning(f"Failed to list files from HF: {e}")
            return set()

    def download_file(self, remote_path: str, local_path: Path) -> bool:
        """
        Downloads a specific file to local_path. Returns True if successful.
        """
        try:
            hf_hub_download(
                repo_id=self.repo_id,
                filename=remote_path,
                repo_type=self.repo_type,
                local_dir=str(local_path.parent), # hf_hub_download expects dir
                local_dir_use_symlinks=False,
                force_filename=local_path.name # Ensure it maps to exact local filename
            )
            return True
        except Exception as e:
            logger.error(f"Failed to download {remote_path}: {e}")
            return False

    def upload_file_async(self, local_path: Path, remote_path: str):
        """
        Queues a file upload to run in the background.
        """
        with self._lock:
            self._pending_uploads += 1
            
        self._upload_executor.submit(self._do_upload, local_path, remote_path)

    def _do_upload(self, local_path: Path, remote_path: str):
        try:
            self.api.upload_file(
                path_or_fileobj=local_path,
                path_in_repo=remote_path,
                repo_id=self.repo_id,
                repo_type=self.repo_type,
                commit_message=f"Upload {Path(remote_path).name}"
            )
        except Exception as e:
            logger.error(f"Background upload failed for {remote_path}: {e}")
        finally:
            with self._lock:
                self._pending_uploads -= 1

    def _upload_bytes_async(self, data: bytes, remote_path: str):
        """Internal helper for in-memory uploads"""
        self._upload_executor.submit(self._do_upload_bytes, data, remote_path)
        
    def _do_upload_bytes(self, data: bytes, remote_path: str):
        try:
            self.api.upload_file(
                path_or_fileobj=data,
                path_in_repo=remote_path,
                repo_id=self.repo_id,
                repo_type=self.repo_type,
                commit_message=f"Upload metadata {Path(remote_path).name}"
            )
        except Exception as e:
            logger.error(f"Background upload bytes failed for {remote_path}: {e}")

    def shutdown(self, wait: bool = True):
        """
        Waits for pending uploads to finish and shuts down the executor.
        """
        logger.info("Shutting down HF background uploader...")
        self._upload_executor.shutdown(wait=wait)
        logger.info("HF uploader shutdown complete.")
