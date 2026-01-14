import json
import logging
import time
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Optional, List

import yaml
from huggingface_hub import HfApi, hf_hub_download, CommitOperationAdd, snapshot_download
from huggingface_hub.utils import EntryNotFoundError

logger = logging.getLogger(__name__)

class HFFileManager:
    """
    Manages synchronization of experiment results with a HuggingFace Dataset.
    """
    DISABLE_SSL_VERIFY = False # Set to True for debug only
    
    BATCH_SIZE = 50
    FLUSH_INTERVAL = 300 # 5 minutes

    def __init__(self, repo_id: str, read_only: bool = False):
        self.repo_id = repo_id
        self.repo_type = "dataset"
        self.read_only = read_only
        
        self.api = HfApi()

        if self.DISABLE_SSL_VERIFY:
             import ssl
             ssl._create_default_https_context = ssl._create_unverified_context
             
        # Batching state
        self._queue: List[tuple[Path, str]] = [] # list of (local_path, remote_path)
        self._lock = threading.Lock()
        
        # Consistency state
        self._watched_folders: set[tuple[Path, str]] = set() # (local_parent, remote_parent)
        
        self._stop_event = threading.Event()
        
        if not self.read_only:
            self._worker_thread = threading.Thread(target=self._worker_loop, daemon=True, name="hf_batch_uploader")
            self._worker_thread.start()
        else:
            self._worker_thread = None
        
        
        self._last_flush_time = time.time()
        self._last_log_sync_time = time.time()
        self._log_file: Optional[Path] = None
        
    def register_log_file(self, log_path: Path):
        """Registers the active log file to be periodically synced."""
        self._log_file = log_path
        logger.info(f"Registered log file for sync: {log_path}")

    def _sync_log_file(self):
        """Uploads the current log file to HF."""
        if not self._log_file or not self._log_file.exists():
            return
            
        try:
            remote_path = f"logs/{self._log_file.name}"
            
            if self.read_only:
                 # Debug log for dry-run verification
                 logger.info(f"Read-only mode: Skipping log upload for {remote_path}")
                 return

            self.api.upload_file(
                path_or_fileobj=self._log_file,
                path_in_repo=remote_path,
                repo_id=self.repo_id,
                repo_type=self.repo_type,
                commit_message=f"Sync log {self._log_file.name}"
            )
        except Exception as e:
            logger.warning(f"Failed to sync log file: {e}")
        
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
            
            if remote_config != local_config:
                raise RuntimeError(
                    f"Configuration MISMATCH! The remote folder '{remote_path}' already exists "
                    "with a DIFFERENT configuration. \n"
                    "Aborting to prevent dataset corruption. Please change your run name/config or fix the config."
                )
            logger.info("Remote config matches local. Proceeding.")
            
        except EntryNotFoundError:
            # 2. Upload our config if it doesn't exist
            logger.info("No existing metadata found.")
            
            if self.read_only:
                 logger.info("Read-only mode: Skipping metadata upload (simulated claim).")
                 return

            logger.info("Claiming this folder.")
            # Immediate upload for metadata to claim the lock/folder
            try:
                self.api.upload_file(
                    path_or_fileobj=yaml.dump(local_config).encode('utf-8'),
                    path_in_repo=remote_path,
                    repo_id=self.repo_id,
                    repo_type=self.repo_type,
                    commit_message=f"Init metadata {Path(remote_path).name}"
                )
            except Exception as e:
                logger.error(f"Failed to upload metadata: {e}")

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

    def prefetch_folder(self, download_dir: Path, allow_patterns: List[str]) -> Optional[Path]:
        """
        Downloads files matching the patterns to the specified directory using snapshot_download.
        Returns the path to the downloaded folder if successful, None otherwise.
        """
        try:
            logger.info(f"Prefetching files with patterns: {allow_patterns}")
            local_path = snapshot_download(
                repo_id=self.repo_id,
                repo_type=self.repo_type,
                local_dir=download_dir,
                allow_patterns=allow_patterns,
                tqdm_class=None
            )
            return Path(local_path)
        except Exception as e:
            logger.error(f"Failed to prefetch folder: {e}")
            return None

    def upload_file_async(self, local_path: Path, remote_path: str):
        """
        Queues a file upload to be committed in batch.
        Also tracks the parent folder for final sync.
        """
        if self.read_only:
            return

        with self._lock:
            self._queue.append((local_path, remote_path))
            
            # Track folder for final consistency check
            # We assume remote_path structure matches local_path parent structure in intent
            # remote_path is full path like "reconstruction/X/Y/file.json"
            # local_path is full path like "/.../results/X/Y/file.json"
            
            # We want to sync the folder /.../results/X/Y to remote reconstruction/X/Y
            remote_parent = str(Path(remote_path).parent)
            local_parent = local_path.parent
            self._watched_folders.add((local_parent, remote_parent))

    def _worker_loop(self):
        """Background daemon polling the queue."""
        while not self._stop_event.is_set():
            time.sleep(1) # Check frequency
            self._check_flush()
            
            # Log sync check
            if time.time() - self._last_log_sync_time > self.FLUSH_INTERVAL:
                self._sync_log_file()
                self._last_log_sync_time = time.time()
            
        # Final flush on stop
        self._check_flush(force=True)
        # Final log sync on stop
        self._sync_log_file()

    def _check_flush(self, force:bool=False):
        batch = []
        with self._lock:
            time_since = time.time() - self._last_flush_time
            is_full = len(self._queue) >= self.BATCH_SIZE
            is_time = time_since >= self.FLUSH_INTERVAL and len(self._queue) > 0
            
            if force or is_full or is_time:
                if self._queue:
                    batch = list(self._queue) # Copy
                    self._queue.clear()
                    self._last_flush_time = time.time()
        
        if batch:
            self._flush_batch(batch)

    def _flush_batch(self, batch: List[tuple[Path, str]]):
        """Executes the batch commit."""
        if self.read_only: return

        count = len(batch)
        logger.info(f"HF Sync: Flushing batch of {count} files...")
        
        try:
            # Prepare operations
            operations = []
            for local, remote in batch:
                if local.exists(): # Safety check
                    operations.append(CommitOperationAdd(
                        path_in_repo=remote,
                        path_or_fileobj=local
                    ))
                else:
                    logger.warning(f"Skipping upload, local file missing: {local}")
            
            if not operations:
                return

            # Commit
            self.api.create_commit(
                repo_id=self.repo_id,
                repo_type=self.repo_type,
                operations=operations,
                commit_message=f"Batch upload {count} results"
            )
            logger.info(f"HF Sync: Successfully committed {count} files.")
            
        except Exception as e:
            logger.error(f"HF Sync: Failed to commit batch: {e}. Retry logic not implemented but files are lost from queue.")
            # In a real robust system, we would put them back in the queue or a retry queue.
            # But for now, we log error. 
            # Re-queueing is risky if it's a persistent error (blocks queue forever).

    def sync_folders(self):
        """
        Performs a final folder synchronization for all tracked folders.
        This ensures that any files missed by the batch queue or lost errors 
        are caught and uploaded.
        """
        if self.read_only:
             logger.info("HF Sync: Read-only mode, skipping final sync.")
             return

        logger.info(f"HF Sync: Starting final consistency check for {len(self._watched_folders)} folders...")
        
        for local_dir, remote_dir in self._watched_folders:
            if not local_dir.exists():
                continue
                
            try:
                # upload_folder is smart: it checks hashes and only uploads changes/missing files.
                self.api.upload_folder(
                    folder_path=str(local_dir),
                    path_in_repo=remote_dir,
                    repo_id=self.repo_id,
                    repo_type=self.repo_type,
                    commit_message=f"Final sync: {remote_dir}"
                )
                logger.info(f"HF Sync: Final sync complete for {remote_dir}")
            except Exception as e:
                logger.error(f"HF Sync: Final sync failed for {remote_dir}: {e}")

    def shutdown(self, wait: bool = True):
        """
        Signals worker to stop, waits for flush, and then performs final folder sync.
        """
        logger.info("Shutting down HF background uploader...")
        self._stop_event.set()
        
        if self._worker_thread:
             if wait:
                 self._worker_thread.join()
        
        # After worker finishes final flush, do folder sync
        self.sync_folders()
            
        logger.info("HF uploader shutdown complete.")
