import json
import logging
import time
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.utils import EntryNotFoundError

logger = logging.getLogger(__name__)

class HFResultsSync:
    def __init__(self, repo_id: str, run_name: str, hyperparams_hash: str, output_dir: Path):
        self.repo_id = repo_id
        self.repo_type = "dataset"
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Consistent filename without timestamp for sync
        self.filename = f"scores_{run_name}_{hyperparams_hash}.json"
        self.local_path = self.output_dir / self.filename
        
        # Remote path can be used to organize files
        self.remote_path = self.filename 

        self.api = HfApi()

    def pull(self, force_download: bool = False) -> dict[str, Any]:
        """
        Attempts to download the existing file from HF.
        Returns the data if found, else returns empty dict.
        """
        logger.info(f"Attempting to pull existing results from {self.repo_id}/{self.remote_path}...")
        try:
            downloaded_path = hf_hub_download(
                repo_id=self.repo_id,
                filename=self.remote_path,
                repo_type=self.repo_type,
                local_dir=self.output_dir, # Download directly to our working dir
                force_download=force_download
                # local_dir_use_symlinks=False # Get the actual file
            )
            # hf_hub_download might create a subfolder structure if we don't be careful, 
            # but usually it puts it in local_dir/filename if it's flat.
            # actually hf_hub_download with local_dir preserves directory structure.
            # If self.remote_path is just a filename, it will be at local_dir/filename.
            
            # Note: hf_hub_download creates a cache structure if local_dir is not used in a specific way,
            # but here we want to modify it. Reading from the returned path is safest.
            
            with open(downloaded_path, 'r') as f:
                data = json.load(f)
            
            logger.info(f"Successfully pulled existing data ({len(data.get('scores', {}))} videos scored).")
            return data

        except EntryNotFoundError:
            logger.info("No existing result file found on remote. Starting fresh.")
            return {}
        except Exception as e:
            logger.warning(f"Could not pull from HF (might be auth or connection): {e}")
            logger.info("Continuing with empty base.")
            return {}

    def push(self, data: dict[str, Any], commit_message: str = "Update scores"):
        """
        Saves data locally and uploads to HF.
        """
        # 1. Save locally
        with open(self.local_path, 'w') as f:
            json.dump(data, f, indent=2)
        logger.info(f"Saved locally to {self.local_path}")

        # 2. Upload
        logger.info(f"Uploading to {self.repo_id}...")
        try:
            self.api.upload_file(
                path_or_fileobj=self.local_path,
                path_in_repo=self.remote_path,
                repo_id=self.repo_id,
                repo_type=self.repo_type,
                commit_message=commit_message
            )
            logger.info("Upload successful.")
        except Exception as e:
            logger.error(f"Failed to upload to HF: {e}")

    def merge_results(self, existing_data: dict[str, Any], new_results: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
        """
        Merges new results into existing data. 
        Updates metadata if needed.
        """
        merged = existing_data.copy()
        
        # Ensure structure
        if "scores" not in merged:
            merged["scores"] = {}
        if "metadata" not in merged:
            merged["metadata"] = {}

        # Merge scores
        for video_id, result in new_results.items():
            merged["scores"][video_id] = result
            
        # Update metadata
        merged["metadata"]["last_updated"] = time.time()
        merged["metadata"]["config"] = config # Always keep latest config
        # We could merge other metadata like run_name, etc if they differ (shouldn't if hash matched)
        
        return merged
