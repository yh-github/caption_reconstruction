
import json
import logging
import shutil
import zipfile
from pathlib import Path
import tempfile
import time

import gdown
import diskcache


import yaml
from common_utils.tracking import get_datetime_str

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

STATE_FILE = Path("local/.download_state.json")

def load_state() -> dict:
    if STATE_FILE.exists():
        with open(STATE_FILE, "r") as f:
            return json.load(f)
    return {}

def save_state(state: dict):
    if not STATE_FILE.parent.exists():
        STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=4)

def update_state(key: str, value: bool = True):
    state = load_state()
    state[key] = value
    save_state(state)

def is_done(key: str) -> bool:
    return load_state().get(key, False)

def download_folder(url: str, output_dir: Path):
    if not output_dir.exists():
        output_dir.mkdir(parents=True)
    gdown.download_folder(url=url, output=str(output_dir), quiet=False, use_cookies=False)

def extract_zip(zip_path: Path, output_dir: Path):
    logger.info(f"Extracting {zip_path} to {output_dir}")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(output_dir)

def backup_directory(path: Path):
    if path.exists():
        backup_dir = Path("local/backup")
        backup_dir.mkdir(parents=True, exist_ok=True)
        backup_path = backup_dir / path.with_name(f"{path.name}_backup_{get_datetime_str()}")
        logger.info(f"Backing up {path} to {backup_path}")
        shutil.move(str(path), str(backup_path))

def merge_disk_caches(target_path: Path, source_path: Path):
    """
    Overwrite local cache with remote cache (safer for reproduction).
    1. Backup existing local cache.
    2. Replace with source.
    """
    logger.info(f"Overwriting disk cache. Target: {target_path}, Source: {source_path}")
    
    # Ensure source exists
    if not source_path.exists():
        logger.warning(f"Source cache {source_path} does not exist. Skipping overwrite.")
        return

    # Backup and Remove Target if exists
    if target_path.exists():
        backup_directory(target_path)
        if target_path.exists(): # precise check if backup moved it or copied
             # If backup moved it, it's gone. If backup copied, we need to delete.
             # backup_directory uses shutil.move, so it handles removal.
             # But let's be safe.
             pass
    
    # Move source to target
    if not target_path.parent.exists():
        target_path.parent.mkdir(parents=True)
        
    shutil.move(str(source_path), str(target_path))
    logger.info("Cache overwrite complete.")

def process_file_dir_merge(target_dir: Path, source_dir: Path):
    """
    Simple file-level merge.
    If file exists in target, SKIP (Local wins).
    If missing, copy from source.
    """
    logger.info(f"Merging files from {source_dir} to {target_dir}")
    if not target_dir.exists():
        shutil.move(str(source_dir), str(target_dir))
        return

    for item in source_dir.rglob("*"):
        if item.is_file():
            rel_path = item.relative_to(source_dir)
            target_item = target_dir / rel_path
            
            if not target_item.exists():
                target_item.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(str(item), str(target_item))
                # logger.debug(f"Copied {rel_path}")

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Download and setup project data from remote cache.")
    parser.add_argument("--force", action="store_true", help="Ignore state file and force redownload/reprocess.")
    args = parser.parse_args()

    if args.force:
        logger.info("Force flag set. ignoring download state.")
        if STATE_FILE.exists():
            STATE_FILE.unlink()

    # Load system config directly
    config_path = Path("config/system.yaml")
    if not config_path.exists():
        logger.error(f"Config file not found: {config_path}")
        return

    with open(config_path, "r") as f:
         config = yaml.safe_load(f)

    remote_url = config.get("paths", {}).get("remote_cache_url")
    
    if not remote_url:
        logger.error("remote_cache_url not found in config/system.yaml (under 'paths')")
        return

    # 1. Download
    temp_download_dir = Path("local/temp_downloads")
    temp_download_dir.mkdir(parents=True, exist_ok=True)
    if not is_done("download_complete"):
        logger.info("Starting Download...")
        download_folder(remote_url, temp_download_dir)
        update_state("download_complete")
    else:
        logger.info("Download already complete (found in state).")

    # Mapping: Zip Name -> (Target Dir, Merge Function)
    tasks = [
        ("disk_cache.zip", Path(config["paths"]["disk_cache"]), merge_disk_caches),
        ("wild_videos_embs.zip", Path("local/wild_videos_embs"), process_file_dir_merge),
        ("results.zip", Path(config["paths"]["results"]), process_file_dir_merge)
    ]

    for zip_name, target_path, merge_func in tasks:
        zip_file = temp_download_dir / zip_name
        task_key = f"processed_{zip_name}"
        
        if is_done(task_key):
             logger.info(f"Task {zip_name} already processed.")
             continue
             
        if not zip_file.exists():
            logger.warning(f"Zip file {zip_name} not found in downloads. Skipping.")
            continue
            
        logger.info(f"Processing {zip_name}...")
        
        with tempfile.TemporaryDirectory() as extract_dir:
            extract_path = Path(extract_dir)
            extract_zip(zip_file, extract_path)
            
            # The zip likely contains the folder itself (e.g. disk_cache/...), so let's find the root inside
            # Assumption based on how we zipped it: zip -r dist/disk_cache.zip disk_cache/
            # So contents are: disk_cache/file1, disk_cache/file2...
            # We want to pass the INNER directory to the merge function if it matches the target name, 
            # Or just pass the extraction root if the zip contents are flat.
            # Let's verify by checking if a single directory exists matching the target stem or name.
            
            content_root = extract_path
            
            # Improved logic: find the inner folder matching the target name
            # This handles cases like zip containing 'local/wild_videos_embs/' vs just 'wild_videos_embs/'
            candidates = [p for p in extract_path.rglob(target_path.name) if p.is_dir()]
            
            if candidates:
                # Use the first match. Should be unique enough.
                content_root = candidates[0]
                logger.info(f"Found content root inside zip: {content_root}")
            else:
                # Fallback: check if the root folder matches zip name (e.g. disk_cache dir inside disk_cache.zip)
                zip_stem = zip_name.replace(".zip", "")
                candidates_stem = [p for p in extract_path.rglob(zip_stem) if p.is_dir()]
                if candidates_stem:
                    content_root = candidates_stem[0]
                    logger.info(f"Found content root by zip name: {content_root}")
                
                 # Else remaining default is extract_path itself
            
            merge_func(target_path, content_root)
            update_state(task_key)

    logger.info("All tasks completed.")
    # shutil.rmtree(temp_download_dir) # Optional cleanup

if __name__ == "__main__":
    main()
