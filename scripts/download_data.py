
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
        backup_path = path.with_name(f"{path.name}_backup_{get_datetime_str()}")
        logger.info(f"Backing up {path} to {backup_path}")
        shutil.move(str(path), str(backup_path))

def merge_disk_caches(target_path: Path, source_path: Path):
    """
    Smart merge of disk caches.
    1. Determine baseline based on size (use larger as base).
    2. Overwrite priority: LOCAL ALWAYS WINS.
    """
    logger.info(f"Merging disk caches. Target: {target_path}, Source: {source_path}")
    
    # Ensure source exists (it should, from unzip)
    if not source_path.exists():
        logger.warning(f"Source cache {source_path} does not exist. Skipping merge.")
        return

    # Scenario: Target (Local) does not exist -> Init from source
    if not target_path.exists():
        logger.info("Local cache not found. Moving source to target.")
        shutil.move(str(source_path), str(target_path))
        return

    # Calculate sizes
    size_target = sum(f.stat().st_size for f in target_path.glob('**/*') if f.is_file())
    size_source = sum(f.stat().st_size for f in source_path.glob('**/*') if f.is_file())

    logger.info(f"Size Local: {size_target/1024/1024:.2f}MB, Size Remote: {size_source/1024/1024:.2f}MB")

    # Temporary merge directory
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_merge_path = Path(temp_dir) / "merged_cache"
        
        # Decide Baseline
        if size_target >= size_source:
             logger.info("Local is larger. Using Local as baseline.")
             # Copy Local to Temp
             shutil.copytree(target_path, temp_merge_path)
             baseline_cache = diskcache.Cache(str(temp_merge_path))
             supplementary_cache = diskcache.Cache(str(source_path))
             
             # Iterate Remote (Source) and add ONLY new keys
             count = 0
             for key in supplementary_cache:
                 if key not in baseline_cache:
                     baseline_cache[key] = supplementary_cache[key]
                     count += 1
             logger.info(f"Added {count} new keys from remote cache.")
             
        else:
             logger.info("Remote is larger. Using Remote as baseline.")
             # Copy Remote to Temp
             shutil.copytree(source_path, temp_merge_path)
             baseline_cache = diskcache.Cache(str(temp_merge_path))
             supplementary_cache = diskcache.Cache(str(target_path)) # Local is supplementary but PRIORITY

             # Iterate Local (Target) and OVERWRITE/ADD all keys
             count = 0
             for key in supplementary_cache:
                 baseline_cache[key] = supplementary_cache[key] # Overwrite ensures local wins
                 count += 1
             logger.info(f"Overwrote/Added {count} keys from local cache.")

        baseline_cache.close()
        supplementary_cache.close()
        
        # Deploy Merged Cache
        backup_directory(target_path)
        shutil.move(str(temp_merge_path), str(target_path))

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
    parser.parse_args()

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
    temp_download_dir = Path("temp_downloads")
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
            
            # Heuristic: if one folder inside matches target name, use that.
            subdirs = [d for d in extract_path.iterdir() if d.is_dir()]
            if len(subdirs) == 1 and subdirs[0].name == target_path.name:
                 content_root = subdirs[0]
            elif len(subdirs) == 1 and subdirs[0].name == zip_name.replace(".zip", ""): # fallback
                 content_root = subdirs[0]
            
            # Specifically for disk_cache, we zipped 'disk_cache/' so it will create a folder 'disk_cache'
            # For wild_videos_embs, we zipped 'local/wild_videos_embs/', creating that structure? 
            # Wait, command was: cd .../local/ && zip ... wild_videos_embs/
            # So it contains 'wild_videos_embs/...' 
            # target is 'local/wild_videos_embs'
            # So content_root should be extract_path / 'wild_videos_embs'
            
            potential_root = extract_path / target_path.name
            if potential_root.exists():
                content_root = potential_root
            
            merge_func(target_path, content_root)
            update_state(task_key)

    logger.info("All tasks completed.")
    # shutil.rmtree(temp_download_dir) # Optional cleanup

if __name__ == "__main__":
    main()
