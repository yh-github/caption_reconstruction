
from huggingface_hub import HfApi, hf_hub_download
import os
import re
from concurrent.futures import ThreadPoolExecutor

REPO_ID = "Y3/dense_video_captions"
TARGET_DIR = "results/recon/manual_download"
API = HfApi()

def get_files_to_download():
    print(f"Listing files in {REPO_ID}...")
    all_files = API.list_repo_files(repo_id=REPO_ID, repo_type="dataset")
    # Filter for phi-3 t=1.5
    # The structure is reconstruction/RUN_NAME/VIDEO.json
    # We want run names containing "phi-3" and "t=1.5"
    target_files = [
        f for f in all_files 
        if "reconstruction" in f 
        and "phi-3" in f 
        and "t=0.1" in f
        and f.endswith(".json")
    ]
    return target_files

def download_file(remote_path):
    try:
        # remote_path: reconstruction/wild_dev_sim_text/phi-3.../video.json
        # we want to save it structurally to preserve the run name folder
        
        # Local structure: results/recon/manual_download/phi-3.../video.json
        # We strip 'reconstruction/wild_dev_sim_text/' prefix to keep it clean?
        # Or just keep relative structure.
        
        # Let's mirror the HF structure partially to distinguish runs
        # HF: reconstruction/wild_dev_sim_text/RUN_NAME/file
        # Local: TARGET_DIR/RUN_NAME/file
        
        parts = remote_path.split('/')
        # parts[0] = reconstruction
        # parts[1] = wild_dev_sim_text
        # parts[2] = RUN_NAME (e.g. phi-3__...)
        # parts[3] = filename
        
        if len(parts) < 4:
            print(f"Skipping weird path: {remote_path}")
            return
            
        run_name = parts[2]
        filename = parts[3]
        
        local_dir = os.path.join(TARGET_DIR, run_name)
        
        hf_hub_download(
            repo_id=REPO_ID,
            filename=remote_path,
            repo_type="dataset",
            local_dir=TARGET_DIR, # This reconstructs full path? No.
            # hf_hub_download with local_dir uses the structure relative to repo root IF local_dir_use_symlinks is True?
            # Actually if we specify local_dir, it mirrors the filename structure?
            # e.g. filename="A/B/C", local_dir="D" -> "D/A/B/C"
            # We want "D/C" (flattened run) or "D/RUN/C"
        )
        # To avoid deep nesting "reconstruction/wild_dev_sim_text/...", let's doing manual moving
        # OR just accept the nesting. Nesting is safer.
        
        # Simpler: just use hf_hub_download default behavior to mirror repo
        # It will be results/recon/manual_download/reconstruction/wild_dev_sim_text/RUN_NAME/file
        # That's fine.
    except Exception as e:
        print(f"Failed to download {remote_path}: {e}")

def main():
    files = get_files_to_download()
    print(f"Found {len(files)} files to download.")
    
    # Check if already filtered
    # Extract unique run names
    runs = set(f.split('/')[2] for f in files if len(f.split('/')) > 2)
    print(f"Unique runs found: {len(runs)}")
    for r in runs:
        print(f" - {r}")
        
    print("Starting download...")
    # Use ThreadPool for speed
    with ThreadPoolExecutor(max_workers=10) as executor:
        # We pass the same args
        # But hf_hub_download call needs to be wrapped
        futures = []
        for f in files:
            futures.append(
                executor.submit(
                    hf_hub_download,
                    repo_id=REPO_ID,
                    filename=f,
                    repo_type="dataset",
                    local_dir=TARGET_DIR
                )
            )
            
        # Wait for all
        for fut in futures:
            try:
                fut.result()
            except Exception as e:
                print(f"Download error: {e}")
                
    print("Download complete.")

if __name__ == "__main__":
    main()
