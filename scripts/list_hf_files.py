
from huggingface_hub import HfApi

repo_id = "Y3/dense_video_captions"
api = HfApi()

print(f"Listing files in {repo_id}...")
try:
    files = api.list_repo_files(repo_id=repo_id, repo_type="dataset")
    recon_files = [f for f in files if "reconstruction" in f and "phi-3" in f and "t=1.5" in f]
    
    print(f"Found {len(recon_files)} phi-3 reconstruction files.")
    for f in recon_files[:20]: # Show first 20
        print(f)
        
except Exception as e:
    print(f"Error: {e}")
