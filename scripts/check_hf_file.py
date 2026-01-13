
from huggingface_hub import hf_hub_download
import json

repo_id = "Y3/dense_video_captions"
filename = "reconstruction/wild_dev_sim_text/phi-3__t=1.5_rp=1.2__fixed_fill(w=3, i=0)/Bertram-Craft_2-clip-3.json"

print(f"Downloading {filename}...")
local_path = hf_hub_download(
    repo_id=repo_id,
    filename=filename,
    repo_type="dataset",
    local_dir="results/temp_check"
)

with open(local_path, 'r') as f:
    data = json.load(f)
    print(json.dumps(data, indent=2))
