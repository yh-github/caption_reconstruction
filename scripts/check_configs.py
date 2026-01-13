
from huggingface_hub import HfApi
import re

repo_id = "Y3/dense_video_captions"
api = HfApi()

print(f"Listing directories in {repo_id}...")
files = api.list_repo_files(repo_id=repo_id, repo_type="dataset")

# Filter for folders (implicitly by unique paths)
# We look for "wild_dev_sim_text/phi-3__t=1.5"
configs = set()
pattern = re.compile(r"phi-3__t=([\d\.]+).*?\(w=(\d+), i=(\d+)\)")

for f in files:
    match = pattern.search(f)
    if match:
        configs.add(f"t={match.group(1)} w={match.group(2)}, i={match.group(3)}")

print(f"Found {len(configs)} configurations:")
for c in sorted(list(configs)):
    print(c)
