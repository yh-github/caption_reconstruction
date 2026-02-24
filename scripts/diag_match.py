"""Check if any local mp4s don't appear in dev.json or test.json."""
import json, os

def normalize(name):
    return name.replace("'", "_").lower()

# Build local cache
local_dirs = ["local/wild_videos_raw/Videos1/", "local/wild_videos_raw/Videos2/"]
local_stems = {}
for d in local_dirs:
    for root, _, files in os.walk(d):
        for f in files:
            if f.endswith('.mp4'):
                stem = f.rsplit('.', 1)[0]
                local_stems[normalize(stem)] = stem

# Collect all video_ids from both JSONs
json_ids = set()
for jf in ["datasets/wildQA/dev.json", "datasets/wildQA/test.json"]:
    with open(jf) as f:
        for e in json.load(f):
            json_ids.add(normalize(e["video_id"]))

orphan_local = {k: v for k, v in local_stems.items() if k not in json_ids}

print(f"Local files: {len(local_stems)}")
print(f"JSON video_ids (dev+test): {len(json_ids)}")
print(f"Local files NOT in any JSON: {len(orphan_local)}/{len(local_stems)}")
if orphan_local:
    for norm, orig in sorted(orphan_local.items()):
        print(f"  {orig}")
else:
    print("  None — all local files are referenced.")
