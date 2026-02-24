"""Check if any local mp4s don't appear in dev.json or test.json."""
import json, os, sys

def normalize(name):
    return name.replace("'", "_").lower()

# Build local cache
local_dirs = ["local/wild_videos_raw/Videos1/", "local/wild_videos_raw/Videos2/"]
local_stems = {}
for d in local_dirs:
    if not os.path.exists(d): continue
    for root, _, files in os.walk(d):
        for f in files:
            if f.endswith('.mp4'):
                stem = f.rsplit('.', 1)[0]
                local_stems[normalize(stem)] = stem

# Collect all video_ids from both JSONs
json_ids = set()
for jf in ["datasets/wildQA/dev.json", "datasets/wildQA/test.json"]:
    if not os.path.exists(jf): continue
    with open(jf) as f:
        for e in json.load(f):
            json_ids.add(normalize(e["video_id"]))

orphan_local = {k: v for k, v in local_stems.items() if k not in json_ids}

print(f"Local files scanned: {len(local_stems)}", flush=True)
print(f"Unique JSON video_ids (dev+test): {len(json_ids)}", flush=True)
print(f"Local files NOT in any JSON: {len(orphan_local)}/{len(local_stems)}", flush=True)

if orphan_local:
    print("\n--- Orphaned Local Files ---", flush=True)
    for norm, orig in sorted(orphan_local.items()):
        print(f"  {orig}", flush=True)
else:
    print("\nNone — all local files are referenced in the datasets.", flush=True)

sys.exit(0)
