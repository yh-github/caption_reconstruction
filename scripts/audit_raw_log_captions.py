import re
import json
from pathlib import Path
import os

WILD4_GT_DIR = Path('datasets/wildQA/captions__wild4')
gt_map = {}
for p in WILD4_GT_DIR.glob('*.json'):
    if p.name == 'categories.json': continue
    try:
        with open(p) as f:
            gt_map[p.stem] = [c.get('caption', '') for c in json.load(f).get('captions', [])]
    except Exception:
        pass

from huggingface_hub import hf_hub_download
token = os.environ.get('HF_TOKEN')
repo_id = 'Y3/dense_video_captions'

for log_name in ['3cbbdf14af3e49adbde783a0b91474a5.log', '6f9240301fa3459daa48da58f5f10ac4.log']:
    print(f"\n========================================")
    print(f"PARSING LOG: {log_name}")
    print(f"========================================")
    log_file = hf_hub_download(repo_id=repo_id, repo_type='dataset', filename=f'logs/{log_name}', token=token, force_download=True)
    with open(log_file) as f:
        content = f.read()

    # Split by: Bad indices found in reconstructed_video
    chunks = content.split("Bad indices found in reconstructed_video ")
    print(f"Found {len(chunks)-1} video chunks in log.")

    for chunk in chunks[1:8]:
        vid = chunk.split(",")[0].strip()
        # Find preceding input_value='...
        # Look in the preceding text of chunk (or at end of prev chunk)
        # In this chunk:
        # Actually input_value is logged right before "Bad indices"
        # Let's search for objects with {"index": ..., "caption": ...}
        # in the whole chunk or preceding
        gt_caps = gt_map.get(vid, [])
        print(f"\n--- Video: {vid} (Missing: [0, 1, 2]) ---")
        if gt_caps and len(gt_caps) > 4:
            print(f"  [Context t=03] {gt_caps[3]}")
            print(f"  [Context t=04] {gt_caps[4]}")
        print("  " + "-"*60)
        
        # Extract indices and captions from chunk
        matches = re.findall(r'\{[^{}]*?"index"\s*:\s*(\d+)[^{}]*?"caption"\s*:\s*"([^"\\]*(?:\\.[^"\\]*)*)"', chunk)
        if not matches:
            # Maybe it was in the text before "Bad indices"
            # Let's search broadly
            matches = re.findall(r'\{[^{}]*?"index"\s*:\s*(\d+)[^{}]*?"caption"\s*:\s*"([^"\\]*(?:\\.[^"\\]*)*)"', content[:content.find(f"reconstructed_video {vid}")][-2000:])
        
        for idx_str, cap in matches[:3]:
            idx = int(idx_str)
            gt_text = gt_caps[idx] if idx < len(gt_caps) else "(None)"
            print(f"  [t={idx:02d}] GT:    {gt_text}")
            print(f"        Llama: {cap}")
        print("  " + "-"*60)
