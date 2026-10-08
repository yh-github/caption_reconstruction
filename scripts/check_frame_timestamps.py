#!/usr/bin/env python3
"""
Check that row t of local/wild_videos_embs_siglip/<vid>.npy is the frame at t seconds.

Extracts frames at exact timestamps with ffmpeg, embeds them with the same SigLIP 2 image tower
(timm vit_base_patch16_siglip_224.v2_webli), and reports which stored row each one matches best.
Used on 2026-10-07 to rule out a frame offset behind the caption lag found by
scripts/caption_vs_siglip_audit.py (caption t matches frame t+1..t+2): all checks gave offset 0.

Usage (from repo root; needs local/wild_videos_raw and ffmpeg):
    .venv/bin/python scripts/check_frame_timestamps.py [n_videos] [out_dir]
"""
import glob
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import timm
import torch
import torch.nn.functional as F
from PIL import Image

N = int(sys.argv[1]) if len(sys.argv) > 1 else 4
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(tempfile.mkdtemp())


def main():
    m = timm.create_model("vit_base_patch16_siglip_224.v2_webli", pretrained=True, num_classes=0).eval()
    tf = timm.data.create_transform(**timm.data.resolve_model_data_config(m), is_training=False)
    vids = sorted(glob.glob("local/wild_videos_raw/**/*.mp4", recursive=True))
    rng = np.random.default_rng(0)
    for path in [vids[i] for i in rng.choice(len(vids), N, replace=False)]:
        vid = Path(path).stem
        npy = np.load(f"local/wild_videos_embs_siglip/{vid}.npy")
        out = []
        for t in (10, 25, 40, 55):
            f = OUT / f"{vid}_{t}.png"
            subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-ss", str(t), "-i", path, "-frames:v", "1", str(f)],
                           check=True)
            with torch.no_grad():
                e = F.normalize(m(tf(Image.open(f).convert("RGB"))[None]), dim=-1)[0].numpy()
            lo = max(0, t - 4)
            sims = npy[lo:t + 5] @ e
            out.append((t, int(np.argmax(sims)) + lo - t, round(float(sims.max()), 3)))
        print(vid, "rows", len(npy), "(t, best row offset, cos):", out)


if __name__ == "__main__":
    main()
