#!/usr/bin/env python
"""
Kaggle runner for the bridge-ceiling captions (scripts/bridge_ceiling.py caption): Florence-2 on 1,800 frames.

Paste this whole file into a Kaggle notebook cell (GPU T4 x1 is enough, Internet on, HF_TOKEN secret set).
Before running: locally, `scripts/bridge_ceiling.py extract` and `push-frames` (uploads frames.zip to the private
HF dataset repo). The notebook downloads the frames, captions them, and uploads captions.jsonl back to the repo;
then locally run `pull-captions` and `eval`. Re-running resumes: frames already captioned are skipped.
"""
import os
import subprocess
import sys

REPO_URL = "https://github.com/yh-github/caption_reconstruction.git"
REPO_DIR = "caption_reconstruction"
# Florence2ForConditionalGeneration is built into recent transformers; this is the version verified locally.
TRANSFORMERS = "transformers==5.14.1"

print("=== Bridge ceiling: Florence-2 per-frame captions (Kaggle) ===")
if not os.path.exists(REPO_DIR):
    subprocess.check_call(["git", "clone", REPO_URL])
else:
    subprocess.check_call(["git", "-C", REPO_DIR, "pull"])
os.chdir(REPO_DIR)
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", TRANSFORMERS, "huggingface_hub", "pillow"])

import torch  # noqa: E402

if not torch.cuda.is_available():
    raise RuntimeError("No CUDA GPU visible; enable a GPU accelerator for this notebook.")
print(f"GPU: {torch.cuda.get_device_name(0)}")

try:
    from kaggle_secrets import UserSecretsClient
    from huggingface_hub import login
    login(token=UserSecretsClient().get_secret("HF_TOKEN"))
except Exception as e:
    raise RuntimeError(f"HF_TOKEN secret is required (private repo): {e}")

run = lambda *a: subprocess.check_call([sys.executable, "-u", "scripts/bridge_ceiling.py", *a])
run("pull-frames")
try:
    run("pull-captions")  # resume: skip frames captioned in an earlier session
except subprocess.CalledProcessError:
    print("no earlier captions on HF; starting fresh")
run("caption", "--device", "cuda", "--batch", "32")
run("push-captions")
print("done: locally run `scripts/bridge_ceiling.py pull-captions` then `eval`")
