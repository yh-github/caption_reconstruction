#!/usr/bin/env python
"""
Kaggle runner for the forced-choice gap test (scripts/forced_choice_gap.py), one worker per GPU.

Paste this whole file into a Kaggle notebook cell (GPU T4 x2, Internet on, HF_TOKEN secret set).
Scores are uploaded to the HF dataset repo every UPLOAD_EVERY items and at the end, so a session
that times out loses little; re-running the notebook resumes from what is already on HF.

Set PILOT_LIMIT = 20 for a first check (20 items per GPU), then None for the full run.
"""
import os
import subprocess
import sys

MODEL_KEY = "llama-3.1-8b"
PILOT_LIMIT = 20          # items per worker; None = full run (1,325 items in total)
UPLOAD_EVERY = 50         # items per worker between HF uploads
MAX_RUNTIME_HOURS = 11.0  # Kaggle sessions stop at 12 h

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
REPO_URL = "https://github.com/yh-github/caption_reconstruction.git"
REPO_DIR = "caption_reconstruction"

print("=== Forced-choice gap test: Kaggle runner ===")
if not os.path.exists(REPO_DIR):
    subprocess.check_call(["git", "clone", REPO_URL])
else:
    subprocess.check_call(["git", "-C", REPO_DIR, "pull"])
os.chdir(REPO_DIR)

subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements_colab.txt",
                       "--extra-index-url", "https://download.pytorch.org/whl/cu121"])
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "-e", ".", "--no-deps"])

import torch  # noqa: E402

if not torch.cuda.is_available():
    raise RuntimeError("No CUDA GPU visible; enable a GPU accelerator for this notebook.")

try:
    from kaggle_secrets import UserSecretsClient
    from huggingface_hub import login
    login(token=UserSecretsClient().get_secret("HF_TOKEN"))
    print("Logged into Hugging Face.")
except Exception as e:
    raise RuntimeError(f"HF_TOKEN secret is required to upload scores: {e}")

num_gpus = torch.cuda.device_count()
for i in range(num_gpus):
    p = torch.cuda.get_device_properties(i)
    print(f"  [GPU {i}] {p.name} | {p.total_memory / 1024 ** 3:.1f} GB")

# Pull any scores already on HF once, before the workers start, so they skip finished items.
subprocess.check_call([sys.executable, "-c",
                       f"import sys; sys.argv=['x']; sys.path.insert(0, 'scripts');"
                       f"import forced_choice_gap as f; f.download_llm_files('{MODEL_KEY}')"])

base = [sys.executable, "-u", "scripts/forced_choice_gap.py", "llm", "--model-key", MODEL_KEY, "--upload",
        "--upload-every", str(UPLOAD_EVERY), "--max-runtime-hours", str(MAX_RUNTIME_HOURS),
        "--total-workers", str(num_gpus)]
if PILOT_LIMIT:
    base += ["--limit", str(PILOT_LIMIT)]

procs = []
for i in range(num_gpus):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(i), PYTHONUNBUFFERED="1")
    print(f"Starting worker {i} on GPU {i}")
    procs.append((i, subprocess.Popen(base + ["--worker-id", str(i)], env=env)))

failed = [(i, rc) for i, p in procs if (rc := p.wait()) != 0]
if failed:
    raise RuntimeError(f"Workers failed: {failed}")
print("=== All workers finished; scores are on HF under forced_choice/ ===")
