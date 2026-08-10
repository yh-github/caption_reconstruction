#!/usr/bin/env python
import argparse
import json
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Generate Kaggle CLI submission kernel directory.")
    parser.add_argument("--config", type=str, required=True, help="Path to experiment config (e.g. config/embs_vs_slms/wild4_sim_text.yaml)")
    parser.add_argument("--worker-id", type=int, default=0, help="Worker ID for dataset partitioning (default: 0)")
    parser.add_argument("--total-workers", type=int, default=1, help="Total number of workers (default: 1)")
    parser.add_argument("--hf-token", type=str, default=None, help="Hugging Face write token (injected into generated kernel)")
    parser.add_argument("--max-runtime-hours", type=float, default=8.0, help="Max runtime before graceful exit (default: 8.0)")
    parser.add_argument("--username", type=str, default="kaggle_user", help="Your Kaggle username")
    parser.add_argument("--output-dir", type=str, default=None, help="Directory to create (default: kaggle_kernel_w{worker_id})")

    args = parser.parse_args()

    output_dir = Path(args.output_dir or f"kaggle_kernel_w{args.worker_id}")
    output_dir.mkdir(parents=True, exist_ok=True)

    kernel_slug = f"caption-reconstruction-w{args.worker_id}"

    # 1. kernel-metadata.json
    metadata = {
        "id": f"{args.username}/{kernel_slug}",
        "title": f"Caption Reconstruction Worker {args.worker_id}",
        "code_file": "run_kaggle.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": "true",
        "enable_gpu": "true",
        "accelerator": "nvidiaTeslaT4",
        "enable_internet": "true",
        "dataset_sources": [],
        "competition_sources": [],
        "kernel_sources": []
    }

    with open(output_dir / "kernel-metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    hf_token_code = ""
    if args.hf_token:
        hf_token_code = f'''
# Login to HF via CLI argument
from huggingface_hub import login
login(token="{args.hf_token}")
print("Logged into Hugging Face via provided HF_TOKEN successfully.")
'''
    else:
        hf_token_code = '''
# Load HF_TOKEN from Kaggle Secrets or environment if available
try:
    from kaggle_secrets import KaggleSecrets
    secrets = KaggleSecrets()
    hf_token = secrets.get_secret("HF_TOKEN")
    if hf_token:
        from huggingface_hub import login
        login(token=hf_token)
        print("Logged into Hugging Face via Kaggle Secrets successfully.")
    else:
        print("Warning: 'HF_TOKEN' key not found in Kaggle Secrets.")
except Exception as e:
    print(f"Note: Could not retrieve HF_TOKEN from Kaggle Secrets: {e}")
'''

    # 2. run_kaggle.py
    run_kaggle_content = f'''#!/usr/bin/env python
import os
import sys
import subprocess

print("=== Starting Kaggle Caption Reconstruction Execution ===")
print(f"Python: {{sys.executable}}")
print(f"Worker ID: {args.worker_id} / Total Workers: {args.total_workers}")
print(f"Config: {args.config}")

REPO_URL = "https://github.com/yh-github/caption_reconstruction.git"
REPO_DIR = "caption_reconstruction"

if not os.path.exists(REPO_DIR):
    print(f"Cloning repo from {{REPO_URL}}...")
    subprocess.check_call(["git", "clone", REPO_URL])
else:
    print(f"Updating repo in {{REPO_DIR}}...")
    subprocess.check_call(["git", "-C", REPO_DIR, "pull"])

os.chdir(REPO_DIR)

print("Installing dependencies...")
subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements_colab.txt"])
subprocess.check_call([sys.executable, "-m", "pip", "install", "-e", ".", "--no-deps"])
{hf_token_code}
# Launch Main Pipeline
cmd = [
    sys.executable, "src/main.py",
    "{args.config}",
    "--worker-id", "{args.worker_id}",
    "--total-workers", "{args.total_workers}",
    "--max-runtime-hours", "{args.max_runtime_hours}"
]

print(f"Executing command: {{' '.join(cmd)}}")
subprocess.check_call(cmd)
print("=== Kaggle Execution Completed Successfully ===")
'''

    with open(output_dir / "run_kaggle.py", "w") as f:
        f.write(run_kaggle_content)

    print(f"Successfully prepared Kaggle kernel directory in '{output_dir}'.")
    print("\nTo deploy via Kaggle CLI, run:")
    print(f"  cd {output_dir}")
    print("  kaggle kernels push")

if __name__ == "__main__":
    main()
