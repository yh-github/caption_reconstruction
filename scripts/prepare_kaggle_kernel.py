#!/usr/bin/env python
import argparse
import json
import os
from pathlib import Path

def print_gpu_audit():
    """Audits and prints GPU count, model name, and memory to stdout."""
    print("--- GPU Hardware Audit ---")
    try:
        import torch
        if torch.cuda.is_available():
            num_gpus = torch.cuda.device_count()
            print(f"Detected CUDA GPUs: {num_gpus}")
            for i in range(num_gpus):
                props = torch.cuda.get_device_properties(i)
                vram_gb = props.total_memory / (1024 ** 3)
                print(f"  [GPU {i}] {props.name} | Total Memory: {vram_gb:.2f} GB | SM: {props.major}.{props.minor}")
        else:
            print("No CUDA GPUs detected on local system.")
    except Exception as e:
        print(f"Could not inspect CUDA devices: {e}")
    print("---------------------------\n")

def main():
    # Auto-load username from local/kaggle.json if available
    default_username = "kaggle_user"
    local_kaggle_path = Path("local/kaggle.json")
    if local_kaggle_path.exists():
        try:
            with open(local_kaggle_path, "r") as f:
                k_data = json.load(f)
                default_username = k_data.get("username", default_username)
        except Exception:
            pass

    parser = argparse.ArgumentParser(description="Generate Kaggle CLI submission kernel directory.")
    parser.add_argument("--config", type=str, required=True, help="Path to experiment config (e.g. config/embs_vs_slms/wild4_sim_text.yaml)")
    parser.add_argument("--worker-id", type=int, default=0, help="Worker ID for dataset partitioning (default: 0)")
    parser.add_argument("--total-workers", type=int, default=1, help="Total number of workers (default: 1)")
    parser.add_argument("--multi-gpu", action="store_true", help="Auto-detect GPUs at runtime and launch 1 parallel worker per GPU")
    parser.add_argument("--hf-token", type=str, default=None, help="Hugging Face write token (injected into generated kernel)")
    parser.add_argument("--max-runtime-hours", type=float, default=8.0, help="Max runtime before graceful exit (default: 8.0)")
    parser.add_argument("--username", type=str, default=default_username, help="Your Kaggle username")
    parser.add_argument("--output-dir", type=str, default=None, help="Directory to create (default: local/kaggle_kernel_multi_gpu or local/kaggle_kernel_w{worker_id})")

    args = parser.parse_args()

    print_gpu_audit()

    output_dir = Path(args.output_dir or (f"local/kaggle_kernel_multi_gpu" if args.multi_gpu else f"local/kaggle_kernel_w{args.worker_id}"))
    output_dir.mkdir(parents=True, exist_ok=True)

    kernel_slug = f"caption-reconstruction-multi-gpu" if args.multi_gpu else f"caption-reconstruction-w{args.worker_id}"

    # 1. kernel-metadata.json
    metadata = {
        "id": f"{args.username}/{kernel_slug}",
        "title": f"Caption Reconstruction {'Multi-GPU' if args.multi_gpu else f'Worker {args.worker_id}'}",
        "code_file": "run_kaggle.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": True,
        "machine_shape": "t4X2" if args.multi_gpu else "nvidiaTeslaT4",
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

    if args.multi_gpu:
        launch_code = f'''
# Auto-detect available GPUs and launch 1 parallel worker per GPU
import torch
print("\\n=== GPU Hardware Audit & Multi-GPU Launch ===")
num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
print(f"Detected CUDA GPUs: {{num_gpus}}")
for i in range(num_gpus):
    props = torch.cuda.get_device_properties(i)
    vram_gb = props.total_memory / (1024 ** 3)
    print(f"  [GPU {{i}}] {{props.name}} | Total Memory: {{vram_gb:.2f}} GB | Capability: {{props.major}}.{{props.minor}}")
print("=============================================\\n")

if num_gpus > 1:
    print(f"=== Launching {{num_gpus}} Parallel Workers (1 per GPU) ===")
    processes = []
    for i in range(num_gpus):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(i))
        cmd = [
            sys.executable, "src/main.py",
            "{args.config}",
            "--worker-id", str(i),
            "--total-workers", str(num_gpus),
            "--max-runtime-hours", "{args.max_runtime_hours}"
        ]
        print(f"Starting Worker {{i}} on GPU {{i}} (CUDA_VISIBLE_DEVICES={{i}})...")
        p = subprocess.Popen(cmd, env=env)
        processes.append((i, p))
    
    failed = []
    for i, p in processes:
        rc = p.wait()
        if rc != 0:
            print(f"❌ Worker {{i}} failed with exit code {{rc}}")
            failed.append((i, rc))
        else:
            print(f"✓ Worker {{i}} completed successfully.")
    
    if failed:
        raise RuntimeError(f"Multi-GPU execution failed for workers: {{failed}}")
else:
    print("=== Single GPU / CPU execution ===")
    cmd = [
        sys.executable, "src/main.py",
        "{args.config}",
        "--worker-id", "0",
        "--total-workers", "1",
        "--max-runtime-hours", "{args.max_runtime_hours}"
    ]
    print(f"Executing command: {{' '.join(cmd)}}")
    subprocess.check_call(cmd)
'''
    else:
        launch_code = f'''
# Launch Single Worker
cmd = [
    sys.executable, "src/main.py",
    "{args.config}",
    "--worker-id", "{args.worker_id}",
    "--total-workers", "{args.total_workers}",
    "--max-runtime-hours", "{args.max_runtime_hours}"
]

print(f"Executing command: {{' '.join(cmd)}}")
subprocess.check_call(cmd)
'''

    # 2. run_kaggle.py
    run_kaggle_content = f'''#!/usr/bin/env python
import os
import sys
import subprocess

print("=== Starting Kaggle Caption Reconstruction Execution ===")
print(f"Python: {{sys.executable}}")
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
{launch_code}
print("=== Kaggle Execution Completed Successfully ===")
'''

    with open(output_dir / "run_kaggle.py", "w") as f:
        f.write(run_kaggle_content)

    print(f"Successfully prepared Kaggle kernel directory in '{output_dir}'.")
    print("\nTo deploy via Kaggle CLI, run:")
    print(f"  kaggle kernels push -p {output_dir}")

if __name__ == "__main__":
    main()
