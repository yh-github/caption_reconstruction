#!/usr/bin/env python3
import os
import sys
import argparse
from datetime import datetime
from huggingface_hub import HfApi

def check_progress(run_name: str = "wild4_llama_w3_window_v2", repo_id: str = "Y3/dense_video_captions", show_videos: bool = False):
    token = os.environ.get("HF_TOKEN")
    api = HfApi(token=token)

    try:
        files = api.list_repo_files(repo_id=repo_id, repo_type="dataset")
    except Exception as e:
        print(f"❌ Error querying Hugging Face repo {repo_id}: {e}")
        return

    # Filter files for the given run
    run_prefix = f"reconstruction/{run_name}/"
    run_files = [f for f in files if f.startswith(run_prefix) and f.endswith(".json")]

    # Find unique positions
    positions = sorted(list(set(
        f.split("/")[2] for f in run_files if len(f.split("/")) >= 3
    )))

    total_files = len(run_files)
    total_valid = sum(1 for f in run_files if "/skip__" not in f)
    total_skips = sum(1 for f in run_files if "/skip__" in f)

    print("=" * 65)
    print(f"📊 RUN PROGRESS: {run_name}")
    print(f"📁 Hugging Face Dataset: https://huggingface.co/datasets/{repo_id}")
    print("=" * 65)
    print(f"Overall Processed: {total_files} / 300 ({total_files/300*100:.1f}%)")
    print(f"  ✓ Valid Reconstructions: {total_valid} ({total_valid/max(1, total_files)*100:.1f}%)")
    print(f"  ⚠ Skips:                 {total_skips} ({total_skips/max(1, total_files)*100:.1f}%)")
    print("-" * 65)

    for pos in positions:
        pos_files = [f for f in run_files if f"/{pos}/" in f]
        valid_files = [f for f in pos_files if "/skip__" not in f]
        skip_files = [f for f in pos_files if "/skip__" in f]
        count = len(pos_files)
        pct = (count / 100.0) * 100

        bar_len = 20
        filled = int(bar_len * (count / 100.0))
        bar = "█" * filled + "░" * (bar_len - filled)

        # Simplify position label
        label = pos
        if "fixed_fill(" in label:
            label = label.split("__")[-1]

        print(f"[{bar}] {label:<22} : {count:3d}/100 ({pct:5.1f}%) | Valid: {len(valid_files):2d} | Skips: {len(skip_files):2d}")
        
        if show_videos and pos_files:
            latest = [os.path.basename(f) for f in pos_files[-3:]]
            print(f"    Latest: {', '.join(latest)}")

    print("-" * 65)

    # Check latest commits
    try:
        commits = list(api.list_repo_commits(repo_id=repo_id, repo_type="dataset"))[:3]
        print("🕒 Recent Syncs:")
        for c in commits:
            dt = c.created_at.strftime("%Y-%m-%d %H:%M:%S UTC")
            print(f"  • {dt}: {c.title}")
    except Exception:
        pass
    print("=" * 65)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check progress of reconstruction run on Hugging Face.")
    parser.add_argument("--run", type=str, default="wild4_llama_w3_window_v2", help="Run directory name under reconstruction/")
    parser.add_argument("--repo", type=str, default="Y3/dense_video_captions", help="Hugging Face repo ID")
    parser.add_argument("--videos", action="store_true", help="Show latest processed video names")
    args = parser.parse_args()

    check_progress(run_name=args.run, repo_id=args.repo, show_videos=args.videos)
