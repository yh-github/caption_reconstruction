#!/usr/bin/env python3
"""
Merge valid reconstructions from two Hugging Face runs into a unified target run.
Uses server-side CommitOperationCopy on Hugging Face (instantaneous, no re-downloading).
"""
import argparse
import os
import sys
from huggingface_hub import HfApi, CommitOperationCopy

def merge_runs(
    base_run: str = "wild4_llama_w3_window_v2",
    override_run: str = "wild4_llama_w3_start",
    dest_run: str = "wild4_llama_w3_window_v3",
    repo_id: str = "Y3/dense_video_captions",
    dry_run: bool = False
):
    token = os.environ.get("HF_TOKEN")
    api = HfApi(token=token)

    print(f"Connecting to Hugging Face dataset: {repo_id}...")
    files = api.list_repo_files(repo_id=repo_id, repo_type="dataset")

    base_prefix = f"reconstruction/{base_run}/"
    override_prefix = f"reconstruction/{override_run}/"
    dest_prefix = f"reconstruction/{dest_run}/"

    operations = []

    # 1. From base run: take middle (i=29) and end (i=59) valid files & metadata
    base_files = [f for f in files if f.startswith(base_prefix) and (f.endswith(".json") or f.endswith("metadata.yaml")) and "/skip__" not in f]
    for f in base_files:
        rel = f[len(base_prefix):]
        # Only take non-opening positions (i=29, i=59)
        if "i=0" not in rel and "(3, 0)" not in rel:
            target_path = f"{dest_prefix}{rel}"
            operations.append(CommitOperationCopy(src_path_in_repo=f, path_in_repo=target_path))

    # 2. From override run: take opening (i=0) valid files & metadata
    override_files = [f for f in files if f.startswith(override_prefix) and (f.endswith(".json") or f.endswith("metadata.yaml")) and "/skip__" not in f]
    for f in override_files:
        rel = f[len(override_prefix):]
        target_path = f"{dest_prefix}{rel}"
        operations.append(CommitOperationCopy(src_path_in_repo=f, path_in_repo=target_path))

    print("=" * 65)
    print(f"Base files (middle/end from {base_run}):   {len([op for op in operations if op.src_path_in_repo.startswith(base_prefix)])}")
    print(f"Override files (opening from {override_run}): {len([op for op in operations if op.src_path_in_repo.startswith(override_prefix)])}")
    print(f"Total files to assemble into {dest_run}:   {len(operations)}")
    print("=" * 65)

    if dry_run:
        print("Dry run requested. No changes made.")
        return

    if not operations:
        print("No operations to commit. Ensure override run has completed files.")
        return

    print(f"Committing {len(operations)} files server-side to Hugging Face...")
    api.create_commit(
        repo_id=repo_id,
        repo_type="dataset",
        operations=operations,
        commit_message=f"feat(data): merge {base_run} (middle/end) and {override_run} (opening) into {dest_run}"
    )
    print(f"✓ Successfully assembled unified run at: {dest_prefix}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge valid results across runs on Hugging Face.")
    parser.add_argument("--base", type=str, default="wild4_llama_w3_window_v2", help="Base run name (for middle and end positions)")
    parser.add_argument("--override", type=str, default="wild4_llama_w3_start", help="Override run name (for opening position)")
    parser.add_argument("--dest", type=str, default="wild4_llama_w3_window_v3", help="Destination run name")
    parser.add_argument("--repo", type=str, default="Y3/dense_video_captions", help="Hugging Face repo ID")
    parser.add_argument("--dry-run", action="store_true", help="Only preview what files would be copied")
    args = parser.parse_args()

    merge_runs(
        base_run=args.base,
        override_run=args.override,
        dest_run=args.dest,
        repo_id=args.repo,
        dry_run=args.dry_run
    )
