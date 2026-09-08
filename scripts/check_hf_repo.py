#!/usr/bin/env python
import argparse
from collections import defaultdict
from huggingface_hub import HfApi

def main():
    parser = argparse.ArgumentParser(description="Audit and list files in Hugging Face dataset repo.")
    parser.add_argument("--repo-id", type=str, default="Y3/dense_video_captions", help="HF repository ID")
    parser.add_argument("--filter", type=str, default=None, help="Filter string in path (e.g. 'v2' or 'phi-3_v2')")
    parser.add_argument("--show", type=int, default=20, help="Number of sample files to show")
    args = parser.parse_args()

    api = HfApi()
    print(f"Connecting to Hugging Face dataset repo: {args.repo_id}...")
    
    try:
        files = api.list_repo_files(repo_id=args.repo_id, repo_type="dataset")
    except Exception as e:
        print(f"Error fetching repo files: {e}")
        return

    print(f"Total files in repository: {len(files)}")

    # Group files by experiment category
    experiment_counts = defaultdict(int)
    for f in files:
        if f.startswith("reconstruction/"):
            parts = f.split("/")
            if len(parts) >= 2:
                experiment_counts[parts[1]] += 1

    print("\n--- Summary of Reconstruction Folders ---")
    for exp_name, cnt in sorted(experiment_counts.items()):
        print(f"  - {exp_name:<35} : {cnt} files")

    if args.filter:
        matched = [f for f in files if args.filter.lower() in f.lower()]
        print(f"\n--- Files matching filter '{args.filter}' ({len(matched)} files) ---")
        for f in matched[:args.show]:
            print(f"  {f}")
        if len(matched) > args.show:
            print(f"  ... and {len(matched) - args.show} more.")
    else:
        # Also check for any 'v2' files
        v2_files = [f for f in files if "v2" in f.lower() or "phi-3_v2" in f.lower()]
        if v2_files:
            print(f"\n--- Recently Added v2 Files ({len(v2_files)} files) ---")
            for f in v2_files[:args.show]:
                print(f"  {f}")
            if len(v2_files) > args.show:
                print(f"  ... and {len(v2_files) - args.show} more.")

if __name__ == "__main__":
    main()
