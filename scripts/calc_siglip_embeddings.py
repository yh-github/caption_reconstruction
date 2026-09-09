import argparse
import logging
import sys
from pathlib import Path
import numpy as np
import yaml
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent.parent))
from src.data.video_embeddings import VideoEmbedder

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Extract SigLIP video embeddings.")
    parser.add_argument(
        "--video_dir",
        type=Path,
        default=Path("local/wild_videos_raw"),
        help="Path to directory containing raw video files.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("local/wild_videos_embs_siglip"),
        help="Path to save output .npy embeddings.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="google/siglip-base-patch16-224",
        help="Model name (HuggingFace or timm).",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for frame inference.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=1,
        help="Frames per second to extract.",
    )
    parser.add_argument(
        "--clip_size",
        type=int,
        default=1,
        help="Seconds per clip to aggregate.",
    )
    parser.add_argument(
        "--match_existing",
        type=Path,
        default=Path("local/wild_videos_embs"),
        help="Path to existing .npy directory. If set, only processes videos matching existing .npy stems.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit on number of videos to process.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if not args.video_dir.exists():
        logger.error(f"Video directory does not exist: {args.video_dir}")
        sys.exit(1)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Save metadata
    metadata_path = args.output_dir / "metadata.yaml"
    with open(metadata_path, "w") as f:
        yaml.dump(
            {
                "type": "video_embeddings",
                "model": args.model_name,
                "dim": 768,
                "fps": args.fps,
                "clip_size": args.clip_size,
            },
            f,
        )

    # Discover all video files recursively
    all_videos = list(args.video_dir.rglob("*.mp4"))
    logger.info(f"Discovered {len(all_videos)} video files in {args.video_dir}")

    # Filter to match existing embeddings if requested
    if args.match_existing and args.match_existing.exists():
        target_stems = {p.stem for p in args.match_existing.glob("*.npy")}
        filtered_videos = [v for v in all_videos if v.stem in target_stems]
        logger.info(
            f"Filtered by {args.match_existing}: {len(filtered_videos)} videos matched (out of {len(target_stems)} existing .npy files)"
        )
        videos_to_process = filtered_videos
    else:
        videos_to_process = all_videos

    if args.limit:
        videos_to_process = videos_to_process[: args.limit]
        logger.info(f"Limiting to first {args.limit} videos")

    # Filter out already computed outputs
    pending_videos = [v for v in videos_to_process if not (args.output_dir / f"{v.stem}.npy").exists()]
    logger.info(
        f"Found {len(videos_to_process)} total targets: {len(videos_to_process) - len(pending_videos)} already exist, {len(pending_videos)} pending."
    )

    if not pending_videos:
        logger.info("All target embeddings already exist. Nothing to do!")
        return

    logger.info(f"Initializing VideoEmbedder with {args.model_name} (batch_size={args.batch_size})...")
    embedder = VideoEmbedder(model_name=args.model_name, batch_size=args.batch_size)

    for video_path in tqdm(pending_videos, desc="Processing videos"):
        output_filepath = args.output_dir / f"{video_path.stem}.npy"
        timestamped_frames = embedder._extract_timestamped_frames(video_path, fps=args.fps)
        if not timestamped_frames:
            logger.warning(f"Skipping {video_path.name}: no frames extracted")
            continue

        timestamps, frames = zip(*timestamped_frames)
        frame_embeddings = embedder._get_frame_embeddings(list(frames))
        if not frame_embeddings:
            logger.warning(f"Skipping {video_path.name}: no embeddings generated")
            continue

        if args.fps == 1 and args.clip_size == 1:
            clip_embeddings = np.array(frame_embeddings)
        else:
            clip_embeddings = embedder._group_and_average_embeddings(list(timestamps), frame_embeddings, args.clip_size)

        np.save(output_filepath, clip_embeddings)

    logger.info("Embedding extraction completed successfully!")


if __name__ == "__main__":
    main()
