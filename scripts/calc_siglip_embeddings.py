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
        "--config",
        type=Path,
        default=None,
        help="Optional path to a YAML configuration file.",
    )
    parser.add_argument(
        "--video_dir",
        type=Path,
        default=None,
        help="Path to directory containing raw video files (default: local/wild_videos_raw).",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help="Path to save output .npy embeddings (default: local/wild_videos_embs_siglip).",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default=None,
        help="Model name (default: google/siglip-base-patch16-224).",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Batch size for frame inference (default: 32).",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=None,
        help="Frames per second to extract (default: 1).",
    )
    parser.add_argument(
        "--clip_size",
        type=int,
        default=None,
        help="Seconds per clip to aggregate (default: 1).",
    )
    parser.add_argument(
        "--match_existing",
        type=Path,
        default=None,
        help="Path to existing .npy directory to match (default: local/wild_videos_embs). Set to '' to disable.",
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

    cfg = {}
    if args.config:
        if not args.config.exists():
            logger.error(f"Config file does not exist: {args.config}")
            sys.exit(1)
        with open(args.config, "r") as f:
            cfg = yaml.safe_load(f) or {}

    video_dir = args.video_dir or Path(cfg.get("video_dir", "local/wild_videos_raw"))
    output_dir = args.output_dir or Path(cfg.get("output_dir", "local/wild_videos_embs_siglip"))
    model_name = args.model_name or cfg.get("model_name", "google/siglip-base-patch16-224")
    batch_size = args.batch_size or int(cfg.get("batch_size", 32))
    fps = args.fps or int(cfg.get("fps", 1))
    clip_size = args.clip_size or int(cfg.get("clip_size", 1))

    # Match existing
    if args.match_existing is not None:
        match_existing = args.match_existing if str(args.match_existing).strip() else None
    elif "match_existing" in cfg:
        match_existing = Path(cfg["match_existing"]) if cfg["match_existing"] else None
    else:
        match_existing = Path("local/wild_videos_embs")

    limit = args.limit if args.limit is not None else cfg.get("limit", None)
    if limit is not None:
        limit = int(limit)

    if not video_dir.exists():
        logger.error(f"Video directory does not exist: {video_dir}")
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Save metadata
    metadata_path = output_dir / "metadata.yaml"
    with open(metadata_path, "w") as f:
        yaml.dump(
            {
                "type": "video_embeddings",
                "model": model_name,
                "dim": 768,
                "fps": fps,
                "clip_size": clip_size,
            },
            f,
        )

    # Discover all video files recursively
    all_videos = list(video_dir.rglob("*.mp4"))
    logger.info(f"Discovered {len(all_videos)} video files in {video_dir}")

    # Filter to match existing embeddings if requested
    if match_existing and match_existing.exists():
        target_stems = {p.stem for p in match_existing.glob("*.npy")}
        filtered_videos = [v for v in all_videos if v.stem in target_stems]
        logger.info(
            f"Filtered by {match_existing}: {len(filtered_videos)} videos matched (out of {len(target_stems)} existing .npy files)"
        )
        videos_to_process = filtered_videos
    else:
        videos_to_process = all_videos

    if limit:
        videos_to_process = videos_to_process[:limit]
        logger.info(f"Limiting to first {limit} videos")

    # Filter out already computed outputs
    pending_videos = [v for v in videos_to_process if not (output_dir / f"{v.stem}.npy").exists()]
    logger.info(
        f"Found {len(videos_to_process)} total targets: {len(videos_to_process) - len(pending_videos)} already exist, {len(pending_videos)} pending."
    )

    if not pending_videos:
        logger.info("All target embeddings already exist. Nothing to do!")
        return

    logger.info(f"Initializing VideoEmbedder with {model_name} (batch_size={batch_size})...")
    embedder = VideoEmbedder(model_name=model_name, batch_size=batch_size)

    for video_path in tqdm(pending_videos, desc="Processing videos"):
        output_filepath = output_dir / f"{video_path.stem}.npy"
        timestamped_frames = embedder._extract_timestamped_frames(video_path, fps=fps)
        if not timestamped_frames:
            logger.warning(f"Skipping {video_path.name}: no frames extracted")
            continue

        timestamps, frames = zip(*timestamped_frames)
        frame_embeddings = embedder._get_frame_embeddings(list(frames))
        if not frame_embeddings:
            logger.warning(f"Skipping {video_path.name}: no embeddings generated")
            continue

        if fps == 1 and clip_size == 1:
            clip_embeddings = np.array(frame_embeddings)
        else:
            clip_embeddings = embedder._group_and_average_embeddings(list(timestamps), frame_embeddings, clip_size)

        np.save(output_filepath, clip_embeddings)

    logger.info("Embedding extraction completed successfully!")


if __name__ == "__main__":
    main()
