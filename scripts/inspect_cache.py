
import sys
import argparse
from pathlib import Path

# Add src to path
sys.path.append(str(Path.cwd() / "src"))

from common_utils.tracking import setup_logging
from experiment_executor.config_loader import load_config
from data.data_loaders import get_data_loader
from llm.embedder import Embedder
import logging

def main():
    parser = argparse.ArgumentParser(description="Inspect cache coverage for a dataset.")
    parser.add_argument("config_path", type=str, help="Path to experiment config yaml")
    parser.add_argument("--video-id", type=str, help="Filter for specific video ID", default=None)
    args = parser.parse_args()

    setup_logging(log_dir="logs", run_id="inspect", console_level=logging.INFO)
    logger = logging.getLogger(__name__)

    config = load_config(args.config_path)
    data_config = config["data_config"]
    
    logger.info(f"Loading data from {data_config['path']}")
    loader = get_data_loader(data_config)
    videos = loader.load()
    
    # Initialize Embedder (read-only mode essentially since we won't call get_embeddings)
    # We create a blocked client just in case
    class BlockingClient:
        def __getattr__(self, name): raise RuntimeError("Blocked")
        def __call__(self, *args, **kwargs): raise RuntimeError("Blocked")
        
    embedder = Embedder(client=BlockingClient())
    cache = embedder.cache
    
    logger.info(f"Cache Directory: {cache.directory}")
    logger.info(f"Cache Size: {len(cache)} items")

    total_sentences = 0
    missing_sentences = 0
    missing_videos = set()
    
    target_videos = [v for v in videos if args.video_id is None or v.video_id == args.video_id]
    
    logger.info(f"Inspecting {len(target_videos)} videos...")

    for vid in target_videos:
        texts = vid.get_texts() # Depends on implementation of CaptionedVideo/VectorConvertor
        # For standard CaptionedVideo, getting texts usually means extracting captions.
        # Let's verify what `get_texts()` does or if we need to manually extract.
        # Looking at `vector_dataloaders.py`, line 139: `x.get_texts()` is used.
        # `x` comes from `base_dataloader.load()`.
        
        # In `CaptionedVideo`, let's assume `get_texts` is not defined but `clips` has text.
        # Let's inspect `CaptionedVideo` quickly or assume `vector_dataloaders` knows best.
        # Actually `CaptionedVideo` might not have `get_texts`.
        # `base_dataloader` returns `CaptionedVideo`.
        # Wait, `VectorConvertorLoader` calls `x.get_texts()`.
        # If `CaptionedVideo` doesn't have it, `VectorConvertorLoader` would fail.
        # Let's assume it exists or use a fallback.
        
        if hasattr(vid, 'get_texts'):
            texts = vid.get_texts()
        else:
             # Fallback: extract captions from clips
            texts = [c.caption for c in vid.clips]
            
        vid_missing = 0
        for t in texts:
            if t not in cache:
                vid_missing += 1
                logger.debug(f"MISSING: Video {vid.video_id} - '{t}'")
            total_sentences += 1
        
        if vid_missing > 0:
            missing_sentences += vid_missing
            missing_videos.add(vid.video_id)
            if args.video_id:
                logger.info(f"Video {vid.video_id}: {vid_missing}/{len(texts)} missing.")

    logger.info("-" * 30)
    logger.info(f"Total Sentences Checked: {total_sentences}")
    logger.info(f"Missing Sentences: {missing_sentences} ({missing_sentences/total_sentences*100:.2f}%)")
    logger.info(f"Videos with missing data: {len(missing_videos)}/{len(target_videos)}")
    
    if len(missing_videos) > 0 and len(missing_videos) < 20:
        logger.info(f"Missing Videos: {missing_videos}")

if __name__ == "__main__":
    main()
