import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from src.data.video_embeddings import VideoEmbedder
import yaml

def main():
    # Directories
    # Update VIDEO_DIR to the actual path where .mp4 files are stored.
    VIDEO_DIR = Path("datasets/wildQA/videos") 
    OUTPUT_DIR = Path("local/wild_videos_embs_siglip")
    
    if not VIDEO_DIR.exists():
        print(f"Warning: {VIDEO_DIR} does not exist. Please update the path.")
        
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Save metadata
    with open(OUTPUT_DIR / "metadata.yaml", "w") as f:
        yaml.dump({"type": "video_embeddings", "model": "google/siglip-base-patch16-224", "dim": 768}, f)
        
    print(f"Initializing SigLIP Video Embedder...")
    embedder = VideoEmbedder(model_name="google/siglip-base-patch16-224")
    
    print(f"Processing videos from {VIDEO_DIR} to {OUTPUT_DIR}...")
    embedder.process_directory(video_dir=VIDEO_DIR, output_dir=OUTPUT_DIR, fps=1, clip_size=1)
    
    print("Done!")

if __name__ == "__main__":
    main()
