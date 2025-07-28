import pandas as pd
import numpy as np
from config_loader import load_config
from data_loaders import get_data_loader

# --- 1. Configuration ---
config = {
    "data_config": {
        "name": "video_storytelling",
        "path": "./datasets/storytelling",
        "limit": 200
    }
}

# --- Helper Functions for Formatting ---

def seconds_to_mmss(seconds):
    """Converts a float number of seconds into an MM:SS string format."""
    if pd.isna(seconds):
        return "N/A"
    minutes = int(seconds // 60)
    remaining_seconds = int(seconds % 60)
    return f"{minutes:02d}:{remaining_seconds:02d}"

def custom_float_formatter(x):
    """
    Formats a float to have at most one decimal place, and no decimal
    places if it's a whole number.
    """
    if pd.isna(x):
        return "N/A"
    if x == int(x):
        return f"{int(x)}"
    else:
        return f"{x:.1f}"

# --- 2. Data Loading ---
print("Loading data...")
data_loader = get_data_loader(config["data_config"])
all_videos = data_loader.load()
print(f"Loaded {len(all_videos)} videos.")

# --- 3. Convert to DataFrame ---
records = []
for video in all_videos:
    for clip in video.clips:
        records.append({
            "video_id": video.video_id,
            "timestamp": clip.timestamp,
            "description": clip.data.description,
            "word_count": len(clip.data.description.split())
        })

df = pd.DataFrame(records)

# --- 4. Calculate and Display Statistics ---
if not df.empty:
    print("\n--- Dataset Statistics ---")
    total_videos = df['video_id'].nunique()
    total_captions = len(df)
    print(f"Total Unique Videos: {total_videos}")
    print(f"Total Captions (Clips): {total_captions}")

    # Use .map() instead of the deprecated .applymap()
    captions_per_video_stats = df.groupby('video_id').size().describe().to_frame().map(custom_float_formatter)
    print("\n--- Captions per Video ---")
    print(captions_per_video_stats)

    word_count_stats = df['word_count'].describe().to_frame().map(custom_float_formatter)
    print("\n--- Caption Length (in words) ---")
    print(word_count_stats)

    video_durations_seconds = df.groupby('video_id')['timestamp'].max()
    duration_stats_seconds = video_durations_seconds.describe().to_frame().map(custom_float_formatter)
    print("\n--- Video Duration (in seconds, based on last timestamp) ---")
    print(duration_stats_seconds)
    
    video_durations_mmss = video_durations_seconds.apply(seconds_to_mmss)
    print("\n--- Video Duration (MM:SS, for readability) ---")
    print(video_durations_mmss.describe())

else:
    print("\nNo data was loaded to analyze.")

df.head()
