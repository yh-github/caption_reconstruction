# src/data_loaders.py
import os
import json
import logging
from abc import ABC, abstractmethod
from data_models import TranscriptClip, NarrativeOnlyPayload

class BaseDataLoader(ABC):
    """
    An abstract base class for data loaders.
    Defines a common interface for all dataset-specific loaders.
    """
    @abstractmethod
    def load(self) -> list[TranscriptClip]:
        """Loads data from a source and returns a list of TranscriptClip objects."""
        pass


def _parse_storytelling_timestamp(ts_str: str) -> float:
    """Helper to parse MM:SS format into seconds."""
    parts = ts_str.split(':')
    minutes = int(parts[0])
    seconds = int(parts[1])
    return float(minutes * 60 + seconds)

class VideoStorytellingLoader(BaseDataLoader):
    """Loads data from the Video Storytelling dataset format (many TXT files)."""
    def __init__(self, data_path: str):
        self.data_path = data_path

    def load(self) -> list[TranscriptClip]:
        logging.info(f"Loading from Video Storytelling dataset at: {self.data_path}")
        all_clips = []
        for filename in os.listdir(self.data_path):
            if filename.endswith(".txt"):
                file_path = os.path.join(self.data_path, filename)
                with open(file_path, 'r') as f:
                    lines = f.readlines()
                    # First line is video ID, rest are captions
                    for line in lines[1:]:
                        parts = line.strip().split()
                        if len(parts) < 3:
                            continue

                        end_time_str = parts[1]
                        description = " ".join(parts[2:])

                        clip = TranscriptClip(
                            timestamp=_parse_storytelling_timestamp(end_time_str),
                            data=NarrativeOnlyPayload(description=description)
                        )
                        all_clips.append(clip)
        return all_clips

class VatexLoader(BaseDataLoader):
    """Loads data from the VATEX dataset format (a single large JSON file)."""
    def __init__(self, data_path: str):
        self.data_path = data_path

    def load(self) -> list[TranscriptClip]:
        logging.info(f"Loading from VATEX dataset at: {self.data_path}")
        all_clips = []
        with open(self.data_path, 'r') as f:
            data = json.load(f)
        
        for video_info in data:
            video_id = video_info["videoID"]
            # We only use the first 5 captions, as specified.
            captions = video_info["enCap"][:5]
            
            for i, caption in enumerate(captions):
                # Since there are no timestamps, we generate placeholder ones.
                clip = TranscriptClip(
                    timestamp=float(i + 1),
                    data=NarrativeOnlyPayload(description=caption)
                )
                all_clips.append(clip)
        return all_clips

def get_data_loader(config: dict) -> BaseDataLoader:
    """
    Factory function that reads the config and returns the appropriate
    data loader instance.
    """
    data_config = config.get("data", {})
    dataset_name = data_config.get("name")
    data_path = data_config.get("path")

    if not dataset_name or not data_path:
        raise ValueError("Dataset 'name' and 'path' must be specified in the config.")

    if dataset_name == "vatex":
        return VatexLoader(data_path)
    elif dataset_name == "video_storytelling":
        return VideoStorytellingLoader(data_path)
    else:
        raise NotImplementedError(f"No data loader found for dataset: '{dataset_name}'")
