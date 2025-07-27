# src/data_loaders.py
import os
import json
import logging
from abc import ABC, abstractmethod
from data_models import CaptionedClip, CaptionedVideo, NarrativeOnlyPayload

def _parse_storytelling_timestamp(ts_str: str) -> float:
    """Helper to parse MM:SS format into seconds."""
    parts = ts_str.split(':')
    minutes = int(parts[0])
    seconds = int(parts[1])
    return float(minutes * 60 + seconds)


class BaseDataLoader(ABC):
    """Abstract base class for all data loaders."""
    @abstractmethod
    def load(self, limit:int|None=None) -> list[CaptionedVideo]:
        """Loads data and returns a list of CaptionedVideo objects."""
        pass

    def load_all_sentences(self) -> list[str]:
        return [c.data.description for x in self.load(limit=10*1000*1000) for c in x.clips]

class ToyDataLoader(BaseDataLoader):
    """
    This serves as our initial ground-truth data for building and debugging
    the experimental pipeline.
    """
    def __init__(self, data_path: str):
        self.data_path = data_path

    def load(self, limit:int|None=None) -> list[CaptionedVideo]:
        all_videos = []
        with open(self.data_path, 'r') as f:
            data = json.load(f)

        if limit:
            data = data[:limit]
        for video_data in data:
            clips = [
                CaptionedClip(
                    timestamp=clip_data["timestamp"],
                    data=NarrativeOnlyPayload(description=clip_data["description"])
                ) for clip_data in video_data["clips"]
            ]
            all_videos.append(
                CaptionedVideo(video_id=video_data["video_id"], clips=clips)
            )
        return all_videos

class VideoStorytellingLoader(BaseDataLoader):
    """Loads data from the Video Storytelling dataset format."""
    def __init__(self, data_path: str, limit=None):
        self.data_path = data_path
        self.limit = limit

    def load(self, limit:int|None=None) -> list[CaptionedVideo]:
        logging.info(f"Loading from Video Storytelling dataset at: {self.data_path} {self.limit=}")
        all_videos = []
        filenames = sorted([f for f in os.listdir(self.data_path) if f.endswith(".txt")])
        if _limit := limit or self.limit:
            filenames = filenames[:_limit]

        for filename in filenames:
            video_id = filename.replace('.txt', '')
            file_path = os.path.join(self.data_path, filename)
            clips = []
            with open(file_path, 'r') as f:
                lines = f.readlines()[1:] # Skip video ID line
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) < 3: continue
                    end_time_str = parts[1]
                    description = " ".join(parts[2:])
                    clips.append(CaptionedClip(
                        timestamp=_parse_storytelling_timestamp(end_time_str),
                        data=NarrativeOnlyPayload(description=description)
                    ))
            all_videos.append(CaptionedVideo(video_id=video_id, clips=clips))
        return all_videos

class VatexLoader(BaseDataLoader):
    """Loads data from the VATEX dataset format."""
    def __init__(self, data_path: str, limit: int|None=None):
        self.data_path = data_path
        self.limit = limit

    def load(self, limit:int|None=None) -> list[CaptionedVideo]:
        logging.info(f"Loading from VATEX dataset at: {self.data_path} {self.limit=}")
        all_videos = []
        with open(self.data_path, 'r') as f:
            data = json.load(f)

        if _limit:= limit or self.limit:
            data = data[:_limit]

        for video_info in data:
            video_id = video_info["videoID"]
            captions = video_info["enCap"][:5]
            clips = []
            for i, caption in enumerate(captions):
                clips.append(CaptionedClip(
                    timestamp=float(i + 1),
                    data=NarrativeOnlyPayload(description=caption)
                ))
            all_videos.append(CaptionedVideo(video_id=video_id, clips=clips))
        return all_videos

def get_data_loader(data_config: dict) -> BaseDataLoader:
    """
    Factory function that reads the config and returns the appropriate
    data loader instance.
    """
    dataset_name = data_config.get("name")
    data_path = data_config.get("path")
    limit = data_config.get("limit")

    if not dataset_name or not data_path:
        raise ValueError("Dataset 'name' and 'path' must be specified in the config.")

    if dataset_name == "vatex":
        return VatexLoader(data_path, limit)
    elif dataset_name == "video_storytelling":
        return VideoStorytellingLoader(data_path, limit)
    elif dataset_name == "toy_data":
        return ToyDataLoader(data_path)
    else:
        raise NotImplementedError(f"No data loader found for dataset: '{dataset_name}'")

