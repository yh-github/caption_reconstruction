import os
import json
import logging
from abc import ABC, abstractmethod
from data_models import CaptionedClip, CaptionedVideo, TimestampRange


def _parse_storytelling_timestamp(ts_str: str) -> float:
    """Helper to parse MM:SS format into seconds."""
    parts = ts_str.split(':')
    minutes = int(parts[0])
    seconds = int(parts[1])
    # TODO check if has hours
    return float(minutes * 60 + seconds)


class BaseDataLoader(ABC):
    """Abstract base class for all data loaders."""
    @abstractmethod
    def load(self, limit:int|None=None) -> list[CaptionedVideo]:
        """Loads data and returns a list of CaptionedVideo objects."""
        pass

    def load_all_sentences(self) -> list[str]:
        return [c.caption for x in self.load(limit=10*1000*1000) for c in x.clips]

    def find(self, video_id):
        return next((v for v in self.load() if v.video_id == video_id), None) # TODO load_iter


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
        for i,video_data in enumerate(data):
            clips = [
                CaptionedClip(
                    index=clip_ind,
                    timestamp=TimestampRange(start=clip_data["timestamp"]-1, duration=1),
                    caption=clip_data["description"]
                ) for clip_ind, clip_data in enumerate(video_data["clips"])
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

    def find(self, video_id):
        return self.load_file(video_id+".txt")

    def load(self, limit:int|None=None) -> list[CaptionedVideo]:
        logging.info(f"Loading from Video Storytelling dataset at: {self.data_path} {self.limit=}")
        filenames = sorted([f for f in os.listdir(self.data_path) if f.endswith(".txt")])
        if _limit := (limit or self.limit):
            filenames = filenames[:_limit]

        return [self.load_file(filename) for filename in filenames]

    def load_file(self, filename:str) -> CaptionedVideo:
        file_path = os.path.join(self.data_path, filename)
        video_id = filename.replace('.txt', '')
        clips = []
        with open(file_path, 'r') as f:
            lines = f.readlines()[1:]  # Skip video ID line
            for line in lines:
                parts = line.strip().split()
                if len(parts) < 3: continue
                start_time = _parse_storytelling_timestamp(parts[0])
                end_time = _parse_storytelling_timestamp(parts[1])
                description = " ".join(parts[2:])
                clips.append(CaptionedClip(
                    index=len(clips),
                    timestamp=TimestampRange(start=start_time, duration=end_time-start_time),
                    caption=description
                ))
            return CaptionedVideo(video_id=video_id, clips=clips)


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

        if _limit:= (limit or self.limit):
            data = data[:_limit]

        for video_info in data:
            video_id = video_info["videoID"]
            captions = video_info["enCap"][:5]
            clips = []
            for i, caption in enumerate(captions):
                clips.append(CaptionedClip(
                    index=i,
                    timestamp=TimestampRange(start=i*2, duration=2),
                    caption=caption
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

