import json
import logging
from pathlib import Path
from typing import Self

from pydantic import BaseModel, RootModel, Field, model_validator

from data_models.captions_only import VideoLinkData

logger = logging.getLogger(__name__)

def time_str_to_seconds(t: str) -> float:
    """
    Convert time string to seconds.

    Supports formats like:
    - "00:15:31.297" (HH:MM:SS.mmm)
    - "15:31.297" (MM:SS.mmm)
    - "3:42" (MM:SS)
    - "42" (SS)
    - "42.5" (SS.mmm)

    Args:
        t: Time string in various formats

    Returns:
        float: Time in seconds

    Raises:
        ValueError: If the time string format is not recognized
    """
    if not t or not isinstance(t, str):
        raise ValueError(f"Invalid time string: {t}")

    # Remove any whitespace
    t = t.strip()

    # Handle empty string
    if not t:
        raise ValueError("Empty time string")

    # Split by colon to determine format
    parts = t.split(':')

    try:
        if len(parts) == 1:
            # Format: "42" or "42.5" (seconds only)
            return float(parts[0])

        elif len(parts) == 2:
            # Format: "3:42" or "3:42.5" (MM:SS or MM:SS.mmm)
            minutes = int(parts[0])
            seconds = float(parts[1])
            return minutes * 60 + seconds

        elif len(parts) == 3:
            # Format: "00:15:31.297" (HH:MM:SS.mmm)
            hours = int(parts[0])
            minutes = int(parts[1])
            seconds = float(parts[2])
            return hours * 3600 + minutes * 60 + seconds

        else:
            raise ValueError(f"Too many colon-separated parts: {len(parts)}")

    except ValueError as e:
        if "could not convert" in str(e).lower() or "invalid literal" in str(e).lower():
            raise ValueError(f"Invalid time format: '{t}' - contains non-numeric parts")
        raise ValueError(f"Invalid time format: '{t}' - {str(e)}")


class TimeInFloats(BaseModel):
    start: float
    end: float

class TimeInFrames(BaseModel):
    """Example:
    {
        "start": "00:15:31.297",
        "end": "00:16:20.279"
    }
    """
    start: str
    end: str

    def to_floats(self) -> TimeInFloats:
        return TimeInFloats(
            start=time_str_to_seconds(self.start),
            end=time_str_to_seconds(self.end)
        )


class TimeInSeconds(BaseModel):
    """Example:
    {
        "start": "931.2970333333334",
        "end": "980.2793"
    }
    """
    start: str
    end: str

    def to_floats(self) -> TimeInFloats:
        return TimeInFloats(
            start=float(self.start),
            end=float(self.end)
        )


class TimeInOriginalVideo(BaseModel):
    split_method: str = Field(alias="split-method")
    frames: TimeInFrames | None = None
    seconds: TimeInSeconds | None = None

    @model_validator(mode='after')
    def validate_and_transform(self) -> Self:
        if self.seconds is not None:
            if ':' in self.seconds.start or ':' in self.seconds.end:
                logger.debug('__post_init__ BEFORE >>>>', self.model_dump())
                assert ':' in self.seconds.start and ':' in self.seconds.end
                assert self.frames is None
                assert self.split_method == 'manual'
                self.frames = TimeInFrames(
                    start=self.seconds.start,
                    end=self.seconds.end
                )
                self.seconds = None
                logger.debug('__post_init__ AFTER >>>>', self.model_dump())
        return self


    def to_floats(self) -> TimeInFloats:
        if self.seconds is not None:
            return self.seconds.to_floats()
        else:
            return self.frames.to_floats()

    



class Evidence(RootModel[dict[str, list[float]]]):
    pass


class WildVideoMetadata(BaseModel):
    question: str
    answer: str
    assignment_id: str
    objective: str
    confidence: str
    original_question: str
    original_answer: str
    question_type: list[str]
    question_base: list[str]
    evidences: list[Evidence]
    domain: str
    evidences_in_min: list[dict[str, list[str]]]
    video_link: str
    video_id: str
    time_in_original_video: TimeInOriginalVideo
    duration: float
    alter_answers: list[str]
    alter_evidences: list[list[list[float]]]
    url_for_original_video: str

    def to_link(self) -> VideoLinkData:
        try:
            def uri(u):
                return u.replace('//watch', '/watch')

            if not self.time_in_original_video:
                raise Exception(f"Missing time_in_original_video in {self.video_id}")
            t = self.time_in_original_video.to_floats()
            return VideoLinkData(
                video_id=self.video_id,
                uri=uri(self.url_for_original_video),
                start_offset=t.start,
                end_offset=t.end
            )
        except Exception as e:
            logger.error(f"Error with video {self.video_id=}", e)
            raise



def load_wild_dataset(path:Path, limit:int=None):
    with open(path) as f:
        j=json.load(f)
    limit=limit or len(j)
    for v in j[:limit]:
        try:
            yield WildVideoMetadata.model_validate(v)
        except Exception as e:
            raise Exception('Error with video', v['video_id'], e) from e
