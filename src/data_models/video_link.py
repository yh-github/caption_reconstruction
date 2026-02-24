import logging
from typing import Self
from pydantic import BaseModel, ConfigDict, field_validator


class VideoLinkData(BaseModel):
    model_config = ConfigDict(frozen=True)
    video_id:str
    uri:str
    start_offset:float
    end_offset:float

    # noinspection PyMethodParameters
    @field_validator('start_offset', 'end_offset')
    def round_timestamp(cls, value):
        return round(value, 3)

    def duration(self) -> float:
        return self.end_offset - self.start_offset

    def limit_duration(self, limit:float|None) -> Self:
        if limit and self.duration()<=limit:
            return self
        new_end_offset = round(self.start_offset + limit, 3)
        while new_end_offset-self.start_offset < limit:
            with_epsilon = round(new_end_offset + 0.001, 3)
            logging.warning(f'adjusting borderline limit {new_end_offset} ~~> {with_epsilon}')
            new_end_offset = with_epsilon
        return self.model_copy(update={'end_offset':new_end_offset})

    def optional_mask(self, start_percentage: float|None, end_percentage: float|None) -> list[Self]:
        """Returns a new VideoLinkData if the mask is valid, otherwise returns list with only self"""
        if start_percentage is None and end_percentage is None:
            return [self]
        return self.mask(start_percentage or 0.0, end_percentage or 1.0)

    def mask(self, start_percentage: float, end_percentage: float) -> list[Self]:
        """Splits into two new VideoLinkData:
        1. from start_offset to start_offset+duration*start_percentage
        2. from start_offset+duration*end_percentage to end_offset

        edge cases (such as start_percentage=0) may return only a single object or raise exceptions (require end-start>=1)
        """
        if not (0 <= start_percentage <= 1) or not (0 <= end_percentage <= 1):
            raise ValueError("Percentages must be between 0 and 1")
        if end_percentage <= start_percentage:
            raise ValueError("end_percentage must be greater than start_percentage")

        duration = self.duration()
        result = []

        if start_percentage > 0:
            result.append(VideoLinkData(
                video_id=self.video_id,
                uri=self.uri,
                start_offset=self.start_offset,
                end_offset=self.start_offset + duration * start_percentage
            ))

        if end_percentage < 1:
            result.append(VideoLinkData(
                video_id=self.video_id,
                uri=self.uri,
                start_offset=self.start_offset + duration * end_percentage,
                end_offset=self.end_offset
            ))

        return result

from pathlib import Path

class VideoLocalData(BaseModel):
    model_config = ConfigDict(frozen=True)
    video_id: str
    path: Path
    clip_duration: float

    def duration(self) -> float:
        return self.clip_duration

    def limit_duration(self, limit: float | None) -> Self:
        if limit is None or self.clip_duration <= limit:
            return self
        return self.model_copy(update={'clip_duration': limit})
