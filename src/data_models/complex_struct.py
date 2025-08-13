from pydantic import BaseModel, Field, RootModel


class KeyMoment(BaseModel):
    """Represents a significant visual event within a segment."""
    start: str = Field(..., description="Start timestamp in HH:MM:SS.mmm format")
    end: str = Field(..., description="End timestamp in HH:MM:SS.mmm format")
    caption: str = Field(..., description="Concise description of the key moment")


class VideoSegment(BaseModel):
    """Represents a distinct video segment with detailed analysis."""
    start: str = Field(..., description="Segment start timestamp in HH:MM:SS.mmm format")
    end: str = Field(..., description="Segment end timestamp in HH:MM:SS.mmm format")
    entities: list[str] = Field(default_factory=list, description="People, animals, or robots")
    objects: list[str] = Field(default_factory=list, description="Solid items like furniture, vehicles, tools")
    stuff: list[str] = Field(default_factory=list, description="Materials/substances like water, sand, smoke")
    primary_activity: str = Field(..., description="Main action or activity in the segment")
    segment_summary: str = Field(..., description="Concise description of overall segment content")
    key_moments: list[KeyMoment] = Field(..., description="2-5 notable visual events within the segment")


class VideoAnalysis(BaseModel):
    video_id: str = Field(..., description="Unique identifier for the video")
    segments: list[VideoSegment]

    @staticmethod
    def _timestamp_to_seconds(timestamp: str) -> float:
        """Convert HH:MM:SS.mmm to seconds for comparison"""
        parts = timestamp.split(':')
        hours = int(parts[0])
        minutes = int(parts[1])
        seconds_parts = parts[2].split('.')
        seconds = int(seconds_parts[0])
        milliseconds = int(seconds_parts[1])

        return hours * 3600 + minutes * 60 + seconds + milliseconds / 1000
