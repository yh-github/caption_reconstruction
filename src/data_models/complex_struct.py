from pydantic import BaseModel, Field, validator
import re


class KeyMoment(BaseModel):
    """Represents a significant visual event within a segment."""
    start: str = Field(..., description="Start timestamp in HH:MM:SS.mmm format")
    end: str = Field(..., description="End timestamp in HH:MM:SS.mmm format")
    caption: str = Field(..., description="Concise description of the key moment")


class Segment(BaseModel):
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
    """Complete video analysis with segment breakdown."""
    segments: list[Segment] = Field(..., description="List of video segments with detailed analysis")

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


# Example usage and testing
if __name__ == "__main__":
    # Example valid data
    sample_data = {
        "segments": [
            {
                "start": "00:15:58.000",
                "end": "00:16:05.000",
                "entities": ["person", "dog"],
                "objects": ["ball", "fence", "trees"],
                "stuff": ["grass", "dirt"],
                "primary_activity": "person playing fetch with dog",
                "segment_summary": "person and dog playing fetch in a park setting",
                "key_moments": [
                    {
                        "start": "00:15:58.000",
                        "end": "00:15:59.500",
                        "caption": "person winds up and throws red ball across field"
                    },
                    {
                        "start": "00:16:02.000",
                        "end": "00:16:03.500",
                        "caption": "dog leaps and catches ball mid-air"
                    }
                ]
            }
        ]
    }

    try:
        analysis = VideoAnalysis(**sample_data)
        print("✓ Validation successful!")
        print(f"Analysis contains {len(analysis.segments)} segment(s)")
        for i, segment in enumerate(analysis.segments):
            print(f"  Segment {i + 1}: {segment.start} - {segment.end}")
            print(f"    Activity: {segment.primary_activity}")
            print(f"    Key moments: {len(segment.key_moments)}")
    except Exception as e:
        print(f"✗ Validation failed: {e}")