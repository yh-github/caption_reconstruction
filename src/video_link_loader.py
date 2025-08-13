from pathlib import Path
import json
from pydantic import BaseModel, RootModel
from data_models.captions_only import VideoLinkData


class TimeInFrames(BaseModel):
    """Example:
    {
        "start": "00:15:31.297",
        "end": "00:16:20.279"
    }
    """
    start: str
    end: str



class TimeInSeconds(BaseModel):
    """Example:
    {
        "start": "931.2970333333334",
        "end": "980.2793"
    }
    """
    start: str
    end: str


class TimeInOriginalVideo(BaseModel):
    frames: TimeInFrames
    seconds: TimeInSeconds


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
        def uri(u):
            return u.replace('//watch', '/watch')

        t = self.time_in_original_video.seconds
        return VideoLinkData(
            uri=uri(self.url_for_original_video),
            start_offset=float(t.start),
            end_offset=float(t.end)
        )



def load_wild_dataset(path:Path, limit:int=None):
    with open(path) as f:
        j=json.load(f)
    limit=limit or len(j)
    for v in j[:limit]:
        yield WildVideoMetadata.model_validate(v)
