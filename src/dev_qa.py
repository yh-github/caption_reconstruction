import logging
from collections import defaultdict
from pathlib import Path
from typing import Iterator
from pydantic import BaseModel, Field
from data_models.complex_struct import VideoAnalysis, VideoSegment
from utils import get_model_schema_lines, dump_model_compact_json
from video_link_loader import load_wild_dataset, WildVideoMetadata



class QAData(BaseModel):
    question:str
    answer:str
    alter_answers:list[str]
    question_type:list[str]

def print_qa_info(wvm:WildVideoMetadata):
    oq = ''
    if wvm.question!=wvm.original_question:
        oq = f" (OQ: {wvm.original_question})"
    oa = ''
    if wvm.answer!=wvm.original_answer:
        oa = f" (OA: {wvm.original_answer})"
    ls=[wvm.video_id,
        f"{wvm.question_base=}, {wvm.question_type=} {wvm.objective=}",
        f"Q> {wvm.question}{oq}",
        f"A> {wvm.answer}{oa} {wvm.alter_answers}",
        f"{wvm.evidences=}",
        f"{wvm.evidences_in_min=}",
        f"{wvm.alter_evidences=}"
        ]
    print("\n".join([str(x) for x in ls]))
    print()

def print_qa_by_id(qa_by_id:dict[str,list[QAData]]):
    for vi, vid_id in enumerate(qa_by_id.keys(), start=1):
        print(f"{vi}. {vid_id}")
        for qi, qa in enumerate(qa_by_id[vid_id], start=1):
            print(f"\t{qi}. {qa.question}")


def load_wild_captions(path: Path) -> Iterator[VideoAnalysis]:
    for json_file in path.glob('*.json'):
        with open(json_file, 'r') as f:
            yield VideoAnalysis.model_validate(json.load(f))


class SupportingEvidence(BaseModel):
    start:str = Field(..., description="Start time of the evidence in the video, in HH:MM:SS.mmm format")
    end:str = Field(..., description="End time of the evidence in the video, in HH:MM:SS.mmm format")
    explanation:str = Field(..., description="Explanation of why you think the evidence supports your answer")

class AnswerResponse(BaseModel):
    question_index:int = Field(...,
        description="Index of the question that you are responding to")
    evidence:list[SupportingEvidence] = Field(...,
        description="List of supporting evidence for your answer")
    confidence:float = Field(..., ge=0.0, le=1.0,
        description="Confidence score for your answer between 0 (wild guess) and 1 (completely certain)")
    is_guess: bool = Field(...,
        description="Whether your answer is a *guess* or not")
    answer: str = Field(...,
        description="Your answer to the question")

import json

def compact_model_dump_json(model) -> str:
    """
    Custom compact model_dump_json implementation.
    Converts the Pydantic object to JSON with small indentation for compactness and readability.
    """
    # Use `model_dump` to get the Python data structure, then serialize with `json.dumps`
    return json.dumps(model.model_dump(), indent=2, separators=(",", ": "))


def gen_prompt(va:VideoAnalysis, qa_info:list[QAData]):

    def input_schema_explained():
        return "\n".join(get_model_schema_lines(VideoSegment, level=1))

    instructions = f"""\
##Instructions:

Answer the given question(s) based on the provided textual video information. Use this escalation approach:
1. **Answer directly** if there's sufficient information
2. **Infer** the most likely answer from the available context if the information is incomplete, indirect, or ambiguous
3. **Make an educated guess** if no evidence exists

###Output instructions:
1. **Answer:** Your answer will be automatically evaluated against the correct answer, so prioritize accurate phrasing and precise terminology.
2. **Evidence:** If you find supporting evidence in the data, always include the most relevant timestamps that support your answer.
3. **Confidence:** Report your confidence level as 0.8-1.0 for direct answers, 0.5-0.7 for inferences, 0.1-0.4 for guesses.

###Input instructions:
The video information is provided below in a JSON array.
Each JSON object represents a distinct video segment with detailed analysis, with the following fields:
{input_schema_explained()}
"""

    questions = "".join(f"{qi}. {qa.question}\n" for qi,qa in enumerate(qa_info, start=1))

    input_data = dump_model_compact_json(va.segments, width=200)
    # input_data = ("[\n" +
    #             ",\n".join([
    #                 '  '+s.model_dump_json(indent=2, exclude_none=True)
    #                 for s in va.segments
    #             ])
    #             + "\n]")

    return (f"{instructions}\n"
            f"##Input:\n\n"
            f"###Video Info:\n"
            f"{input_data}\n"
            f"\n###Question(s):\n"
            f"{questions}")


vs = load_wild_dataset(Path('/home/yoavh/code/research/caption_reconstruction/datasets/wildQA/dev_Agriculture.json'))
qa_by_id = defaultdict(list)
for i,v in enumerate(vs):
    qa_by_id[v.video_id].append(QAData.model_validate(v.model_dump()))
    # print_qa_info(v)

for va in load_wild_captions(Path('/home/yoavh/code/research/caption_reconstruction/datasets/wildQA/captions__wild1')):
    if not va.video_id in qa_by_id:
        logging.warning(f"no QA for video_id={va.video_id}")
        continue
    print(va.video_id)
    print(gen_prompt(va, qa_by_id[va.video_id]))
    print('---')
    # print()
    # print('\n'.join(get_model_schema_lines(AnswerResponse, level=0)))