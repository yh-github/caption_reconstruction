import logging
from collections import defaultdict
from pathlib import Path
from typing import Iterator
from pydantic import BaseModel, Field
from data_models.complex_struct import VideoAnalysis, VideoSegment
from prompting import JSONPromptBuilder
from utils import get_model_schema_lines, dump_model_compact_json, numbered_list
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


prompt_builder = JSONPromptBuilder.from_path('/home/yoavh/code/research/caption_reconstruction/prompts/qa/text1.txt')
prompt_builder.set_consts({'INSTRUCT_INPUT_SCHEMA': "\n".join(get_model_schema_lines(VideoSegment, level=1))})

def gen_prompt(va:VideoAnalysis, qa_info:list[QAData]):
    return prompt_builder.with_vars({
        'INPUT_VIDEO': dump_model_compact_json(va.segments, width=200),
        'INPUT_QUESTIONS': numbered_list((qa.question for qa in qa_info))
    })

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