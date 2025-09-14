import json
import logging
import statistics
from pathlib import Path
from typing import Iterator
from bert_score import BERTScorer
from pydantic import BaseModel, Field, RootModel
from data_models.complex_struct import VideoAnalysis
from data.video_link_loader import WildVideoMetadata

logger = logging.getLogger(__name__)

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

class AnswerResponses(RootModel[list[AnswerResponse]]):
    pass


class QAEvaluator_BertScore:
    """
    Encapsulates the logic for evaluating caption reconstruction using BERTScore.
    """

    def __init__(self, model_type:str|None=None, text_for_idf:list[str]|None=None, verbose=False):
        self.model_type = model_type
        self.verbose = verbose
        self.bert_scorer = BERTScorer(
            model_type=self.model_type,
            idf=bool(text_for_idf),
            idf_sents=text_for_idf,
            use_fast_tokenizer=False,
            lang="en"
        )
        idf_msg="without IDF"
        if text_for_idf:
            # noinspection PyProtectedMember
            idf_msg = f'calc_idf for {len(text_for_idf)} sentences, idf_dict size = {len(self.bert_scorer._idf_dict.keys())}'
        logger.info(f"Evaluator initialized with model: {self.model_type}, {idf_msg}")


    @staticmethod
    def candidate_reference_pairs(answer_res: AnswerResponse, ground_truth: QAData) -> tuple[list[str], list[str]]:
        """
        Helper method to extract reference and candidate sentences.
        """
        references = [ground_truth.answer]+ground_truth.alter_answers
        candidates = [answer_res.answer]*len(references)
        return candidates, references

    def evaluate(self, answer_res: AnswerResponse, ground_truth: QAData) -> float:
        logger.debug("Aligning answers for BERTScore evaluation...")

        candidates, references = self.candidate_reference_pairs(answer_res, ground_truth)

        if not candidates:
            raise Exception(f"Nothing found to evaluate {answer_res=} {ground_truth=}")

        logger.debug(f"Calculating BERTScore for {len(candidates)} pairs.")

        bs_p, bs_r, bs_f1 = self.bert_scorer.score(
            cands=candidates,
            refs=references,
            batch_size=4
        )

        return bs_f1.max().item()

    @staticmethod
    def agg_metrics(all_metrics):
        mean_f1 = statistics.mean([m['bs_f1'].min().item() for m in all_metrics])
        mean_precision = statistics.mean([m['bs_p'].min().item() for m in all_metrics])
        mean_recall = statistics.mean([m['bs_r'].min().item() for m in all_metrics])

        return {
            "num_of_instances": len(all_metrics),
            "mean_f1_score": mean_f1,
            "mean_precision": mean_precision,
            "mean_recall": mean_recall
        }

def build_evaluator(qa_by_id:dict[str, list[QAData]], wild_captions:list[VideoAnalysis]):
    sents = []
    for vs in qa_by_id.values():
        for v in vs:
            sents.append(v.question)
            sents.append(v.answer)
            sents.extend(v.alter_answers)
    for c in wild_captions:
        if c.video_id not in qa_by_id:
            continue
        for s in c.segments:
            sents.extend(s.segment_summary)
            for k in s.key_moments:
                sents.append(k.caption)
    return QAEvaluator_BertScore(model_type='microsoft/deberta-large-mnli', text_for_idf=sents)


MASKED_VIDEO_INSTRUCTIONS="""
###Input instructions:
The video has a *masked* section, so the input consists of 2 parts.
The masked part, between the end of the first part and the start of the second part, is hidden from you.
It was chosen randomly and may or may not contain important information.
"""