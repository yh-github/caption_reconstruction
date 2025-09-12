import logging
import os
import time
from collections import defaultdict
from pathlib import Path

import diskcache
import yaml
from google import genai

from common_utils.error_handling import ExceptionStr
from common_utils.jsonables import get_model_schema_lines, dump_model_compact_json
from common_utils.tracking import setup_logging, get_datetime_str
from experiment_executor.config_loader import load_config, get_llm_config
from data_models.complex_struct import VideoAnalysis, VideoSegment
from dev_qa import QAData, load_wild_captions, build_evaluator, AnswerResponse, AnswerResponses
from llm.llm_interaction import LLM_Manager_Builder, LLM_Response
from llm.parsers import parse_llm_response
from llm.prompting import JSONPromptBuilder, numbered_list
from data.video_link_loader import load_wild_dataset

logger = logging.getLogger(__name__)


def save_to_file(path, video_id:str, thoughts:str, _answers:AnswerResponses, _scores:list[float], mask_start:float|None=None, mask_end:float|None=None):
    with open(path, 'w') as f:
        f.write(yaml.dump({
            "video_id": video_id,
            "mask_start": mask_start,
            "mask_end": mask_end,
            "thoughts": thoughts,
            "answers": _answers.model_dump(),
            "scores": _scores
        }, sort_keys=False))


def save_error(path, video_id:str, llm_response:LLM_Response, exception:Exception, mask_start:float|None=None, mask_end:float|None=None):
    with open(path, 'w') as f:
        return f.write(yaml.dump(
            {
                "video_id": video_id,
                "mask_start": mask_start,
                "mask_end": mask_end,
                "llm_response": None if not llm_response else llm_response.model_dump_json(exclude_none=True),
                # "last_raw_response": None if not last_raw_response else last_raw_response.model_dump_json(),
                "exception": ExceptionStr(exception)
            }, sort_keys=False
        ))


######################
# MAIN
######################

if __name__ == "__main__":

    config = load_config('/home/yoavh/code/research/caption_reconstruction/config/qa/wild_text1.yaml')

    llm_config = get_llm_config(config)

    run_id = Path(__file__).stem + "__" + get_datetime_str()
    log_path, notification_logger = setup_logging(
        run_id=run_id,
        log_dir=config["paths"]["log_dir"],
        base_level=logging.INFO,
        console_level=logging.WARNING
    )

    wild_vs = load_wild_dataset(Path(config['data_config']['path']))
    qa_by_id:defaultdict[str, list[QAData]] = defaultdict(list)
    for i,v in enumerate(wild_vs):
        qa_by_id[v.video_id].append(QAData.model_validate(v.model_dump()))

    wild_captions = list(load_wild_captions(Path(config['data_config']['path_captions'])))

    evaluator = build_evaluator(qa_by_id, wild_captions)

    prompt_builder = JSONPromptBuilder.from_config(llm_config)
    prompt_builder.set_consts({'INSTRUCT_INPUT_SCHEMA': "\n".join(get_model_schema_lines(VideoSegment, level=1))})
    def gen_prompt(_va:VideoAnalysis, qa_info:list[QAData]):
        return prompt_builder.with_vars({
            'INPUT_VIDEO': dump_model_compact_json(_va.segments, width=200),
            'INPUT_QUESTIONS': numbered_list((qa.question for qa in qa_info))
        })


    run_name = f'llm_{config["__parent_run_name__"]}'
    def do_eval(video_id:str, answer_res:list[AnswerResponse], ground_truth:list[QAData]) -> list[float]:
        qi = 1
        scores = []
        for ar, gt in zip(answer_res, ground_truth):
            assert ar.question_index == qi, f"Question index mismatch at position {qi}. Expected index {qi}, but got {ar.question_index}."
            score =  evaluator.evaluate(ar, gt)
            scores.append(round(score, 3))
            qi += 1
            m = f"{run_name=} {video_id}__{qi} bs_f1={score}"
            logger.info(m)
            notification_logger.info(m)
        return scores

    out_path = Path('results') / run_id / run_name
    out_path.mkdir(parents=True, exist_ok=True)

    with diskcache.Cache(directory=config['paths']['disk_cache']) as llm_cache:
        llm_builder = LLM_Manager_Builder(genai.Client(), llm_cache)
        llm = llm_builder.from_config(llm_config)
        for va in wild_captions:
            if not va.video_id in qa_by_id:
                logging.warning(f"no QA for video_id={va.video_id}")
                continue

            OUT_FILE = out_path / f'{va.video_id}.yaml'
            ERR_FILE = out_path / f'error__{va.video_id}__{get_datetime_str()}.yaml'
            if os.path.exists(OUT_FILE) and os.path.getsize(OUT_FILE) > 0:
                logging.info(f"Skipping {va.video_id} - output file already exists")
                continue
            else:
                logging.info(f'processing {va.video_id}, writing to {OUT_FILE}')

            llm_input = gen_prompt(va, qa_by_id[va.video_id])
            logging.debug(f'{llm_input=}')

            try:
                res = llm.call(llm_input)
                assert res and res.text, f"Bad LLM Response for {va.video_id}"
                answers = parse_llm_response(model=AnswerResponses, response_text=res.text)
                scores=do_eval(va.video_id, answers.root, qa_by_id[va.video_id])
                save_to_file(OUT_FILE, va.video_id, res.thoughts, answers, scores)
            except Exception as e:
                logging.error(f"Error saving {va.video_id} to file, {e=}, saving to {ERR_FILE=}")
                save_error(ERR_FILE, va.video_id, res, e)
                time.sleep(1)
