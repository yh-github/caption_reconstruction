import logging
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import diskcache
from google import genai

from config_loader import load_config, get_llm_config
from data_models.video_link import VideoLinkData
from dev_qa import QAData, load_wild_captions, build_evaluator, AnswerResponse, AnswerResponses, \
    MASKED_VIDEO_INSTRUCTIONS
from dev_qa_main import save_to_file, save_error
from llm_interaction import LLM_Manager_Builder
from parsers import parse_llm_response
from prompting import JSONPromptBuilder
from utils import numbered_list, get_datetime_str, setup_logging
from video_link_loader import load_wild_dataset
from yt_video_processing import gen_content_prompt_multi

logger = logging.getLogger(__name__)

######################
# MAIN
######################

def main(config:dict[str, Any], mask_start:float|None=None, mask_end:float|None=None):
    _mask_str = ""
    if mask_start is not None:
        assert mask_end > mask_start, "mask_end must be specified if mask_start is specified"
        _mask_str = f"__{mask_start}-{mask_end}"
    assert config['data_config']['input_type'] == "video links"
    llm_config = get_llm_config(config)

    run_id = config['__run_id__']

    log_path, notification_logger = setup_logging(
        run_id=run_id,
        log_dir=config["paths"]["log_dir"],
        base_level=logging.INFO,
        console_level=logging.WARNING
    )

    qa_by_id:defaultdict[str, list[QAData]] = defaultdict(list)
    link_by_id:dict[str, VideoLinkData] = {}
    for v in load_wild_dataset(Path(config['data_config']['path'])):
        qa_by_id[v.video_id].append(QAData.model_validate(v.model_dump()))
        link = v.to_link()
        if link in link_by_id and link_by_id[v.video_id] != link:
            raise Exception(f"Duplicate link for {v.video_id}")
        link_by_id[v.video_id] = link

    # captions are just for IDF and data filtering
    wild_captions = list(load_wild_captions(Path(config['data_config']['path_captions'])))
    evaluator = build_evaluator(qa_by_id, wild_captions)

    prompt_builder = JSONPromptBuilder.from_config(llm_config)

    def gen_text_prompt(qa_info: list[QAData], masked:bool=False) -> str:

        return prompt_builder.with_vars({
            'INPUT_QUESTIONS': numbered_list(qa.question for qa in qa_info),
            'MASKED_VIDEO_INSTRUCTIONS': MASKED_VIDEO_INSTRUCTIONS if masked else ""
        })

    run_name = f'vlm_{config["__parent_run_name__"]}'
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
            v_id = va.video_id+_mask_str
            if not va.video_id in qa_by_id:
                logging.warning(f"no QA for video_id={va.video_id}")
                continue

            OUT_FILE = out_path / f'{v_id}.yaml'
            ERR_FILE = out_path / f'error__{v_id}__{get_datetime_str()}.yaml'
            if os.path.exists(OUT_FILE) and os.path.getsize(OUT_FILE) > 0:
                logging.info(f"Skipping {v_id} - output file already exists")
                continue
            else:
                logging.info(f'processing {v_id}, writing to {OUT_FILE}')

            vls = link_by_id[va.video_id].optional_mask(mask_start, mask_end)
            if len(vls)>1:
                logging.debug(f'video masked by splitting: {[f"{v.start_offset}-{v.end_offset}" for v in vls]}')
            llm_input = gen_content_prompt_multi(
                vls=vls,
                prompt=gen_text_prompt(qa_by_id[va.video_id]),
                fps=llm_config['fps']
            )
            logging.debug(f'{llm_input=}')
            try:
                res = llm.call(llm_input)
                assert res and res.text, f"Bad response for {v_id}"
                answers = parse_llm_response(model=AnswerResponses, response_text=res.text)
                scores=do_eval(v_id, answers.root, qa_by_id[va.video_id])
                save_to_file(OUT_FILE, va.video_id, res.thoughts, answers, scores, mask_start, mask_end)
            except Exception as e:
                logging.error(f"Error saving {va.video_id} to file, {e=}, saving to {ERR_FILE=}")
                save_error(ERR_FILE, va.video_id, res, e, mask_start, mask_end)
                time.sleep(1)

if __name__ == "__main__":
    dargs = dict(enumerate(sys.argv[1:], start=1))
    config_path = dargs.get(1, 'config/qa/wild_video1.yaml')
    config = load_config(config_path)
    config['__run_id__'] = dargs.get(2, Path(__file__).stem) # + "__" + get_datetime_str()
    main(config, 0.4, 0.6)
