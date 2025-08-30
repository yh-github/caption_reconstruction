import json
import logging
import os
import sys
import time
from pathlib import Path

import diskcache
import yaml
from google import genai
from google.genai import types
from google.genai.types import GenerateContentResponse

from config_loader import load_config, get_llm_config
from data_models.video_link import VideoLinkData
from data_models.complex_struct import VideoAnalysis
from llm_interaction import LLM_Response, LLM_Manager_Builder
from utils import setup_logging, get_datetime_str
from video_link_loader import load_wild_dataset, WildVideoMetadata


def gen_content_prompt_multi(vls:list[VideoLinkData], prompt:str, fps:int) -> types.Content:
    video_parts=[
        types.Part(
            file_data=types.FileData(file_uri=vl.uri),
            video_metadata=types.VideoMetadata(
                start_offset=f'{vl.start_offset}s',
                end_offset=f'{vl.end_offset}s',
                fps=fps
            )
        ) for vl in vls
    ]
    return types.Content(parts=video_parts+[types.Part(text=prompt)])

def gen_content_prompt(vl:VideoLinkData, prompt:str, fps:int) -> types.Content:
    return types.Content(
        parts=[
            types.Part(
                file_data=types.FileData(file_uri=vl.uri),
                video_metadata=types.VideoMetadata(
                    start_offset=f'{vl.start_offset}s',
                    end_offset=f'{vl.end_offset}s',
                    fps=fps
                )
            ),
            types.Part(text=prompt)
        ]
    )

def to_yt_links(vs:list[WildVideoMetadata]) -> list[VideoLinkData]:
    _links = {v.to_link() for v in vs}
    _v_ids = {v.video_id for v in vs}
    assert len(_v_ids) == len(_links)
    return list(_links)

def load_wild_links(path:str|Path, duration_limit:int|None) -> list[VideoLinkData]:
    try:
        path = Path(path)
        vs = list(load_wild_dataset(path))
        links = to_yt_links(vs)
        if duration_limit:
            links = [x.limit_duration(duration_limit) for x in links]
        return links
    except Exception as e:
        print(f'Usage: {sys.argv[0]} <config>')
        print('Error:', e)
        sys.exit(-1)

def save_to_file(path, video_id, text):
    segments = json.loads(text)
    va = VideoAnalysis(video_id=video_id, segments=segments)
    with open(path, 'w') as f:
        f.write(va.model_dump_json())


def save_error(path, video_id:str, llm_response:LLM_Response, last_raw_response:GenerateContentResponse|None, exception:Exception):
    with open(path, 'w') as f:
        return f.write(yaml.dump(
            {
                "video_id": video_id,
                "llm_response": None if not llm_response else llm_response.model_dump_json(),
                "last_raw_response": None if not last_raw_response else last_raw_response.model_dump_json(),
                "exception": str(exception)
            }
        ))

def read_prompt(path:str|Path) -> str:
    with open(path, 'r') as f:
        return f.read()


def main(config):
    llm_config = get_llm_config(config)
    prompt_text = read_prompt(llm_config['prompt_template'])
    run_id = Path(__file__).stem +"__"+ get_datetime_str()
    setup_logging(
        run_id=run_id,
        log_dir=config["paths"]["log_dir"],
        base_level=logging.INFO,
        console_level=logging.WARNING
    )
    duration_limit = config["data_config"].get("duration_limit")
    links = load_wild_links(config["data_config"]["path"], duration_limit)
    print(f'{len(links) = }')

    cache_dir = config["paths"]["disk_cache"]
    logging.info(f"Cache dir: {cache_dir}")
    out_path = Path(config["data_config"]["out_path"])
    out_path = out_path.with_name(out_path.name+f"__{config['__parent_run_name__']}")
    out_path.mkdir(parents=True, exist_ok=True)

    with diskcache.Cache(cache_dir) as llm_cache:
        llm_builder = LLM_Manager_Builder(genai.Client(), llm_cache)

        llm = llm_builder.from_config(llm_config)

        for x in links:
            if x.duration() < duration_limit:
                logging.info(f"Skipping {x.video_id} - too short, duration={x.duration()} < {duration_limit=}")
                continue
            OUT_FILE = out_path/f'{x.video_id}.json'
            ERR_FILE = out_path/f'error__{x.video_id}__{get_datetime_str()}.yaml'
            if os.path.exists(OUT_FILE) and os.path.getsize(OUT_FILE) > 0:
                logging.info(f"Skipping {x.video_id} - output file already exists")
                continue
            else:
                logging.info(f'processing {x}, writing to {OUT_FILE}')

            llm_input=gen_content_prompt(x, prompt_text, llm_config['fps'])
            logging.info(f'{llm_input=}')

            try:
                res = llm.call(llm_input)
                assert res and res.text, f"No response for {x.video_id}"
                if duration_limit and len(res.text.splitlines())!=duration_limit:
                    logging.warning(f'video_id={x.video_id} llm_output_len={len(res.text.splitlines())} but should be {duration_limit}')
                save_to_file(OUT_FILE, x.video_id, res.text)
            except Exception as e:
                logging.error(f"Error saving {x.video_id} to file, {e=}, saving to {ERR_FILE=}")
                save_error(ERR_FILE, x.video_id, res, llm.last_raw_response, e)
                time.sleep(1)

if __name__ == "__main__":
    main(load_config(sys.argv[1]))
