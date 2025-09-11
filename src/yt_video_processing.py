import logging
import os
import sys
import time
from datetime import timedelta
from pathlib import Path

import diskcache
import yaml
from google import genai
from google.genai import types
from pydantic import BaseModel

from config_loader import load_config, get_llm_config
from data_models.video_link import VideoLinkData
from llm.llm_interaction import LLM_Response, LLM_Manager_Builder
from llm.parsers import parse_llm_response_list, T_BaseModel
from utils import setup_logging, get_datetime_str, ExceptionStr
from data.video_link_loader import load_wild_dataset, WildVideoMetadata


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

def load_wild_links(
    path:str|Path,
    duration_limit:float|None,
    max_size:int|None
) -> list[VideoLinkData]:
    try:
        path = Path(path)
        vs = list(load_wild_dataset(path))
        links = to_yt_links(vs)
        if duration_limit:
            links = [x.limit_duration(duration_limit+0.5) for x in links
                     if x.duration()>=duration_limit]
        links.sort(key=lambda x: (x.video_id, x.start_offset, x.end_offset))
        if max_size:
            return links[:max_size]
        return links
    except Exception as e:
        print(f'Usage: {sys.argv[0]} <config>')
        print('Error:', e)
        sys.exit(-1)

def save_to_file(path, video_id, validated_captions:list[T_BaseModel], thoughts:str|None):
    # va = VideoAnalysis(video_id=video_id, segments=segments)
    class VideoCaptions(BaseModel):
        video_id: str
        captions: list[T_BaseModel]
        thoughts: str|None

    va = VideoCaptions.model_validate({
        'video_id': video_id,
        'captions': validated_captions,
        'thoughts': thoughts
    })
    with open(path, 'w') as f:
        f.write(va.model_dump_json())

def save_error(
    path:Path,
    video_id:str,
    llm_response:LLM_Response|None,
    exception:Exception|None
):
    with open(path, 'w') as f:
        f.write(yaml.dump(
            {
                "video_id": video_id,
                "llm_response": None if not llm_response else llm_response.model_dump_json(exclude_none=True),
                # "last_raw_response": None if not last_raw_response else last_raw_response.model_dump_json(),
                "exception": ExceptionStr(exception) if exception else None
            }
        ))
        time.sleep(1)

def read_prompt(path:str|Path) -> str:
    with open(path, 'r') as f:
        return f.read()

def main(config):
    llm_config = get_llm_config(config)

    prompt_text = read_prompt(llm_config['prompt_template'])
    run_id = Path(__file__).stem +"__"+ get_datetime_str()
    log_path, notification_logger = setup_logging(
        run_id=run_id,
        log_dir=config["paths"]["log_dir"],
        base_level=logging.INFO,
        console_level=logging.WARNING
    )
    duration_limit = config["data_config"].get("duration_limit")

    links = load_wild_links(
        path=config["data_config"]["path"],
        duration_limit=duration_limit,
        max_size=config["data_config"]["limit"]
    )
    print(f'{len(links) = }')

    cache_dir = config["paths"]["disk_cache"]
    logging.info(f"Cache dir: {cache_dir}")
    out_path = Path(config["data_config"]["out_path"])
    out_path = out_path.with_name(out_path.name+f"__{config['__parent_run_name__']}")
    out_path.mkdir(parents=True, exist_ok=True)

    with diskcache.Cache(cache_dir) as llm_cache:
        llm_builder = LLM_Manager_Builder(genai.Client(), llm_cache)

        response_schema = llm_builder.config_response_schema(llm_config.get('response_schema'))
        assert response_schema is not None

        ok = 0
        def validate_res(video_link: VideoLinkData, text: str, expected_len: int = duration_limit) -> list[T_BaseModel]:
            video_id=video_link.video_id
            captions:list[T_BaseModel] = parse_llm_response_list(response_schema, text)
            assert len(captions)
            if expected_len and abs(len(captions) - expected_len) > 1:
                logging.warning(f'{video_id=} {len(captions)=} but {expected_len=}')

            start=captions[0].model_dump().get('start',"NO_START")
            end=captions[-1].model_dump().get('end', "NO_END")
            captions_time=f"{start}-{end}"
            video_link_time=f"{timedelta(seconds=video_link.start_offset)}-{timedelta(seconds=video_link.end_offset)}"
            notification_logger.info(f'{ok+1} {video_id=} {video_link_time} {captions_time}')

            return captions

        llm = llm_builder.from_config(llm_config)

        for x in links:
            if x.duration() < duration_limit:
                logging.warning(f"Skipping {x.video_id} - too short, duration={x.duration()} < {duration_limit=}")
                ok += 1
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

            res = None
            try:
                res = llm.call(llm_input)
                assert res and res.text, f"Bad LLM Response for {x.video_id}"
                save_to_file(
                    OUT_FILE,
                    x.video_id,
                    validate_res(x, res.text, duration_limit),
                    res.thoughts
                )
                ok += 1
            except Exception as e:
                logging.error(f"Error with {x.video_id}, {e=}, saving to {ERR_FILE=}")
                save_error(ERR_FILE, x.video_id, res, e)
    print(f'{log_path = }')

if __name__ == "__main__":
    main(load_config(sys.argv[1]))
