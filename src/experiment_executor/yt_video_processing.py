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

import subprocess
import tempfile
from experiment_executor.config_loader import load_config, get_llm_config
from data_models.video_link import VideoLinkData, VideoLocalData
from llm.llm_interaction import LLM_Response, LLM_Manager_Builder
from llm.parsers import parse_llm_response_list, T_BaseModel
from common_utils.tracking import setup_logging, get_datetime_str
from common_utils.error_handling import ExceptionStr
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

def gen_content_prompt_local(vl: VideoLocalData, prompt: str, fps: int) -> types.Content:
    duration_to_extract = vl.clip_duration
    
    with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp_file:
        tmp_path = tmp_file.name
        
    try:
        cmd = [
            'ffmpeg', '-y', '-i', str(vl.path), 
            '-t', str(duration_to_extract), 
            '-c', 'copy', tmp_path
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        with open(tmp_path, 'rb') as f:
            video_bytes = f.read()
            
        return types.Content(
            parts=[
                types.Part(
                    inline_data=types.Blob(
                        data=video_bytes,
                        mime_type='video/mp4'
                    ),
                    video_metadata=types.VideoMetadata(fps=fps)
                ),
                types.Part(text=prompt)
            ]
        )
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

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

def _normalize_name(name: str) -> str:
    """Normalize video name for matching: apostrophes -> underscores, lowercase."""
    return name.replace("'", "_").lower()

def _build_local_cache(local_dirs: list[Path]) -> dict[str, Path]:
    """Build a dict mapping normalized stem -> Path for all video files in given directories."""
    cache = {}
    for local_dir in local_dirs:
        for root, _dirs, files in os.walk(local_dir):
            for f in files:
                if Path(f).suffix.lower() in ['.mp4', '.mkv', '.webm']:
                    stem = Path(f).stem
                    normalized = _normalize_name(stem)
                    if normalized not in cache:
                        cache[normalized] = Path(root) / f
    return cache

def load_local_links(
    path: str | Path,
    local_dirs: list[Path],
    duration_limit: float | None,
    max_size: int | None
) -> list[VideoLocalData]:
    try:
        path = Path(path)
        vs = list(load_wild_dataset(path))
        links = []
        
        # Build a normalized-stem -> path lookup across all local dirs
        video_paths_cache = _build_local_cache(local_dirs)
        logging.info(f"Local video cache: {len(video_paths_cache)} files from {len(local_dirs)} directories")
        
        not_found = []
        for v in vs:
            video_id = v.video_id
            normalized_id = _normalize_name(video_id)
            
            if normalized_id not in video_paths_cache:
                not_found.append(video_id)
                continue
                
            local_path = video_paths_cache[normalized_id]
            vl = VideoLocalData(
                video_id=video_id,
                path=local_path,
                clip_duration=v.duration
            )
            links.append(vl)
        
        if not_found:
            logging.warning(f"Could not find local files for {len(not_found)} video_ids (out of {len(vs)}). "
                            f"First 5: {not_found[:5]}")
            
        if duration_limit:
            links = [x.limit_duration(duration_limit + 0.5) for x in links if x.duration() >= duration_limit]
            
        links.sort(key=lambda x: x.video_id)
        if max_size:
            return links[:max_size]
        return links
    except Exception as e:
        print(f'Error loading local links: {e}')
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

import argparse
from google.api_core import exceptions

def main(args):
    config = load_config(args.config_path)
    dry_run = args.dry_run

    llm_loader = None 
    # Only verify LLM config validness in dry-run, don't build client yet unless needed? 
    # actually getting config is safe.
    llm_config = get_llm_config(config)

    prompt_text = read_prompt(llm_config['prompt_template'])
    run_id = Path(__file__).stem +"__"+ get_datetime_str()
    
    # In dry-run, maybe use console logging only? kept same for consistency
    log_path, notification_logger = setup_logging(
        run_id=run_id,
        log_dir=config["paths"]["log_dir"],
        base_level=logging.INFO,
        console_level=logging.WARNING
    )
    
    data_conf = config["data_config"]
    duration_limit = data_conf.get("duration_limit")
    
    # local_dir can be a single string or a list of strings (in config or CLI)
    local_dir_conf = data_conf.get("local_dir")
    if not local_dir_conf:
        local_dir_conf = getattr(args, "local_dir", None)

    # Normalize to a list of Paths
    if isinstance(local_dir_conf, str):
        local_dirs = [Path(local_dir_conf)]
    elif isinstance(local_dir_conf, list):
        local_dirs = [Path(p) for p in local_dir_conf]
    else:
        local_dirs = []
    
    # Filter to existing directories
    local_dirs = [d for d in local_dirs if d.exists()]
    
    if local_dirs:
        logging.info(f"Using local video directories: {local_dirs}")
        links = load_local_links(
            path=data_conf["path"],
            local_dirs=local_dirs,
            duration_limit=duration_limit,
            max_size=data_conf["limit"]
        )
        is_local = True
    else:
        logging.info("Using YouTube URLs")
        links = load_wild_links(
            path=data_conf["path"],
            duration_limit=duration_limit,
            max_size=data_conf["limit"]
        )
        is_local = False
    
    out_path = Path(data_conf["out_path"])
    out_path = out_path.with_name(out_path.name+f"__{config['__parent_run_name__']}")
    
    print(f"--- Configuration Summary ---")
    print(f"Dataset: {data_conf['name']}")
    print(f"Input Path: {data_conf['path']}")
    print(f"Total Links Found: {len(links)}")
    print(f"Output Directory: {out_path}")
    print(f"Model: {llm_config['model_name']}")
    print(f"Dry Run: {dry_run}")
    print(f"-----------------------------")

    if dry_run:
        # Check how many would be skipped
        skipped_count = 0
        to_process = []
        for x in links:
            if duration_limit is not None and x.duration() < duration_limit:
                 continue
            
            OUT_FILE = out_path/f'{x.video_id}.json'
            if os.path.exists(OUT_FILE) and os.path.getsize(OUT_FILE) > 0:
                skipped_count += 1
            else:
                to_process.append(x.video_id)
        
        print(f"Dry Run Analysis:")
        print(f"  Total Valid Links: {len(links)}")
        print(f"  Already Processed: {skipped_count}")
        print(f"  Would Process: {len(to_process)}")
        if to_process:
            print(f"  First 5 to process: {to_process[:5]}")
        return

    out_path.mkdir(parents=True, exist_ok=True)
    cache_dir = config["paths"]["disk_cache"]
    logging.info(f"Cache dir: {cache_dir}")

    with diskcache.Cache(cache_dir) as llm_cache:
        llm_builder = LLM_Manager_Builder(genai.Client(), llm_cache)

        response_schema = llm_builder.config_response_schema(llm_config.get('response_schema'))
        assert response_schema is not None

        ok = 0
        def validate_res(video_link: VideoLinkData | VideoLocalData, text: str, expected_len: int = duration_limit) -> list[T_BaseModel]:
            video_id=video_link.video_id
            captions:list[T_BaseModel] = parse_llm_response_list(response_schema, text)
            assert len(captions)
            if expected_len and abs(len(captions) - expected_len) > 1:
                logging.warning(f'{video_id=} {len(captions)=} but {expected_len=}')

            start=captions[0].model_dump().get('start',"NO_START")
            end=captions[-1].model_dump().get('end', "NO_END")
            captions_time=f"{start}-{end}"
            
            if isinstance(video_link, VideoLocalData):
                video_link_time = f"local file duration {video_link.clip_duration}"
            else:
                video_link_time=f"{timedelta(seconds=video_link.start_offset)}-{timedelta(seconds=video_link.end_offset)}"
            notification_logger.info(f'{ok+1} {video_id=} {video_link_time} {captions_time}')

            return captions

        llm = llm_builder.from_config(llm_config)

        consecutive_errors = 0
        
        for x in links:
            if duration_limit is not None and x.duration() < duration_limit:
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

            if is_local:
                llm_input=gen_content_prompt_local(x, prompt_text, llm_config['fps'])
            else:
                llm_input=gen_content_prompt(x, prompt_text, llm_config['fps'])
                
            logging.info(f'{llm_input=}')

            res = None
            try:
                res = llm.call(llm_input)
                
                # Check for Blocked/Error response that might contain the exception
                if hasattr(res, 'exception') and res.exception:
                    # If it's a Quota issue that surfaced here
                    if "429" in str(res.exception) or "ResourceExhausted" in str(res.exception):
                         raise exceptions.ResourceExhausted(str(res.exception))

                assert res and res.text, f"Bad LLM Response for {x.video_id}"
                
                save_to_file(
                    OUT_FILE,
                    x.video_id,
                    validate_res(x, res.text, duration_limit),
                    res.thoughts
                )
                ok += 1
                consecutive_errors = 0 # Reset on success
            except exceptions.ResourceExhausted as re:
                logging.critical(f"QUOTA EXCEEDED for {x.video_id}: {re}")
                save_error(ERR_FILE, x.video_id, res, re)
                print(f"\nCRITICAL: Quota Limit Reached. Stopping execution to prevent further errors.")
                break
            except Exception as e:
                logging.error(f"Error with {x.video_id}, {e=}, saving to {ERR_FILE=}")
                save_error(ERR_FILE, x.video_id, res, e)
                consecutive_errors += 1
                if consecutive_errors >= 5:
                     logging.critical(f"Too many consecutive errors ({consecutive_errors}). Stopping.")
                     break
                
    print(f'{log_path = }')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process YT videos to generate captions.")
    parser.add_argument("config_path", help="Path to the yaml configuration file")
    parser.add_argument("--dry-run", action="store_true", help="Run without calling LLMs to check inputs/outputs")
    parser.add_argument("--local-dir", type=str, default="local/wild_videos_raw/Videos1/", help="Path to local videos directory")
    
    args = parser.parse_args()
    main(args)
