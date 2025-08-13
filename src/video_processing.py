import json
import sys
from pathlib import Path

import diskcache
from google.genai import types

from data_models.captions_only import VideoLinkData
from llm_interaction import LLM_Manager
from video_link_loader import load_wild_dataset

llm_config = {
    'model_name': "gemini-2.5-pro",
    'thought_budget': -1,
    'temperature': 1.0,
    'seed': 0x5EED,
    'system_instructions': None
}

prompt = """
Create dense captions for the video. Your output will consist of a list of objects:
{"start":str, "end":str, "caption":str}

For example:
{
    "start": "00:00:00.000",
    "end": "00:00:06.803",
    "caption": "a man in a gray t-shirt and hat is standing in a field of tall green plants, speaking to the camera."
}
""".strip()

def gen_content_prompt(vl:VideoLinkData) -> types.Content:
    return types.Content(
        parts=[
            types.Part(
                file_data=types.FileData(file_uri=vl.uri),
                video_metadata=types.VideoMetadata(
                    start_offset=f'{vl.start_offset}s',
                    end_offset=f'{vl.end_offset}s',
                    fps=1
                )
            ),
            types.Part(text=prompt)
        ]
    )


if __name__ == "__main__":
    vs = load_wild_dataset(Path(sys.argv[1]), limit=1)

    with diskcache.Cache('./disk_cache/llm_video_cache') as llm_cache:

        llm = LLM_Manager(
            model_name=llm_config['model_name'],
            seed=llm_config['seed'],
            temperature=llm_config['temperature'],
            system_instruction=llm_config.get('system_instructions'),
            thought_budget=llm_config.get('thought_budget', 0),
            llm_cache=llm_cache,
            response_schema=None
        )

        for v in vs:
            l = v.to_link()
            print(v.video_id, l)
            p=gen_content_prompt(l)
            print()
            print(p)
            print()
            res=llm.call(p)
            print('===> response <===')
            print(res)
            print()
            print('===> thoughts parsed <===')
            print(res.thoughts.encode().decode('utf-8'))
            print()
            print('===> response parsed as json <===')
            for x in json.loads(res.text):
                print(x)
            print()