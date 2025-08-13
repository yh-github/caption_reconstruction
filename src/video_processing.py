import json
import os
import sys
from pathlib import Path

import diskcache
from google.genai import types

from data_models.captions_only import VideoLinkData
from data_models.complex_struct import VideoSegment, VideoAnalysis
from llm_interaction import LLM_Manager
from video_link_loader import load_wild_dataset

llm_config = {
    'model_name': "gemini-2.5-pro",
    'thought_budget': -1,
    'temperature': 1.0,
    'seed': 0x5EED,
    'system_instructions': None
}

prompt1 = """
Create dense captions for the video. Your output will consist of a list of objects:
{"start":str, "end":str, "caption":str}

For example:
{
    "start": "00:00:00.000",
    "end": "00:00:06.803",
    "caption": "a man in a gray t-shirt and hat is standing in a field of tall green plants, speaking to the camera."
}
""".strip()


prompt2 = """
# Video Analysis Prompt: Segment-Based Dense Captioning

Analyze this video and provide a high-level breakdown into distinct segments with detailed captioning in JSON format.

## Output Format

```json
{
  "segments": [
    {
      "start": "00:15:58.000",
      "end": "00:16:05.000",
      "entities": ["person", "dog"],
      "objects": ["ball", "fence", "trees"],
      "stuff": ["grass", "dirt"],
      "primary_activity": "playing fetch",
      "segment_summary": "person and dog playing fetch in a park setting",
      "key_moments": [
        {
          "start": "00:15:58.000",
          "end": "00:15:59.500",
          "caption": "person winds up and throws red ball across field"
        },
        {
          "start": "00:16:02.000",
          "end": "00:16:03.500",
          "caption": "dog leaps and catches ball mid-air"
        },
        {
          "start": "00:16:04.000",
          "end": "00:16:05.000",
          "caption": "dog trots back toward person with ball in mouth"
        }
      ]
    }
  ]
}
```

## Field Definitions

- **start/end**: Timestamps in HH:MM:SS.mmm format marking segment boundaries
- **entities**: People, animals, or robots (e.g., man, woman, child, dog, cat, robot)
- **objects**: Solid items like furniture, vehicles, tools, buildings, sports equipment
- **stuff**: Materials, substances, or textures like water, sand, smoke, fabric, grass
- **primary_activity**: Main action or activity occurring throughout the segment
- **segment_summary**: Concise description of the overall segment content
- **key_moments**: 2-5 notable visual events within the segment, each with precise timing

## Instructions

### Segmentation Rules
Create new segments when there are significant changes in:
- Scene or location
- Main subjects entering or leaving the frame
- Primary activity or focus of action
- Major camera perspective or angle shifts
- Lighting or environmental conditions

### Content Guidelines
1. **Independence**: Analyze each segment based solely on its visual content
2. **Capitalization**: Use lowercase except for proper nouns (brand names, specific locations) or acronyms
3. **Precision**: Be accurate and concise in descriptions
4. **Visual focus**: Describe only what is directly observable in the video
"""

more = """
5. **Objectivity**: Avoid inferring emotions, thoughts, or off-screen events

### Technical Requirements
1. **Timing**: Segments should typically be 3-30 seconds long
2. **JSON validity**: Ensure proper formatting and quote escaping
3. **Completeness**: Cover the entire video duration without gaps
4. **Key moments**: Include the most visually significant events within each segment
"""

examples = """## Additional Examples

### Example 1: Cooking Scene
```json
{
  "start": "00:02:15.000",
  "end": "00:02:28.000",
  "entities": ["chef"],
  "objects": ["knife", "cutting board", "carrots", "bowl"],
  "stuff": ["water"],
  "primary_activity": "chef chopping vegetables",
  "segment_summary": "chef prepares ingredients by dicing carrots on wooden cutting board",
  "key_moments": [
    {
      "start": "00:02:15.000",
      "end": "00:02:17.000",
      "caption": "chef positions knife above orange carrots"
    },
    {
      "start": "00:02:18.000",
      "end": "00:02:25.000",
      "caption": "chef rapidly dices carrots into small cubes"
    },
    {
      "start": "00:02:26.000",
      "end": "00:02:28.000",
      "caption": "chef sweeps diced carrots into white ceramic bowl"
    }
  ]
}
```

### Example 2: Street Scene
```json
{
  "start": "00:07:42.000",
  "end": "00:07:55.000",
  "entities": ["pedestrians", "cyclist"],
  "objects": ["bicycle", "cars", "traffic light", "crosswalk"],
  "stuff": ["asphalt", "concrete"],
  "primary_activity": "urban street activity with mixed traffic",
  "segment_summary": "busy intersection with pedestrians crossing and cyclist navigating traffic",
  "key_moments": [
    {
      "start": "00:07:42.000",
      "end": "00:07:44.000",
      "caption": "traffic light changes from red to green"
    },
    {
      "start": "00:07:45.000",
      "end": "00:07:49.000",
      "caption": "group of pedestrians crosses intersection in crosswalk"
    },
    {
      "start": "00:07:50.000",
      "end": "00:07:55.000",
      "caption": "cyclist weaves between stopped cars approaching intersection"
    }
  ]
}
```

## Quality Checklist

Before submitting, verify:
- [ ] All timestamps follow HH:MM:SS.mmm format
- [ ] Segments cover entire video without gaps or overlaps
- [ ] Each field contains appropriate content type
- [ ] Key moments highlight the most significant visual events
- [ ] JSON is properly formatted and valid
- [ ] Descriptions are concise yet descriptive
- [ ] Only observable visual elements are included"""

def gen_content_prompt(vl:VideoLinkData, prompt:str) -> types.Content:
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

def link_dict():
    d = {}
    for v in vs:
        k = v.video_id
        if k not in d:
            d[k] = []
        d[k].append(v.to_link())
    print(f'{len(vs) = }   {len(d) = }')
    for k,xs in d.items():
        print(f'{k}:')
        for x in xs:
            print(f'    {x}')

if __name__ == "__main__":
    path = Path("/home/yoavh/code/research/caption_reconstruction/datasets/wildQA/dev_Agriculture.json") # Path(sys.argv[1])
    vs=list(load_wild_dataset(path))
    links = {v.to_link() for v in vs}
    v_ids = {v.video_id for v in vs}
    assert len(v_ids) == len(links)
    print(f'{len(links) = }')

    with diskcache.Cache('./disk_cache/llm_video_cache') as llm_cache:

        llm = LLM_Manager(
            model_name=llm_config['model_name'],
            seed=llm_config['seed'],
            temperature=llm_config['temperature'],
            system_instruction=llm_config.get('system_instructions'),
            thought_budget=llm_config.get('thought_budget', 0),
            llm_cache=llm_cache,
            response_schema=list[VideoSegment]
        )

        for x in links:
            print(x)
            OUT_FILE = f'datasets/wildQA/captions/{x.video_id}.json'

            if os.path.exists(OUT_FILE) and os.path.getsize(OUT_FILE) > 0:
                print(f"Skipping {x.video_id} - output file already exists")
                continue

            p=gen_content_prompt(x, prompt2)
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
            segments = json.loads(res.text)
            for i,s in enumerate(segments):
                print(f'{i+1}. {s}')
            print()
            va = VideoAnalysis(video_id=x.video_id, segments=segments)
            with open(OUT_FILE, 'w') as f:
                f.write(va.model_dump_json())



