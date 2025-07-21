# src/prompting.py
import json
import mlflow
from data_models import CaptionedClip
from constants import DATA_MISSING

def build_prompt(masked_captions: list[CaptionedClip], config: dict) -> str:
    """Builds the final JSON prompt to be sent to the LLM."""
    prompt_template_path = config.get("strategy", {}).get("prompt_template")
    if not prompt_template_path:
        raise ValueError("Prompt template path not specified in config.")
    
    with open(prompt_template_path, 'r') as f:
        instruction_template = f.read().strip()

    instruction = instruction_template.format(DATA_MISSING=DATA_MISSING)
    
    caption_for_json = [clip.model_dump() for clip in masked_captions]
    json_prompt_data = json.dumps(caption_for_json, indent=2)
    
    final_prompt = f"{instruction}\n\n---\n\n{json_prompt_data}"
    
    if mlflow.active_run():
        mlflow.log_text(final_prompt, "prompt.txt")
        
    return final_prompt
