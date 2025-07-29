import sys
import json
from pydantic import BaseModel, Field
from typing import Dict, Optional

from data_models import CaptionedVideo, CaptionedClip
from config_loader import load_config
from data_loaders import get_data_loader
from exceptions import UserFacingError
from reconstruction_strategies import Reconstructed

def ls_recon(path):
    with open(path, 'r') as f:
        i = 1
        for line in f:
            r = Reconstructed.model_validate_json(line)
            print(f"{i}. {r.video_id} {r.metrics or 'FAIL'}")
            i += 1

def load_recon(path, index=None, video_id=None):
    if index is None and not video_id:
        raise Exception('need index or video_id')
    with open(path, 'r') as f:
        i = 1
        for line in f:
            r = Reconstructed.model_validate_json(line)
            if i==index or r.video_id==video_id:
                return r
            i += 1
    raise Exception('not found')

def main():
    """
    Loads a dataset, finds a specific video, and compares its original clips
    to a hypothetical reconstructed version.
    """
    # --- Setup ---
    if len(sys.argv) < 3:
        raise UserFacingError("[cmd] [config] [artifact]")

    cmd = sys.argv[1]
    config = load_config(sys.argv[2])
    art_path = sys.argv[3].removeprefix('file://')

    data_loader = get_data_loader(config["data_config"])

    if cmd=='ls':
        ls_recon(art_path)
        return

    if cmd.startswith('i='):
        reconstructed_data = load_recon(path=art_path, index=int(cmd.split('=')[1]))
    else:
        reconstructed_data = load_recon(path=art_path, video_id=cmd)

    print()
    print(f"Analyzing reconstruction for video_id: '{reconstructed_data.video_id}'")

    # --- Find the Original Video ---
    original_video = data_loader.find(reconstructed_data.video_id)

    if not original_video:
        print(f"❌ Error: Could not find original video with ID '{reconstructed_data.video_id}' in the dataset.")
        return

    if reconstructed_data.debug_data:
        llm_response = reconstructed_data.debug_data.pop('llm_response_text')
        print(reconstructed_data.debug_data)
        print('LLM_RESPONSE:')
        print(llm_response)
        print(':LLM_RESPONSE')
        print("ORIG:")
        print(original_video.model_dump_json(indent=4))
        print(":ORIG")
        return

    # --- Print the Comparison ---
    # print(f"--- Comparison for Video: {original_video.video_id} ---")
    # print(f'| Index | Original Caption | Reconstructed Caption | Metrics |')
    # print(f'|---|---|---|---|')  # Markdown table header separator

    j = 0
    for i, original_clip in enumerate(original_video.clips):
        original_desc = original_clip.data.caption

        # Check if this clip was reconstructed
        if i in reconstructed_data.reconstructed_clips:
            recon_clip = reconstructed_data.reconstructed_clips[i]
            recon_desc = recon_clip.data.caption

            # Format the metrics for this specific clip
            f1 = reconstructed_data.metrics.get('bs_f1')[j]
            j += 1
            metrics_str = f"F1={f1}"

            # Use markdown for emphasis
            print(f'{i}. ~~{original_desc}~~')
            print('  ', recon_desc)
            print('  ', metrics_str)
        else:
            # This clip was not in the masked set
            print(f'{i}. {original_desc}')


if __name__ == "__main__":
    main()