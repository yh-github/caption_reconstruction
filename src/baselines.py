# src/baselines.py
import logging
from data_models import TranscriptClip
from constants import DATA_MISSING

def repeat_last_known_baseline(masked_transcript: list[TranscriptClip]) -> list[TranscriptClip]:
    """
    Fills masked clips by repeating the data from the last known clip.

    This is a simple, non-LLM baseline for reconstruction.
    """
    logging.info("Running 'repeat_last_known' baseline...")
    reconstructed_clips = []
    last_known_data = None

    for clip in masked_transcript:
        if clip.data != DATA_MISSING:
            # This is an unmasked clip, so we use its data and update our memory.
            reconstructed_clips.append(clip)
            last_known_data = clip.data
        else:
            # This is a masked clip.
            new_clip = clip.model_copy()
            if last_known_data is not None:
                # If we have a previously seen clip, use its data.
                new_clip.data = last_known_data
            else:
                # Edge case: The very first clip is masked.
                # We have nothing to repeat, so we leave it masked.
                # The evaluation function will treat this as a failure.
                new_clip.data = DATA_MISSING
            reconstructed_clips.append(new_clip)
            
    return reconstructed_clips
