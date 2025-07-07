# src/baselines.py
import logging
from data_models import TranscriptClip
from constants import DATA_MISSING

def repeat_last_known_baseline(masked_transcript: list[TranscriptClip]) -> list[TranscriptClip]:
    """
    Fills masked clips by repeating the data from the last known clip.
    If the initial clips are masked, it back-fills them using the first
    available non-masked clip.
    """
    logging.info("Running 'repeat_last_known' baseline...")
    if not masked_transcript:
        return []

    # --- First Pass: Find the first available data payload ---
    first_valid_data = None
    for clip in masked_transcript:
        if clip.data != DATA_MISSING:
            first_valid_data = clip.data
            break

    # --- Second Pass: Reconstruct the transcript ---
    reconstructed_clips = []
    last_known_data = first_valid_data # Start with the first valid data

    for clip in masked_transcript:
        new_clip = clip.model_copy()
        if clip.data != DATA_MISSING:
            # This is an unmasked clip, so we use its data and update our memory.
            last_known_data = clip.data
            new_clip.data = clip.data
        else:
            # This is a masked clip. Fill it with the last known data.
            # If all clips were masked, `last_known_data` will be None,
            # and they will correctly remain masked.
            new_clip.data = last_known_data
        
        reconstructed_clips.append(new_clip)
            
    return reconstructed_clips
