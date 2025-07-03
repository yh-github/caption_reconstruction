from data_models import TranscriptClip, NarrativeOnlyPayload

def create_toy_transcript() -> list[TranscriptClip]:
    """Creates the toy transcript using the new data models."""
    clips_data = [
        {"timestamp": 1.0, "description": "A person walks into a room from the left."},
        # ... (rest of the descriptions from before) ...
        {"timestamp": 10.0, "description": "The person walks out of the room to the left."}
    ]

    # Create the list of TranscriptClip objects, wrapping the data in the payload model.
    transcript = [
        TranscriptClip(
            timestamp=item["timestamp"],
            data=NarrativeOnlyPayload(description=item["description"])
        )
        for item in clips_data
    ]
    return transcript
