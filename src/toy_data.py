from data_models import TranscriptClip, NarrativeOnlyPayload

def create_toy_transcript() -> list[TranscriptClip]:
    """
    Creates and returns a simple, manually-defined transcript with 10 clips.
    This serves as our initial ground-truth data for building and debugging
    the experimental pipeline.
    """
    clips_data = [
        {"timestamp": 1.0, "description": "A person walks into a room from the left."},
        {"timestamp": 2.0, "description": "The person approaches a table in the center."},
        {"timestamp": 3.0, "description": "The person picks up a red book from the table."},
        {"timestamp": 4.0, "description": "The person holds the book and looks at its cover."},
        {"timestamp": 5.0, "description": "The person opens the book to the first page."},
        {"timestamp": 6.0, "description": "The person turns a page in the book."},
        {"timestamp": 7.0, "description": "The person closes the book."},
        {"timestamp": 8.0, "description": "The person places the red book back on the table."},
        {"timestamp": 9.0, "description": "The person turns around."},
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
