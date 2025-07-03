# src/parsers.py
import json
import logging
from pydantic import BaseModel, ValidationError
from data_models import TranscriptClip, NarrativeOnlyPayload, StructuredPayload

class LLMResponse(BaseModel):
    """
    A Pydantic model to validate the structure of the JSON response
    we expect from the LLM. It expects a list of TranscriptClip objects.
    """
    reconstructed_transcript: list[TranscriptClip]

def parse_llm_response(response_text: str) -> list[TranscriptClip] | None:
    """
    Parses the raw text response from the LLM.

    It expects the response to be a JSON string that can be validated
    by the LLMResponse model.

    Args:
        response_text: The raw string output from the LLM.

    Returns:
        A list of TranscriptClip objects if parsing is successful,
        otherwise returns None.
    """
    logging.info("Parsing LLM response...")
    try:
        # Pydantic can directly validate the JSON string.
        # This is a robust way to parse and validate in one step.
        # We need to wrap our list in a dictionary to match the model.
        wrapped_json_string = f'{{"reconstructed_transcript": {response_text}}}'
        
        validated_response = LLMResponse.model_validate_json(wrapped_json_string)
        
        logging.info("LLM response parsed and validated successfully.")
        return validated_response.reconstructed_transcript

    except json.JSONDecodeError:
        logging.error("Failed to parse LLM response: Invalid JSON format.")
        return None
    except ValidationError as e:
        logging.error(f"Failed to validate LLM response: {e}")
        return None
