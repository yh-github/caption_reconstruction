import logging
from pydantic import ValidationError
from data_models import ReconstructedCaptions
import json


def parse_llm_response(response_text: str) -> ReconstructedCaptions | None:
    """
    Parses the raw text response from the LLM.

    It expects the response to be a JSON string that can be validated
    by the LLMResponse model.

    Args:
        response_text: The raw string output from the LLM.

    Returns:
        A list of CaptionedClip objects if parsing is successful,
        otherwise returns None.
    """
    logging.debug("Parsing LLM response...")
    try:
        validated_response = ReconstructedCaptions.model_validate_json(response_text)
        if validated_response:
            logging.debug(f"LLM response parsed and validated successfully.")
        return validated_response
    except json.JSONDecodeError:
        logging.error("Failed to parse LLM response: Invalid JSON format.")
        return None
    except ValidationError as e:
        logging.error(f"Failed to validate LLM response: {response_text=} {e=}")
        return None
