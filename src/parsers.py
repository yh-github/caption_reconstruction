import json
import logging
from typing import Type, TypeVar
from pydantic import BaseModel, ValidationError, TypeAdapter

# Define a generic type for BaseModel
T = TypeVar("T", bound=BaseModel)

from typing import get_origin
def validate_json(model:Type[list[T]]|Type[T], text:str) -> list[T]:
    if get_origin(model) == list:
        return TypeAdapter(model).validate_json(text)
    return [model.model_validate_json(text)]

def parse_llm_response_list(model: Type[T]|Type[list[T]], response_text: str) -> list[T]:
    """
    Parses the raw text response from the LLM and validates it against the provided model.

    Args:
        model: The Pydantic model class to validate the response against.
        response_text: The raw string output from the LLM.


    Returns:
        list of instances of the provided model if parsing is successful, otherwise [].
    """
    logging.debug("Parsing LLM response...")

    try:
        if not response_text:
            logging.warning("Empty LLM response received.")
            return []

        # Handle cases where the response might be wrapped in code blocks
        if response_text.startswith("```json") and response_text.endswith("```"):
            response_text = response_text[7:-3]

        # Validate against the provided model
        validated_response:list[T] = validate_json(model, response_text)
        logging.debug(f"LLM response parsed and validated successfully: {validated_response}")
        return validated_response
    except json.JSONDecodeError:
        logging.error("Failed to parse LLM response: Invalid JSON format.")
        return []
    except ValidationError as e:
        logging.error(f"Failed to validate LLM response: {response_text=} {e=}")
        return []


def parse_llm_response(model: Type[T]|Type[list[T]], response_text: str) -> T | None:
    """
    Parses the raw text response from the LLM and validates it against the provided model.

    Args:
        model: The Pydantic model class to validate the response against.
        response_text: The raw string output from the LLM.


    Returns:
        An instance of the provided model if parsing is successful, otherwise None.
    """
    logging.debug("Parsing LLM response...")

    try:
        if not response_text:
            logging.warning("Empty LLM response received.")
            return None

        # Handle cases where the response might be wrapped in code blocks
        if response_text.startswith("```json") and response_text.endswith("```"):
            response_text = response_text[7:-3]

        # Validate against the provided model
        validated_response = model.model_validate_json(response_text)
        logging.debug(f"LLM response parsed and validated successfully: {validated_response}")
        return validated_response

    except json.JSONDecodeError:
        logging.error("Failed to parse LLM response: Invalid JSON format.")
        return None
    except ValidationError as e:
        logging.error(f"Failed to validate LLM response: {response_text=} {e=}")
        return None
