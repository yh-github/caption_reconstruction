import logging
from typing import Type, TypeVar
from pydantic import BaseModel, TypeAdapter

# Define a generic type for BaseModel
T_BaseModel = TypeVar("T_BaseModel", bound=BaseModel)

from typing import get_origin
def validate_json(model:Type[list[T_BaseModel]]|Type[T_BaseModel], text:str) -> list[T_BaseModel]:
    if get_origin(model) == list:
        return TypeAdapter(model).validate_json(text)
    return [model.model_validate_json(text)]

def parse_llm_response_list(
        model: Type[T_BaseModel]|Type[list[T_BaseModel]],
        response_text: str
) -> list[T_BaseModel]:
    """
    Parses the raw text response from the LLM and validates it against the provided model.

    Args:
        model: The Pydantic model class to validate the response against.
        response_text: The raw string output from the LLM.


    Returns:
        list of instances of the provided model if parsing is successful, otherwise [].
    """
    logging.debug("Parsing LLM response...")

    assert response_text, "Empty LLM response received."
    # Handle cases where the response might be wrapped in code blocks
    if response_text.startswith("```json") and response_text.endswith("```"):
        response_text = response_text[7:-3]

    # Validate against the provided model
    validated_response:list[T_BaseModel] = validate_json(model, response_text)
    logging.debug(f"LLM response parsed and validated successfully: {validated_response}")
    return validated_response


def parse_llm_response(model: Type[T_BaseModel], response_text: str) -> T_BaseModel | None:
    """
    Parses the raw text response from the LLM and validates it against the provided model.

    Args:
        model: The Pydantic model class to validate the response against.
        response_text: The raw string output from the LLM.


    Returns:
        An instance of the provided model if parsing is successful, otherwise None.
    """
    logging.debug("Parsing LLM response...")
    assert response_text, "Empty LLM response received."

    # Handle cases where the response might be wrapped in code blocks
    if response_text.startswith("```json") and response_text.endswith("```"):
        response_text = response_text[7:-3]

    # Validate against the provided model
    validated_response = model.model_validate_json(response_text)
    logging.debug(f"LLM response parsed and validated successfully: {validated_response}")
    return validated_response
