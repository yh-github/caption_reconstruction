from __future__ import annotations

import re
import logging
from typing import Type, TypeVar, Union, Optional
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


def repair_truncated_json_array(text: str) -> str | None:
    """Repairs unclosed JSON arrays caused by token limit truncation."""
    last_brace = text.rfind("}")
    if last_brace == -1:
        return None
    candidate = text[:last_brace + 1].strip()
    if not candidate.endswith("]"):
        candidate += "\n]"
    if not candidate.startswith("["):
        first_bracket = candidate.find("[")
        if first_bracket != -1:
            candidate = candidate[first_bracket:]
        else:
            candidate = "[\n" + candidate
    return candidate


def fallback_extract_captions(text: str) -> list[dict]:
    """Extracts valid {'index': X, 'caption': Y} objects from malformed or truncated text."""
    results = []
    pattern = re.compile(
        r'\{\s*(?:'
        r'\"index\"\s*:\s*(\d+)[^{}]*?\"caption\"\s*:\s*\"((?:[^\"\\\\]|\\\\.)*)\"'
        r'|'
        r'\"caption\"\s*:\s*\"((?:[^\"\\\\]|\\\\.)*)\"[^{}]*?\"index\"\s*:\s*(\d+)'
        r')',
        re.DOTALL
    )
    for m in pattern.finditer(text):
        if m.group(1) is not None:
            idx = int(m.group(1))
            cap = m.group(2)
        else:
            idx = int(m.group(4))
            cap = m.group(3)
        try:
            cap = cap.encode("utf-8").decode("unicode_escape")
        except Exception:
            pass
        results.append({"index": idx, "caption": cap.strip()})
    return results


def parse_llm_response(model: Type[T_BaseModel], response_text: str) -> T_BaseModel | None:
    """
    Parses the raw text response from the LLM and validates it against the provided model.
    Handles markdown code blocks and conversational wrapping gracefully.

    Args:
        model: The Pydantic model class to validate the response against.
        response_text: The raw string output from the LLM.

    Returns:
        An instance of the provided model if parsing is successful, otherwise None.
    """
    logging.debug("Parsing LLM response...")
    if not response_text:
        logging.error("Empty LLM response received.")
        return None

    cleaned_text = response_text.strip()
    # 1. Extract from markdown code block if present
    match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", cleaned_text)
    if match:
        cleaned_text = match.group(1).strip()
    elif not (cleaned_text.startswith("[") or cleaned_text.startswith("{")):
        # 2. Extract first JSON array or object
        match_json = re.search(r"(\[[\s\S]*\]|\{[\s\S]*\})", cleaned_text)
        if match_json:
            cleaned_text = match_json.group(1).strip()

    try:
        validated_response = model.model_validate_json(cleaned_text)
        logging.debug(f"LLM response parsed and validated successfully: {validated_response}")
        return validated_response
    except Exception as e:
        # 3. Attempt repair of truncated JSON array (e.g. cut off at token ceiling)
        repaired = repair_truncated_json_array(cleaned_text)
        if repaired:
            try:
                validated_response = model.model_validate_json(repaired)
                logging.info("LLM response parsed successfully after repairing truncated JSON array.")
                return validated_response
            except Exception:
                pass

        # 4. Fallback recovery: extract individual complete caption entries via regex
        fallback_items = fallback_extract_captions(cleaned_text)
        if fallback_items:
            try:
                validated_response = model.model_validate(fallback_items)
                logging.info(f"LLM response recovered {len(fallback_items)} items via fallback regex parser.")
                return validated_response
            except Exception:
                pass

        logging.warning(f"Failed to parse LLM response as {model.__name__}: {e}. Snippet: {cleaned_text[:200]}")
        return None
