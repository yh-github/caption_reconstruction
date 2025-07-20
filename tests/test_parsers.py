# tests/test_parsers.py
import pytest
from parsers import parse_llm_response
from data_models import CaptionedClip

def test_parse_llm_response_success():
    """
    Tests successful parsing of a clean, valid JSON response from the LLM.
    """
    # Arrange: A perfect JSON string as we'd hope to get from the LLM.
    llm_output = """
    [
        {
            "timestamp": 2.0,
            "data": {
                "description": "The person approaches a table."
            }
        },
        {
            "timestamp": 3.0,
            "data": {
                "description": "The person picks up a book."
            }
        }
    ]
    """

    # Act
    parsed_clips = parse_llm_response(llm_output)

    # Assert
    assert parsed_clips is not None
    assert len(parsed_clips) == 2
    assert isinstance(parsed_clips[0], CaptionedClip)
    assert parsed_clips[1].data.description == "The person picks up a book."

def test_parse_llm_response_invalid_json():
    """
    Tests that the parser returns None when given a malformed JSON string.
    """
    # Arrange: A string that is not valid JSON.
    llm_output = """
    [
        {"timestamp": 2.0, "data": {"description": "A bad response"}
    ]
    """ # Missing closing curly brace

    # Act
    parsed_clips = parse_llm_response(llm_output)

    # Assert
    assert parsed_clips is None

def test_parse_llm_response_validation_error():
    """
    Tests that the parser returns None when the JSON has the wrong structure.
    """
    # Arrange: Valid JSON, but it doesn't match our Pydantic model.
    llm_output = """
    [
        {
            "time": 2.0, 
            "payload": {"desc": "Wrong key names"}
        }
    ]
    """

    # Act
    parsed_clips = parse_llm_response(llm_output)

    # Assert
    assert parsed_clips is None
