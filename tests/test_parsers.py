from data_models import ReconstructedOutput
from parsers import parse_llm_response
import textwrap

def test_parse_llm_response_success():
    """
    Tests successful parsing of a clean, valid JSON response from the LLM.
    """
    # Arrange: A perfect JSON string as we'd hope to get from the LLM.
    recon_obj = ReconstructedOutput(reconstructed_captions={
        0: "The person approaches a table.",
        1: "The person picks up a book."
    })

    llm_output = recon_obj.model_dump_json(indent=4)

    # Act
    parsed:ReconstructedOutput|None = parse_llm_response(llm_output)
    assert parsed is not None
    assert parsed == recon_obj

def test_parse_llm_response_invalid_json():
    """
    Tests that the parser returns None when given a malformed JSON string.
    """
    # Arrange: A string that is not valid JSON.
    llm_output = """
    [
        {"timestamp": 2.0, "data": {"caption": "A bad response"}
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
    {
        "time": 2.0, 
        "payload": {"desc": "Wrong key names"}
    }
    """

    # Act
    parsed_clips = parse_llm_response(llm_output)

    # Assert
    assert parsed_clips is None
