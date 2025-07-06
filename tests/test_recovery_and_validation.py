# tests/test_recovery_and_validation.py
import os
import json
import pytest
from unittest.mock import MagicMock
from tenacity import RetryError

# Functions to test
from main import call_llm
from parsers import parse_llm_response

# A sample valid JSON response string from the LLM
VALID_JSON_RESPONSE = """
[
    {
        "timestamp": 1.0,
        "data": { "description": "This is a valid reconstruction." }
    }
]
"""

# A sample malformed JSON response string
MALFORMED_JSON_RESPONSE = '["invalid json"'

# A sample non-JSON text response
NON_JSON_RESPONSE = "I am sorry, I cannot fulfill this request."


def test_parser_with_valid_llm_response():
    """Tests that the parser succeeds with a perfect LLM response."""
    # Act
    result = parse_llm_response(VALID_JSON_RESPONSE)
    # Assert
    assert result is not None
    assert len(result) == 1
    assert result[0].data.description == "This is a valid reconstruction."

def test_parser_with_malformed_llm_response():
    """Tests that the parser fails gracefully with malformed JSON."""
    # Act
    result = parse_llm_response(MALFORMED_JSON_RESPONSE)
    # Assert
    assert result is None

def test_parser_with_non_json_llm_response():
    """Tests that the parser fails gracefully with a non-JSON string."""
    # Act
    result = parse_llm_response(NON_JSON_RESPONSE)
    # Assert
    assert result is None

def test_llm_call_with_retry_on_failure(mocker):
    """
    Tests that the tenacity @retry decorator is working.
    We simulate the API failing multiple times before succeeding.
    """
    # Arrange
    mocker.patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"})
    mock_genai = mocker.patch('main.genai')

    # Configure the mock to first raise an exception, then return a valid response
    mock_genai.GenerativeModel.return_value.generate_content.side_effect = [
        Exception("API Error"), # First call fails
        MagicMock(text="Success") # Second call succeeds
    ]

    # Act
    response_text = call_llm("test prompt", {'llm': {'model_name': 'test'}})

    # Assert
    # Check that the API was called twice (1 failure + 1 success)
    assert mock_genai.GenerativeModel.return_value.generate_content.call_count == 2
    assert response_text == "Success"

def test_llm_call_fails_after_all_retries(mocker):
    """
    Tests that the function raises a RetryError after all attempts fail.
    """
    # Arrange
    mocker.patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"})
    mock_genai = mocker.patch('main.genai')

    # Configure the mock to always raise an exception
    mock_genai.GenerativeModel.return_value.generate_content.side_effect = Exception("Persistent API Error")
    
    # Act & Assert
    # Use pytest.raises to confirm that a RetryError is the final exception
    with pytest.raises(RetryError):
        call_llm("test prompt", {'llm': {'model_name': 'test'}})

    # Check that the API was called 5 times (our configured number of retries)
    assert mock_genai.GenerativeModel.return_value.generate_content.call_count == 5
