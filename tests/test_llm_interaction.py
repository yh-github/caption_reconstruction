# tests/test_llm_interaction.py
import os
import pytest
from unittest.mock import MagicMock
from tenacity import RetryError

# Import from the new, correct module
from llm_interaction import call_llm

def test_llm_call_with_retry_on_failure(mocker):
    """
    Tests that the tenacity @retry decorator is working.
    """
    # Arrange
    mocker.patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"})
    mock_genai = mocker.patch('llm_interaction.genai')

    mock_genai.GenerativeModel.return_value.generate_content.side_effect = [
        Exception("API Error"),
        MagicMock(text="Success")
    ]

    # Act
    response_text = call_llm(mock_genai.GenerativeModel.return_value, "test prompt")

    # Assert
    assert mock_genai.GenerativeModel.return_value.generate_content.call_count == 2
    assert response_text == "Success"

def test_llm_call_fails_after_all_retries(mocker):
    """
    Tests that the function raises a RetryError after all attempts fail.
    """
    # Arrange
    mocker.patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"})
    mock_genai = mocker.patch('llm_interaction.genai')
    mock_genai.GenerativeModel.return_value.generate_content.side_effect = Exception("Persistent API Error")
    
    # Act & Assert
    with pytest.raises(RetryError):
        call_llm(mock_genai.GenerativeModel.return_value, "test prompt")
    
    assert mock_genai.GenerativeModel.return_value.generate_content.call_count == 6 # Tenacity default + 5 retries
