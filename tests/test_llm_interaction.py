import pytest
from unittest.mock import MagicMock
from tenacity import RetryError
import google.api_core.exceptions

# Import the function we are going to test
from llm_interaction import call_llm

@pytest.fixture
def mock_llm_model(mocker):
    """A fixture to create a mock LLM model."""
    mock_model = MagicMock()
    # Mock the main genai module to return our mock_model instance
    mocker.patch('llm_interaction.genai.GenerativeModel', return_value=mock_model)
    return mock_model

def test_llm_call_with_retry_on_failure(mock_llm_model):
    """
    Tests that the tenacity @retry decorator works by simulating a
    rate limit error that is resolved on the second attempt.
    """
    # Arrange
    mock_llm_model.generate_content.side_effect = [
        google.api_core.exceptions.ResourceExhausted("Rate limit exceeded"),
        MagicMock(text="Success")
    ]
    
    # Act
    response_text = call_llm(mock_llm_model, "test prompt")

    # Assert
    assert mock_llm_model.generate_content.call_count == 2
    assert response_text == "Success"

def test_llm_call_fails_after_all_retries(mock_llm_model):
    """
    Tests that the function raises a RetryError after all attempts
    fail with the specific rate limit error.
    """
    # Arrange
    mock_llm_model.generate_content.side_effect = google.api_core.exceptions.ResourceExhausted("Rate limit exceeded")
    
    # Act & Assert
    with pytest.raises(RetryError):
        call_llm(mock_llm_model, "test prompt")
    
    assert mock_llm_model.generate_content.call_count == 6

def test_llm_call_fails_immediately_on_other_exceptions(mock_llm_model):
    """
    Tests that if a non-rate-limit error occurs, the function fails
    immediately without retrying.
    """
    # Arrange
    # Configure the mock to raise a generic ValueError
    mock_llm_model.generate_content.side_effect = ValueError("An unexpected error")
    
    # Act & Assert
    # Use pytest.raises to confirm that the original ValueError is raised,
    # not a tenacity.RetryError.
    with pytest.raises(ValueError, match="An unexpected error"):
        call_llm(mock_llm_model, "test prompt")
    
    # Assert that the API was called exactly once, proving no retries happened.
    mock_llm_model.generate_content.assert_called_once()
