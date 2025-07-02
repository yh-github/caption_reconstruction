# tests/test_main.py
import os
from unittest.mock import MagicMock

# The function we want to test
from main import call_llm

def test_call_llm_api_invoked(mocker):
    """
    Tests that the call_llm function correctly invokes the genai library.
    It mocks the entire 'google.generativeai' library.
    """
    # Arrange
    
    # 1. Set a dummy API key as an environment variable for the test
    mocker.patch.dict(os.environ, {"GEMINI_API_KEY": "test-api-key"})

    # 2. Create a mock response object that the fake model will return
    # This simulates the structure of the real response object.
    mock_response = MagicMock()
    mock_response.text = "This is a mocked LLM response."

    # 3. Mock the genai library itself
    # We replace the entire 'google.generativeai' module used by main.py
    mock_genai = mocker.patch('main.genai')
    
    # Configure the mock so that when GenerativeModel is called, it returns
    # an object that has a 'generate_content' method, which in turn
    # returns our mock_response.
    mock_genai.GenerativeModel.return_value.generate_content.return_value = mock_response

    # 4. Define the inputs for the function we are testing
    test_prompt = "This is a test prompt."
    test_config = {
        'llm': {
            'model_name': 'gemini-1.5-pro-latest'
        }
    }

    # Act: Call the function we are testing
    response_text = call_llm(test_prompt, test_config)

    # Assert
    
    # 1. Check that the genai library was configured with our API key
    mock_genai.configure.assert_called_once_with
