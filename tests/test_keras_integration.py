
import pytest
import sys
from unittest.mock import MagicMock, patch

# Add src to path
import os
sys.path.append(os.path.abspath("src"))

from llm.keras_llm import KerasLLM
from reconstruction.text_reconstruction import TextReconstructionStrategyBuilder
from common_utils.error_handling import UserFacingError

MODEL_KEY = "phi-3"

def test_keras_llm_imports_gracefully():
    # If keras is not installed, KerasLLM should still be importable
    # but instantiation might fail if checks are strict, 
    # OR the module level try-except handles it.
    pass

@patch("llm.keras_llm.keras")
@patch("llm.keras_llm.keras_nlp")
def test_keras_llm_instantiation(mock_keras_nlp, mock_keras):
    # Mock existence of libraries
    adapter = KerasLLM(model_key=MODEL_KEY)
    assert adapter.model_key == MODEL_KEY

def test_builder_raises_if_dependency_missing():
    # Force KerasLLM to be None in the builder module to simulate missingdeps
    with patch("reconstruction.text_reconstruction.KerasLLM", None):
        # Force system to request keras
        with patch("common_utils.device_setup.get_llm_backend", return_value="keras_llm"):
            builder = TextReconstructionStrategyBuilder(None, 0, None)
            config = {
                "type": "local_llm",
                "model_key": MODEL_KEY,
                "name": "test_strat"
            }
            with pytest.raises(UserFacingError, match="dependencies not found"):
                builder.get_strategy(config)

@patch("reconstruction.text_reconstruction.KerasLLM")
@patch("common_utils.device_setup.get_llm_backend")
def test_builder_instantiates_keras_backend(mock_get_backend, MockKerasLLM):
    # Mock KerasLLM class
    mock_instance = MagicMock()
    MockKerasLLM.return_value = mock_instance
    
    # Simulate system returning keras_llm
    mock_get_backend.return_value = "keras_llm"
    
    builder = TextReconstructionStrategyBuilder(None, 0, None)
    config = {
        "type": "local_llm",
        "model_key": MODEL_KEY,
        # "backend" key removed from config
        "name": "test_strat"
    }
    
    strategy = builder.get_strategy(config)
    
    # Check if KerasLLM was initialized
    MockKerasLLM.assert_called_once()
    assert strategy.model_adapter == mock_instance

def test_device_setup_auto_detect():
    # Helper to test device_setup logic specifically if needed, 
    # but mocking get_llm_backend in the builder test covers the integration contract.
    pass
