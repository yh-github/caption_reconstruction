import sys
from unittest.mock import MagicMock, patch
import logging

# MOCKING dependencies before import to bypass broken environment
sys.modules["sentence_transformers"] = MagicMock()
sys.modules["sentence_transformers.SentenceTransformer"] = MagicMock()

# Now we can import our class (it will import the mock)
from llm.local_embedder import LocalEmbedder

logging.basicConfig(level=logging.DEBUG)

def test_local_embedder_logic():
    print("Testing LocalEmbedder logic with MOCKS...")
    
    # Setup Mock
    mock_model_cls = sys.modules["sentence_transformers"].SentenceTransformer
    mock_model_instance = MagicMock()
    mock_model_cls.return_value = mock_model_instance
    mock_model_instance.get_sentence_embedding_dimension.return_value = 3 # Match mock side_effect output size
    
    # Mock encode to return dummy embeddings
    # Input: list of N strings. Output: list of N arrays.
    def side_effect(texts, **kwargs):
        return [[0.1, 0.2, 0.3]] * len(texts)
    
    mock_model_instance.encode.side_effect = side_effect
    
    # Initialize
    embedder = LocalEmbedder(model_name="test-model")
    embedder.cache.clear() # Clear cache explicitly for test
    
    # Test 1: New Embeddings
    texts = ["a_unique_1", "b_unique_1"]
    embs = embedder.get_embeddings("vid1", texts)
    
    assert len(embs) == 2
    assert embs[0] == [0.1, 0.2, 0.3]
    # Verify model was called
    mock_model_instance.encode.assert_called()
    print("Test 1 (New) Passed")
    
    # Test 2: Caching
    # Reset mock
    mock_model_instance.encode.reset_mock()
    
    # Call again with same texts
    embs2 = embedder.get_embeddings("vid1", texts)
    assert len(embs2) == 2
    # Model should NOT be called (cache hit)
    mock_model_instance.encode.assert_not_called()
    print("Test 2 (Cache) Passed")
    
    # Test 3: Mixed
    new_texts = ["a_unique_1", "c_unique_1"]
    embs3 = embedder.get_embeddings("vid1", new_texts)
    assert len(embs3) == 2
    # Model should be called ONLY for "c_unique_1"
    args, _ = mock_model_instance.encode.call_args
    assert args[0] == ["c_unique_1"]
    print("Test 3 (Mixed) Passed")

    # Test 4: Duplicates
    # Reset mock
    mock_model_instance.encode.reset_mock()
    dup_texts = ["d_dup", "d_dup", "e_unique"]
    
    # We expect 3 embeddings back
    embs4 = embedder.get_embeddings("vid1", dup_texts)
    assert len(embs4) == 3
    
    # But the model should only compute 2 (d_dup and e_unique)
    args, _ = mock_model_instance.encode.call_args
    # The set of new texts order is not guaranteed, so we check length or set equality
    computed_texts = args[0]
    assert len(computed_texts) == 2
    assert set(computed_texts) == {"d_dup", "e_unique"}
    print("Test 4 (Duplicates) Passed")

if __name__ == "__main__":
    test_local_embedder_logic()
