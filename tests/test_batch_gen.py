import unittest
from unittest.mock import MagicMock
import torch
from llm.local_llm import HuggingFaceModelAdapter
from llm.logits_processor import HeterogeneousLogitsProcessor

class TestBatchGeneration(unittest.TestCase):
    
    def test_generate_batch_mock(self):
        """
        Tests the orchestration logic of generate_batch using mocks.
        Does NOT load actual model.
        """
        # 1. Setup Mock Adapter
        adapter = HuggingFaceModelAdapter(model_key="phi-3") # Valid key
        adapter.model = MagicMock()
        adapter.tokenizer = MagicMock()
        adapter.device = "cpu"
        
        # Mock Tokenizer
        adapter.tokenizer.pad_token = "[PAD]"
        adapter.tokenizer.eos_token = "[EOS]"
        adapter.tokenizer.pad_token_id = 0
        adapter.tokenizer.eos_token_id = 1
        adapter.tokenizer.apply_chat_template.side_effect = lambda msgs, **kwargs: " ".join([m['content'] for m in msgs])
        
        # Mock Tokenizer call (batch encoding)
        mock_inputs = MagicMock()
        # Use a real tensor but don't try to assign shape manually
        mock_inputs.input_ids = torch.tensor([[1, 2, 3], [4, 5, 0]]) # Batch 2, varying length
        mock_inputs.input_ids.shape # Accessing is fine
        mock_inputs.to.return_value = mock_inputs
        adapter.tokenizer.return_value = mock_inputs

        # Mock Model Generation
        # Return something slightly longer than input
        # Input len 3. Output len 5.
        mock_output = torch.tensor([
            [1, 2, 3, 10, 11], 
            [4, 5, 0, 12, 13]
        ])
        adapter.model.generate.return_value = mock_output
        
        # Mock Decoding
        adapter.tokenizer.decode.side_effect = ["RESPONSE_A", "RESPONSE_B"]
        
        # 2. Call generate_batch
        messages = [[{"role":"user", "content":"A"}], [{"role":"user", "content":"B"}]]
        temps = [0.1, 0.9]
        pens = [1.0, 1.2]
        
        responses = adapter.generate_batch(messages, temps, pens)
        
        # 3. Assertions
        self.assertEqual(responses, ["RESPONSE_A", "RESPONSE_B"])
        
        # Check if LogitsProcessor was passed
        _, kwargs = adapter.model.generate.call_args
        self.assertIn("logits_processor", kwargs)
        lp_list = kwargs["logits_processor"]
        self.assertEqual(len(lp_list), 1)
        self.assertIsInstance(lp_list[0], HeterogeneousLogitsProcessor)
        
        # Verify params in LP
        proc = lp_list[0]
        self.assertTrue(torch.equal(proc.temperatures, torch.tensor([[0.1], [0.9]], dtype=torch.float16)))
        self.assertTrue(torch.equal(proc.penalties, torch.tensor([1.0, 1.2], dtype=torch.float16)))

if __name__ == "__main__":
    unittest.main()
