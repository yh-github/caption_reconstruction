
import os
import logging
from typing import Any
try:
    import keras
    import keras_nlp
    # Optimize for TPU/JAX if available
    if keras.backend.backend() == "jax":
        # Enable mixed precision for TPU performance
        keras.mixed_precision.set_global_policy("mixed_bfloat16")
except ImportError:
    keras = None
    keras_nlp = None

from llm.local_llm import MODELS, ModelConfig

logger = logging.getLogger(__name__)

class KerasLLM:
    """
    Adapter for KerasNLP models to replace HuggingFaceModelAdapter.
    Targeted for TPU v5e usage via JAX backend.
    """
    def __init__(self, model_key: str = "phi-3", device: Any = None, block_llm: bool = False):
        if keras is None or keras_nlp is None:
             raise ImportError("Keras / KerasNLP not installed. Please install them to use type='keras_llm'.")
        
        if model_key not in MODELS:
             raise ValueError(f"Model {model_key} not found in registry.")

        self.model_key = model_key
        # KerasNLP handles devices internally via distribution API, 
        # but JAX usually defaults to all TPUs. 
        # We ignore explicit 'device' arg here as JAX/Keras auto-sharding is preferred.
        
        self.block_llm = block_llm
        self.model = None
        self.preprocessor = None
        
        # Map our registry keys to KerasNLP presets
        # This mapping might need expansion. 
        # Note: KerasHub names often differ from HF hub names.
        self._preset_map = {
            "phi-3": "phi3_mini_4k_instruct_en", # Approx equivalent
            "mistral-v0.3": "mistral_7b_en",
            "llama-3-8b": "llama3_8b_en", 
            "gemma": "gemma_2b_en" 
        }

    def _ensure_loaded(self):
        if self.block_llm:
            raise RuntimeError(f"Attempted to load KerasLLM {self.model_key} in BLOCKED mode.")

        if self.model is not None:
            return

        preset = self._preset_map.get(self.model_key)
        if not preset:
             # Fallback: try using the HF ID directly if possible, or raise
             # Keras 3 allows `from_preset("hf://...")`
             config = MODELS[self.model_key]
             preset = f"hf://{config['id']}"
             logger.info(f"Using HF path for Keras: {preset}")

        logger.info(f"Loading KerasNLP model from preset: {preset} ...")
        
        # Load Causal LM
        # KerasNLP CausalLM models include the tokenizer/preprocessor usually
        self.model = keras_nlp.models.CausalLM.from_preset(preset)
        
        # If we need the tokenizer separately:
        # self.preprocessor = self.model.preprocessor

    def call(
        self, 
        messages: list[dict[str, str]], 
        temperature: float = 0.2,
        max_new_tokens: int = 60,
        repetition_penalty: float = 1.0, # KerasNLP might not support this identical param in generate()
        do_sample: bool = True
    ) -> str:
        self._ensure_loaded()
        
        # 1. Format Prompt
        # KerasNLP models often expect raw text or have specific chat templates.
        # Ideally we use a chat template utility.
        # For simplicity, let's assume raw text generation or simple chat templating.
        # Phi-3 / Mistral have specific templates.
        
        # Naive reconstruction of prompt from messages (simple chat template)
        prompt = ""
        for m in messages:
            role = m["role"]
            content = m["content"]
            if role == "system":
                 prompt += f"<|system|>\n{content}<|end|>\n" 
            elif role == "user":
                 prompt += f"<|user|>\n{content}<|end|>\n"
            elif role == "assistant":
                 prompt += f"<|assistant|>\n{content}<|end|>\n"
        
        prompt += "<|assistant|>\n" # Trigger generation
        
        # 2. Generate
        # Keras generate() takes raw strings or tensors
        # It usually returns the FULL string (prompt + generation)
        output = self.model.generate(
            prompt, 
            max_length=max_new_tokens + 100, # Rough heuristic since max_length includes prompt
            stop_token_ids=None # We might need to handle EOS manually if not auto-handled
        )
        
        # 3. Strip prompt to get response
        if isinstance(output, str):
            response = output
        else:
             # If tensor/numpy
             response = output.numpy().decode('utf-8') if hasattr(output, 'numpy') else str(output)

        # Simple cleaning
        if response.startswith(prompt):
             response = response[len(prompt):]
             
        return response.strip()
