import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig # type: ignore
from typing import TypedDict, Any
import logging

# --------------------------------------------------------------------------------
# MODEL REGISTRY
# --------------------------------------------------------------------------------

class ModelConfig(TypedDict):
    id: str
    load_in_4bit: bool
    trust_remote_code: bool

MODELS: dict[str, ModelConfig] = {
    "phi-3": {
        "id": "microsoft/Phi-3-mini-128k-instruct",
        "load_in_4bit": False, 
        "trust_remote_code": False
    },
    "mistral-v0.3": {
        "id": "mistralai/Mistral-7B-Instruct-v0.3",
        "load_in_4bit": True,
        "trust_remote_code": False
    },
    "qwen-2-7b": {
        "id": "Qwen/Qwen2-7B-Instruct",
        "load_in_4bit": True,
        "trust_remote_code": False
    },
    "llama-3-8b": {
        "id": "meta-llama/Meta-Llama-3-8B-Instruct",
        "load_in_4bit": True,
        "trust_remote_code": False
    },
    "smollm2-1.7b": {
        "id": "HuggingFaceTB/SmolLM2-1.7B-Instruct",
        "load_in_4bit": False,
        "trust_remote_code": True
    },
    "smollm2-135m": {
        "id": "HuggingFaceTB/SmolLM2-135M-Instruct",
        "load_in_4bit": False,
        "trust_remote_code": True
    }
}

class HuggingFaceModelAdapter:
    def __init__(self, model_key: str = "phi-3", device: str = "cuda", block_llm: bool = False) -> None:
        if model_key not in MODELS:
            raise ValueError(f"Model {model_key} not found in registry. Available: {list(MODELS.keys())}")
        
        self.config: ModelConfig = MODELS[model_key]
        self.model_key: str = model_key
        self.device = device
        self.block_llm = block_llm
        self.tokenizer: Any = None
        self.model: Any = None
        
    def _ensure_loaded(self):
        if self.block_llm:
            raise RuntimeError(f"Attempted to load/use Local LLM {self.model_key} in BLOCKED mode. This likely means a cache miss occurred during offline analysis/dry-run.")

        if self.model is not None:
            return

        logging.info(f"Loading {self.model_key} ({self.config['id']})...")

        bnb_config: BitsAndBytesConfig | None = None
        if self.config["load_in_4bit"]:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
            )

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config["id"], 
            trust_remote_code=self.config["trust_remote_code"]
        )
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config["id"],
            quantization_config=bnb_config,
            dtype=torch.float16 if not self.config["load_in_4bit"] else None,
            device_map=self.device,
            trust_remote_code=self.config["trust_remote_code"]
        )

    def call(
        self, 
        messages: list[dict[str, str]], 
        temperature: float = 0.2,
        max_new_tokens: int = 60,
        repetition_penalty: float = 1.2,
        do_sample: bool = True
    ) -> str:
        """
        Generates text based on the list of messages (chat format).
        """
        self._ensure_loaded()
        
        input_ids: torch.Tensor = self.tokenizer.apply_chat_template(
            messages, 
            add_generation_prompt=True, 
            return_tensors="pt"
        ).to(self.device)

        # Dynamic parameter handling
        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "repetition_penalty": repetition_penalty,
            "pad_token_id": self.tokenizer.eos_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "do_sample": do_sample
        }

        # Temperature is only valid if we are sampling
        if do_sample:
            gen_kwargs["temperature"] = temperature
            gen_kwargs["top_p"] = 0.95 # Slight truncation of tail
        
        outputs: torch.Tensor = self.model.generate(
            input_ids,
            **gen_kwargs
        )

        generated_ids = outputs[0][input_ids.shape[1]:]
        response: str = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        return response.strip()
