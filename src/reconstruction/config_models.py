from pydantic import BaseModel, Field, field_validator
from typing import Literal, Union, Any

class BaseStrategyConfig(BaseModel):
    name: str

class LLMStrategyConfig(BaseStrategyConfig):
    type: Literal["llm"]
    llm: dict[str, Any]

class LocalLLMStrategyConfig(BaseStrategyConfig):
    type: Literal["local_llm"]
    model_key: str = "phi-3"
    prompt_dir: str = "iterative_cloze"
    temperature: float = 0.2
    repetition_penalty: float = 1.2
    max_new_tokens: int = 60
    length_multiplier: float = 2.5
    min_tokens: int = 20
    max_tokens_cap: int = 100
    
    # Catch-all for extra fields needed for IterativeReconstructionStrategy
    extra_params: dict[str, Any] = Field(default_factory=dict)
    
    @field_validator('extra_params')
    def validate_extra(cls, v):
        return v

class BaselineRepeatConfig(BaseStrategyConfig):
    type: Literal["baseline_repeat_last"]

# Union for polymorphic parsing
StrategyConfig = Union[LLMStrategyConfig, LocalLLMStrategyConfig, BaselineRepeatConfig]
