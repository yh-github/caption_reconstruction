import base64
import hashlib
import json
import logging
from typing import Any

import diskcache
import google.api_core.exceptions
from google import genai
from google.genai.types import GenerateContentConfig, ThinkingConfig, GenerateContentResponse, Content, \
    ContentListUnion, SafetySetting, HarmCategory, HarmBlockThreshold
from pydantic import BaseModel
from tenacity import retry, wait_random_exponential, stop_after_attempt, retry_if_exception_type

from data_models.schema import type_from_str, HashType

logger = logging.getLogger(__name__)

class LLM_Response(BaseModel):
    text:str|None
    thoughts:str|None = None

    def should_cache(self):
        return self.text is not None

    @staticmethod
    def from_raw(raw_response: GenerateContentResponse):
        if raw_response is None:
            logger.error("LLM response is None")
            return LLM_Response(text=None)

        if not raw_response.candidates:
            logger.error("LLM response has no candidates")
            return LLM_Response(text=None)

        text = None
        thoughts = None

        if len(raw_response.candidates) > 1:
            logger.warning(f"Expected 1 candidate, got {len(raw_response.candidates)}")

        for part in raw_response.candidates[0].content.parts:
            if part.thought:
                if thoughts is not None:
                    raise Exception("Thought already exists")
                thoughts = part.text
            elif part.text:
                if text is not None:
                    raise Exception("Text already exists")
                text = part.text

        if text != raw_response.text:
            logger.warning(f"Text mismatch: {text=} {raw_response.text=}")

        return LLM_Response(text=text, thoughts=thoughts)

    @staticmethod
    def from_str(s:str):
        if s is None:
            return LLM_Response(text=None)
        if s.startswith("{"):
            return LLM_Response.model_validate_json(s)
        else:
            return LLM_Response(text=s)

class LLM_ResponseBlocked(LLM_Response):
    text: None = None
    raw_response: GenerateContentResponse | None

    def should_cache(self):
        return self.raw_response is not None


def is_perm_error(last_raw_response: GenerateContentResponse | None):
    if not last_raw_response:
        return False
    return last_raw_response.prompt_feedback.block_reason is not None

class LLM_Manager:

    def __init__(self,
        llm_client:genai.Client,
        model_name:str,
        llm_config:GenerateContentConfig,
        llm_cache:diskcache.Cache|dict[str,str],
        base_cache_key:HashType
    ):
        self.llm_client = llm_client
        self.model_name = model_name
        self.llm_config = llm_config
        self.disk_cache = llm_cache
        self.base_cache_key = base_cache_key
        self.last_raw_response: GenerateContentResponse | None = None

    def cache_key(self, prompt:str):
        sha = self.base_cache_key.copy()
        sha.update(prompt.encode())
        return base64.urlsafe_b64encode(sha.digest()).decode('utf-8')

    @retry(
        wait=wait_random_exponential(multiplier=2, min=60, max=60*5),
        stop=stop_after_attempt(6),
        retry=retry_if_exception_type((
            google.api_core.exceptions.ResourceExhausted,  # For rate limits
            google.api_core.exceptions.ServerError  # For all 5xx server issues
        ))
    )
    def _invoke_llm(self, prompt:ContentListUnion) -> GenerateContentResponse:
        return self.llm_client.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config=self.llm_config
        )

    def _call_retry(self, prompt:ContentListUnion) -> LLM_Response:
        self.last_raw_response = None
        try:
            self.last_raw_response = self._invoke_llm(prompt)
        except Exception as e:
            logger.warning(f"INVOKE_LLM_EXCEPTION {e.__class__.__qualname__} {e=}")
            raise
        return LLM_Response.from_raw(self.last_raw_response)

    def call(self, prompt:str|Content) -> LLM_Response:
        if isinstance(prompt, str):
            k = self.cache_key(prompt)
        elif isinstance(prompt, Content):
            k = self.cache_key(prompt.model_dump_json(exclude_none=True, fallback=str))
        assert k
        if k in self.disk_cache:
            logger.debug(f'Cache hit: {k=}')
            return LLM_Response.from_str(self.disk_cache[k])
        res = self._call_retry(prompt)
        if not res.text and is_perm_error(self.last_raw_response):
            res = LLM_ResponseBlocked(raw_response=self.last_raw_response)
        if res.should_cache():
            self.disk_cache[k] = res.model_dump_json(exclude_none=True)

        if self.llm_config.thinking_config and not res.thoughts and res.text:
            logger.warning(f"No thoughts in LLM response: {res.text=}")

        return res

class LLM_Manager_Builder:

    def __init__(self, llm_client:genai.Client, llm_cache:diskcache.Cache|dict[str,str]):
        self.llm_client = llm_client
        self.llm_cache = llm_cache

    def from_config(self, llm_config:dict[str, Any]) -> LLM_Manager:
        logger.info(f"Initializing Gemini model {llm_config['model_name']}...")

        model_name:str = llm_config['model_name']
        llm_config:GenerateContentConfig = GenerateContentConfig(
            system_instruction=llm_config.get('system_instructions'),
            temperature=llm_config['temperature'],
            # max_output_tokens=400, # top_k=2,# top_p=0.5,
            response_mime_type='application/json',
            response_schema=self.config_response_schema(llm_config.get('response_schema')),
            seed=llm_config['seed'],
            thinking_config=self.build_thinking_config(llm_config.get('thought_budget', 0))
        )

        base_cache_key:HashType = hashlib.sha256(json.dumps(obj={
            "model_name": model_name,
            "llm_config": llm_config.model_dump_json(exclude_none=True, fallback=str)
        }, sort_keys=True).encode())

        self.add_transient_config(llm_config)

        return LLM_Manager(self.llm_client, model_name, llm_config, self.llm_cache, base_cache_key)

    @staticmethod
    def add_transient_config(llm_config: GenerateContentConfig):
        """
        These are the settings that do not affect the cache key.
        Refrain from using this to change parameters that might affect the content of the result.
        Use only for parameters that are all or nothing, like (not) blocking content.
        """
        llm_config.safety_settings = [
            SafetySetting(
                category=HarmCategory.HARM_CATEGORY_HARASSMENT,
                threshold=HarmBlockThreshold.BLOCK_NONE,
            ),
            SafetySetting(
                category=HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                threshold=HarmBlockThreshold.BLOCK_NONE,
            ),
            SafetySetting(
                category=HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                threshold=HarmBlockThreshold.BLOCK_NONE,
            ),
            SafetySetting(
                category=HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                threshold=HarmBlockThreshold.BLOCK_NONE,
            )
        ]

    @staticmethod
    def config_response_schema(schema: str | None):
        if not schema:
            return None
        if schema not in type_from_str:
            logger.warning(f"Unknown response schema: {schema}")
            return None
        return type_from_str[schema]

    @staticmethod
    def build_thinking_config(thinking_budget:int):
        return None if thinking_budget==0 else ThinkingConfig(
            thinking_budget=thinking_budget,
            include_thoughts=True
        )
