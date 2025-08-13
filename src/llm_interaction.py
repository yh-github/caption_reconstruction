import logging

from tenacity import retry, wait_random_exponential, stop_after_attempt, retry_if_exception_type

import google.api_core.exceptions
from google import genai
from google.genai.types import GenerateContentConfig, ThinkingConfig, GenerateContentResponse, Content, ContentListUnion

import diskcache
import hashlib
import base64
import json
from pydantic import BaseModel


logger = logging.getLogger(__name__)

def build_llm_manager(llm_config, llm_cache):
    logger.info(f"Initializing Gemini model {llm_config['model_name']}...")
    return LLM_Manager(
        model_name=llm_config['model_name'],
        seed=llm_config['seed'],
        temperature=llm_config['temperature'],
        system_instruction=llm_config.get('system_instructions'),
        thought_budget=llm_config.get('thought_budget', 0),
        llm_cache=llm_cache
    )

class LLM_Response(BaseModel):
    text:str|None
    thoughts:str|None = None

    @staticmethod
    def from_raw(raw_response: GenerateContentResponse):
        if raw_response is None:
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


class LLM_Manager:

    def __init__(self, model_name, seed, temperature, system_instruction, thought_budget:int, llm_cache, response_schema=None):
        self.model_name = model_name
        self.temperature = temperature
        self.system_instruction = system_instruction
        self.seed = seed

        self.llm = genai.Client()

        thinking_config = None if thought_budget==0 else ThinkingConfig(
            thinking_budget=thought_budget,
            include_thoughts=True
        )

        self.llm_config = GenerateContentConfig(
            system_instruction=self.system_instruction,
            temperature=self.temperature,
            # max_output_tokens=400, # top_k=2,# top_p=0.5,
            response_mime_type='application/json',
            response_schema=response_schema,
            seed=self.seed,
            thinking_config=thinking_config
        )

        self.disk_cache:diskcache.Cache = llm_cache
        # noinspection PyTypeChecker
        self.base_cache_key = hashlib.sha256(json.dumps(obj={
            "model_name": model_name,
            "llm_config": self.llm_config.model_dump_json(exclude_none=True, fallback=str)
        }, sort_keys=True).encode())

        self.last_raw_response = None
        # self.cached_call = self.disk_cache.cache(self._call_retry, ignore=['self'])

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
        return self.llm.models.generate_content(
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
        if res.text:
            self.disk_cache[k] = res.model_dump_json(exclude_none=True)

        if self.llm_config.thinking_config and not res.thoughts:
            logger.warning(f"No thoughts in LLM response: {res.text=}")

        return res


