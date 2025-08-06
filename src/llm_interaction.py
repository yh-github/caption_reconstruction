import logging
from tenacity import retry, wait_random_exponential, stop_after_attempt, retry_if_exception_type

import google.api_core.exceptions
from google import genai
from google.genai.types import GenerateContentConfig

import diskcache
import hashlib
import base64
import json


logger = logging.getLogger(__name__)

def build_llm_manager(llm_config, llm_cache):
    logger.info(f"Initializing Gemini model {llm_config['model_name']}...")
    return LLM_Manager(
        model_name=llm_config['model_name'],
        seed=llm_config['seed'],
        temperature=llm_config['temperature'],
        system_instruction=llm_config.get('system_instructions'),
        llm_cache=llm_cache
    )

class LLM_Manager:

    def __init__(self, model_name, seed, temperature, system_instruction, llm_cache):
        self.model_name = model_name
        self.temperature = temperature
        self.system_instruction = system_instruction
        self.seed = seed

        self.llm = genai.Client()
        self.llm_config = GenerateContentConfig(
            system_instruction=self.system_instruction,
            temperature=self.temperature,
            # max_output_tokens=400, # top_k=2,# top_p=0.5,
            response_mime_type='application/json',
            # response_schema=
            seed=self.seed
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
    def _invoke_llm(self, prompt:str):
        return self.llm.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config=self.llm_config
        )

    def _call_retry(self, prompt:str) -> str|None:
        self.last_raw_response = None
        try:
            self.last_raw_response = self._invoke_llm(prompt)
        except Exception as e:
            logger.warning(f"INVOKE_LLM_EXCEPTION {e.__class__.__qualname__} {e=}")
            raise
        return self.last_raw_response.text

    def _cached_call(self, prompt:str) -> str|None:
        k = self.cache_key(prompt)
        if k in self.disk_cache:
            logger.debug(f'Cache hit: {k=}')
            return self.disk_cache[k]

        res = self._call_retry(prompt)
        if res:
            self.disk_cache[k] = res
        return res

    def call(self, prompt:str) -> str|None:
        # return self._call_retry(prompt)
        return self._cached_call(prompt)
