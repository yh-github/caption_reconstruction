import os
import logging
from tenacity import retry, wait_random_exponential, stop_after_attempt, retry_if_exception_type
import google.generativeai as genai
import google.api_core.exceptions
from google.generativeai.types import GenerationConfig

import diskcache
import hashlib
import base64
import json

def init_llm(api_key:str|None=None):
    if not api_key:
        api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable not set.")
    genai.configure(api_key=api_key)


def build_llm_manager(llm_config, llm_cache):
    logging.info(f"Initializing Gemini model {llm_config['model_name']}...")
    return LLM_Manager(
        model_name=llm_config['model_name'],
        temperature=llm_config['temperature'],
        system_instruction=llm_config.get('system_instructions'),
        llm_cache=llm_cache
    )

class LLM_Manager:

    def __init__(self, model_name, temperature, system_instruction, llm_cache):
        self.model_name = model_name
        self.temperature = temperature
        self.system_instruction = system_instruction

        generation_config = GenerationConfig(
            temperature=temperature,
            response_mime_type="application/json"
            #,response_schema=list[ReconstructedCaption] # doesn't work
        )

        self.llm = genai.GenerativeModel(
            model_name=model_name,
            generation_config=generation_config,
            system_instruction=system_instruction
        )
        self.disk_cache:diskcache.Cache = llm_cache
        self.base_cache_key = hashlib.sha256(json.dumps(obj={
            "model_name": model_name,
            "temperature": temperature,
            "system_instruction": system_instruction
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
        return self.llm.generate_content(prompt)

    def _call_retry(self, prompt:str) -> str|None:
        self.last_raw_response = None
        try:
            self.last_raw_response = self._invoke_llm(prompt)
        except Exception as e:
            logging.warning(f"INVOKE_LLM_EXCEPTION {e=} for {prompt=}", exc_info=e)
            raise
        return self.last_raw_response.text

    def _cached_call(self, prompt:str) -> str|None:
        k = self.cache_key(prompt)
        if k in self.disk_cache:
            logging.debug(f'Cache hit: {k=}')
            return self.disk_cache[k]

        res = self._call_retry(prompt)
        if res:
            self.disk_cache[k] = res
        return res

    def call(self, prompt:str) -> str|None:
        # return self._call_retry(prompt)
        return self._cached_call(prompt)
