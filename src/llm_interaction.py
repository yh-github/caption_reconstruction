# src/llm_interaction.py
import os
import logging
import google.generativeai as genai
from joblib import Memory
from google.generativeai.types import GenerationConfig
from tenacity import retry, wait_random_exponential, stop_after_attempt, retry_if_exception_type
import google.api_core.exceptions

def build_llm_manager(config):
    logging.info("Initializing Gemini model...")
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable not set.")
    genai.configure(api_key=api_key)
    llm_config = config['llm']
    return LLM_Manager(
        base_cache_dir=config['paths']['joblib_cache'].removesuffix('/'),
        model_name=llm_config['model_name'],
        temperature=llm_config['temperature'],
        system_prompt=None # TODO system_prompt
    )

class LLM_Manager:

    def __init__(self, base_cache_dir, model_name, temperature, system_prompt=None):
        self.model_name = model_name
        self.temperature = temperature
        self.system_prompt = system_prompt

        generation_config = GenerationConfig(
            temperature=temperature,
            response_mime_type="application/json"
        )

        self.llm = genai.GenerativeModel(
            model_name="gemini-2.0-flash-exp",
            generation_config=generation_config,
            system_instruction=system_prompt
        )
        self.cache_path = f"{base_cache_dir}/{model_name}/t{temperature}"
        self.disk_cache = Memory(self.cache_path, compress=3, verbose=0)
        self.last_raw_response = None
        self.cached_call = self.disk_cache.cache(self._call_retry, ignore=['self'])

    @retry(
        wait=wait_random_exponential(min=5, max=120),
        stop=stop_after_attempt(6),
        retry=retry_if_exception_type(google.api_core.exceptions.ResourceExhausted)
    )
    def _invoke_llm(self, prompt):
        return self.llm.generate_content(prompt)

    def _call_retry(self, prompt):
        self.last_raw_response = None
        self.last_raw_response = self._invoke_llm(prompt)
        return self.last_raw_response.text

    def call(self, prompt):
        return self.cached_call(prompt)


