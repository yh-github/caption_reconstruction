import os
import logging
from tenacity import retry, wait_random_exponential, stop_after_attempt, retry_if_exception_type
import google.generativeai as genai
import google.api_core.exceptions
from google.generativeai.types import GenerationConfig
from data_models import ReconstructedCaption


def init_llm(api_key:str|None=None):
    if not api_key:
        api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable not set.")
    genai.configure(api_key=api_key)


def build_llm_manager(llm_config):
    logging.info(f"Initializing Gemini model {llm_config['model_name']}...")
    return LLM_Manager(
        model_name=llm_config['model_name'],
        temperature=llm_config['temperature'],
        system_instruction=llm_config['system_instructions']
    )

class LLM_Manager:

    def __init__(self, model_name, temperature, system_instruction):
        self.model_name = model_name
        self.temperature = temperature
        self.system_instruction = system_instruction

        generation_config = GenerationConfig(
            temperature=temperature,
            response_mime_type="application/json",
            response_schema=list[ReconstructedCaption]
        )

        self.llm = genai.GenerativeModel(
            model_name=model_name,
            generation_config=generation_config,
            system_instruction=system_instruction
        )

        self.last_raw_response = None

    @retry(
        wait=wait_random_exponential(multiplier=2, min=60, max=60*5),
        stop=stop_after_attempt(6),
        retry=retry_if_exception_type((
            google.api_core.exceptions.ResourceExhausted,  # For rate limits
            google.api_core.exceptions.ServerError  # For all 5xx server issues
        ))
    )
    def _invoke_llm(self, prompt):
        try:
            return self.llm.generate_content(prompt)
        except Exception as e:
            logging.warning(f"INVOKE_LLM_EXCEPTION {e}", exc_info=e)
            raise

    def _call_retry(self, prompt):
        self.last_raw_response = None
        self.last_raw_response = self._invoke_llm(prompt)
        return self.last_raw_response.text

    def call(self, prompt):
        return self._call_retry(prompt)
