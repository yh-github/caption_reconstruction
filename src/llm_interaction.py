# src/llm_interaction.py
import os
import logging
import google.generativeai as genai
from google.generativeai.types import GenerationConfig
from tenacity import retry, wait_random_exponential, stop_after_attempt, retry_if_exception_type
import google.api_core.exceptions

def initialize_llm(config):
    """Configures the API key and initializes the GenerativeModel."""
    logging.info("Initializing Gemini model...")
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable not set.")
    genai.configure(api_key=api_key)
    
    model_name = config['llm']['model_name']
    return genai.GenerativeModel(model_name)

@retry(
    wait=wait_random_exponential(min=5, max=120),
    stop=stop_after_attempt(6)
    #Do we need this: ,retry=retry_if_exception_type(google.api_core.exceptions.ResourceExhausted)
)
def call_llm(model, prompt):
    """Calls the LLM API using its dedicated JSON mode."""
    logging.info("Calling LLM API in JSON mode...")
    
    json_output_config = GenerationConfig(response_mime_type="application/json")

    response = model.generate_content(prompt, generation_config=json_output_config)
    
    logging.info(f"LLM response received. Length: {len(response.text)} characters.")
    return response.text
