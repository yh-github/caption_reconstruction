import base64
import hashlib
import json
import logging
from typing import Any

import diskcache
from google.api_core import retry

from google import genai
from google.genai.types import GenerateContentConfig, ThinkingConfig, GenerateContentResponse, Content, \
    ContentListUnion, SafetySetting, HarmCategory, HarmBlockThreshold
from pydantic import BaseModel
from data_models.schema import type_from_str, HashType
from common_utils.error_handling import ExceptionStr, raise_if

logger = logging.getLogger(__name__)

class LLM_Response(BaseModel):
    text:str|None
    thoughts:str|None = None
    raw_response: GenerateContentResponse | None = None

    def dump(self):
        return self.model_dump_json(exclude_none=True, exclude={"raw_response"})

    def should_cache(self):
        return self.text is not None

    @staticmethod
    def from_raw(raw_response: GenerateContentResponse):
        raise_if(raw_response is None, "LLM response is None")
        raise_if(not raw_response.candidates, "LLM response has no candidates")

        text:str|None = None
        thoughts:str|None = None

        if len(raw_response.candidates) > 1:
            logger.warning(f"Expected 1 candidate, got {len(raw_response.candidates)}")

        for part in raw_response.candidates[0].content.parts:
            if part.thought:
                raise_if(thoughts, "Thought already exists")
                thoughts = part.text
            elif part.text:
                raise_if(text, "Text already exists")
                text = part.text

        if text != raw_response.text:
            logger.warning(f"Text mismatch: {text=} {raw_response.text=}")

        return LLM_Response(text=text, thoughts=thoughts, raw_response=raw_response)

    @staticmethod
    def from_str(s:str):
        if s is None:
            return LLM_Response(text=None)
        if s.startswith("{"):
            return LLM_Response.model_validate_json(s)
        else:
            return LLM_Response(text=s)

class LLM_ResponseError(LLM_Response):
    text: None = None
    raw_response: GenerateContentResponse | None
    exception: ExceptionStr|None

    def dump(self):
        return self.model_dump_json(exclude_none=True)

    def should_cache(self):
        return False

class LLM_ResponseBlocked(LLM_ResponseError):
    exception: ExceptionStr|None = None

    def should_cache(self):
        return self.raw_response is not None

def is_perm_error(last_raw_response: GenerateContentResponse | None):
    if not last_raw_response or not last_raw_response.prompt_feedback:
        return False
    return last_raw_response.prompt_feedback.block_reason is not None

class LLM_Exception(Exception):
    def __init__(self, raw_response: GenerateContentResponse | None, message:str):
        self.raw_response = raw_response
        self.message = message
        super().__init__(self.message)

    def __str__(self):
        return f"{self.message=} {self.raw_response=}"

class LLM_Manager:

    def __init__(self,
        llm_client:genai.Client,
        model_name:str,
        llm_config:GenerateContentConfig,
        llm_cache:diskcache.Cache|dict[str,str],
        base_cache_key:HashType
    ):
        self._llm_client = llm_client
        self._model_name = model_name
        self._llm_config = llm_config
        self._disk_cache = llm_cache
        self._base_cache_key = base_cache_key
        # self.last_raw_response: GenerateContentResponse | None = None

    def _cache_key(self, prompt:str) -> str:
        sha = self._base_cache_key.copy()
        sha.update(prompt.encode())
        return base64.urlsafe_b64encode(sha.digest()).decode('utf-8')


    @staticmethod
    def log_retry(exception:Exception, try_num:int):
        logger.warning(f"log_retry {try_num=}, transient={retry.if_transient_error(exception)}, {type(exception)} -- {exception}")

    @retry.Retry(
        predicate=retry.if_transient_error,  # Retry on transient API errors (e.g., 500, 503)
        initial=5.0,  # Initial delay in seconds
        maximum=60.0,  # Maximum delay in seconds
        multiplier=2.0,  # Multiplier for exponential backoff
        timeout=600.0,  # Total timeout for all retries in seconds
    )
    def _invoke_llm(self, prompt:ContentListUnion) -> GenerateContentResponse:
        try:
            self._try_num += 1
            return self._llm_client.models.generate_content(
                model=self._model_name,
                contents=prompt,
                config=self._llm_config
            )
        except Exception as ex:
            self.log_retry(ex, self._try_num)
            raise

    def _call_retry(self, prompt:ContentListUnion) -> LLM_Response:
        raw_response = None
        self._try_num = 0
        try:
            raw_response = self._invoke_llm(prompt)
            if is_perm_error(raw_response):
                logger.info(f"LLM PERM ERROR without Exception")
                return LLM_ResponseBlocked(raw_response=raw_response)
            return LLM_Response.from_raw(raw_response)
        except Exception as e:
            logger.warning(f"INVOKE_LLM_EXCEPTION {e.__class__.__qualname__} {e=}")
            if is_perm_error(raw_response):
                logger.info(f"LLM PERM ERROR with Exception")
                return LLM_ResponseBlocked(raw_response=raw_response, exception=ExceptionStr(e))
            return LLM_ResponseError(raw_response=raw_response, exception=ExceptionStr(e))
            # raise LLM_Exception(raw_response=raw_response, message=f"{type(e)}: {e}") from e

    def call(self, prompt:str|Content) -> LLM_Response:
        k:str|None = None
        if isinstance(prompt, str):
            k = self._cache_key(prompt)
        elif isinstance(prompt, Content):
            k = self._cache_key(prompt.model_dump_json(exclude_none=True, fallback=str))
        assert k, f"{type(prompt) = }"
        if k in self._disk_cache:
            logger.debug(f'Cache hit: {k=}')
            return LLM_Response.from_str(self._disk_cache[k])
        res = self._call_retry(prompt)
        if res.should_cache():
            self.log_token_usage(k, res)
            self._disk_cache[k] = res.dump()

        if self._llm_config.thinking_config and res.text and not res.thoughts:
            logger.warning(f"No thoughts in LLM response: {res.text=}")

        return res

    def log_token_usage(self, cache_key:str, response: LLM_Response) -> None:
        if response.raw_response and response.raw_response.usage_metadata:
            metadata_str = response.raw_response.usage_metadata.model_dump_json(exclude_none=True, fallback=str)
            logger.info(f"{cache_key=} metadata={metadata_str}")


class LLM_Manager_Builder:

    def __init__(self, llm_client:genai.Client, llm_cache:diskcache.Cache|dict[str,str]):
        self.llm_client = llm_client
        self.llm_cache = llm_cache

    def from_config(self, llm_config:dict[str, Any]) -> LLM_Manager:
        logger.info(f"Initializing Gemini model {llm_config['model_name']}...")

        model_name:str = llm_config['model_name']
        response_schema = llm_config.get('response_schema')
        llm_config:GenerateContentConfig = GenerateContentConfig(
            system_instruction=llm_config.get('system_instructions'),
            temperature=llm_config['temperature'],
            # max_output_tokens=400, # top_k=2,# top_p=0.5,
            response_mime_type=self.response_mime_type(response_schema),
            response_schema=self.config_response_schema(response_schema),
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
                category=category,
                threshold=HarmBlockThreshold.BLOCK_NONE
            ) for category in [
                HarmCategory.HARM_CATEGORY_HARASSMENT,
                HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                HarmCategory.HARM_CATEGORY_CIVIC_INTEGRITY
            ]
        ]

    @staticmethod
    def build_thinking_config(thinking_budget:int):
        return None if thinking_budget==0 else ThinkingConfig(
            thinking_budget=thinking_budget,
            include_thoughts=True
        )

    @staticmethod
    def config_response_schema(schema: str | None):
        if not schema or schema=='text/plain':
            return None
        if schema not in type_from_str:
            logger.warning(f"Unknown response schema: {schema}")
            return None
        return type_from_str[schema]

    @staticmethod
    def response_mime_type(response_schema: str | None) -> str:
        if response_schema and response_schema=='text/plain':
            return 'text/plain'
        return 'application/json'
