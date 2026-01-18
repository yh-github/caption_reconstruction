import logging
from abc import ABC, abstractmethod
from typing import Any
from pathlib import Path
import re

from google import genai
from pydantic import BaseModel
from pydantic_core import PydanticSerializationError

from data_models.captions_only import CaptionedClip, ReconstructedCaptions
from data_models.captions_only import CaptionedVideo
from llm.llm_interaction import LLM_Manager_Builder, LLM_Response, LLM_ResponseError
from llm.llm_interaction import LLM_Manager_Builder, LLM_Response, LLM_ResponseError
from llm.local_llm import HuggingFaceModelAdapter
try:
    from llm.keras_llm import KerasLLM
except ImportError:
    KerasLLM = None
from llm.embedder import CacheMissError
from llm.parsers import parse_llm_response
from llm.prompting import PromptBuilder, JSONPromptBuilder, ClozePromptBuilder
from common_utils import device_setup
from common_utils.error_handling import UserFacingError, ExceptionStr


class Reconstructed(BaseModel):
    video_id: str
    reconstructed_captions: dict[int, str]
    debug_data: dict[str, Any]|None = None
    skip_reason: str|None = None
    metrics: dict|None = None

    def align(self, orig_clips: list[CaptionedClip]) -> tuple[list[str], list[str]]:
        """
        Helper method to extract reference and candidate sentences.
        """
        references = []
        candidates = []

        for i, c in self.reconstructed_captions.items():
            assert i == orig_clips[i].index
            candidates.append(c)
            references.append(orig_clips[i].caption)

        return candidates, references

    def skip(self, reason: str):
        if self.skip_reason:
            self.skip_reason += f" | {reason}"
        else:
            self.skip_reason = reason
        return self

    def with_metrics(self, metrics: dict):
        self.metrics = metrics
        return self

    def json_str(self):
        try:
            return self.model_dump_json(exclude_none=True)
        except PydanticSerializationError as e:
            print(f"{e!r} {e!s}")
            raise e



class ReconstructionStrategy(ABC):
    """An abstract base class for all reconstruction methods."""
    def __init__(self, name: str):
        self.name = name

    def __repr__(self):
        return self.name

    @abstractmethod
    def reconstruct(self, masked_video: CaptionedVideo) -> Reconstructed:
        """Takes a masked CaptionedVideo and returns a reconstructed one."""
        pass

    @staticmethod
    def create_error_result(video_id: str, error_message: str, extra_debug_data: dict = None) -> Reconstructed:
        """Create a Reconstructed result for error cases."""
        debug_data = {
            "error": error_message
        }

        if extra_debug_data:
            debug_data.update(extra_debug_data)

        return Reconstructed(
            video_id=video_id,
            reconstructed_captions={},
            debug_data=debug_data,
            skip_reason="error"
        )


class BaselineRepeatStrategy(ReconstructionStrategy):
    """The strategy for using the 'repeat last known' baseline."""
    def __init__(self):
        super().__init__('BaselineRepeatStrategy')

    def reconstruct(self, masked_video: CaptionedVideo) -> Reconstructed:
        """
        Fills masked clips by repeating the data from the last known clip.
        If initial clips are masked, it back-fills them with the first valid data.
        """
        _clips = masked_video.clips

        # First Pass: Find the first available data payload
        first_valid_caption = None
        for clip in _clips:
            if clip.caption is not None:
                first_valid_caption = clip.caption
                break

        # Second Pass: Reconstruct the captions
        reconstructed_captions = {}
        last_known_caption = first_valid_caption

        for clip in _clips:
            if clip.caption is not None:
                last_known_caption = clip.caption
            else: # clip is MASKED
                reconstructed_captions[clip.index]=last_known_caption
        try:
            return Reconstructed(video_id=masked_video.video_id, reconstructed_captions=reconstructed_captions)
        except Exception:
            logging.error(f"{masked_video=} {reconstructed_captions=}")
            raise


class LLMStrategy(ReconstructionStrategy):
    """The strategy for using an LLM for reconstruction."""

    # Error message constants
    EMPTY_RESPONSE_ERROR = "LLM error - llm_response_text empty"
    PARSING_ERROR = "LLM error - failed parsing"
    DUPLICATE_INDICES_ERROR = "LLM error - duplicate indices found"

    def __init__(self, name: str, llm_model, prompt_builder: PromptBuilder):
        super().__init__(name)
        self.llm_model = llm_model
        self.prompt_builder: PromptBuilder = prompt_builder

    def reconstruct(self, masked_video: CaptionedVideo) -> Reconstructed:
        def err(message: str, extra:dict=None):
            return self.create_error_result(
                video_id=masked_video.video_id,
                error_message=message,
                extra_debug_data=extra
            )

        llm_res = None
        try:
            llm_res = self._get_llm_response(masked_video)
            if not llm_res.text:
                if isinstance(llm_res, LLM_ResponseError):
                    return err(
                        llm_res.__class__.__qualname__,
                        {
                            "exception": llm_res.exception,
                            "raw_response": llm_res.raw_response
                        })
                return err(self.EMPTY_RESPONSE_ERROR)

            recon_caps, dups = self._parse_and_validate_response(llm_res.text)
            if not recon_caps:
                return err(self.PARSING_ERROR)

            if dups:
                return err(
                    self.DUPLICATE_INDICES_ERROR,
                    {"llm_response_text": llm_res.text, "dups": dups}
                )

            return self._process_reconstruction_results(masked_video, recon_caps, llm_res.text)

        except CacheMissError as e:
            logging.warning(f"Cache miss for {masked_video.video_id}: {e}")
            return err("cache_miss", {"exception": str(e)})
        except Exception as e:
            logging.error(f"{e} for {masked_video.video_id=}", exc_info=True)
            return err(
                "exception",
                {
                    "raw_response": llm_res.raw_response if llm_res else None,
                    "llm_response_text": llm_res.text if llm_res else None,
                    "exception": ExceptionStr(e)
                }
            )

    def _get_llm_response(self, masked_video: CaptionedVideo) -> LLM_Response:
        """Generate prompt and get response from LLM."""
        prompt = self.prompt_builder.build_prompt(masked_video)
        logging.debug(f"video_id={masked_video.video_id} {prompt=}")
        return self.llm_model.call(prompt)

    @staticmethod
    def _parse_and_validate_response(llm_response_text: str) -> tuple[dict[int, str], dict[int, int]]:
        """Parse LLM response and convert to dictionary format."""
        reconstructed_video = parse_llm_response(model=ReconstructedCaptions, response_text=llm_response_text)
        if not reconstructed_video:
            return {}, {}
        return reconstructed_video.to_dict()

    def _process_reconstruction_results(self, masked_video: CaptionedVideo, recon_caps: dict[int, str],
                                        llm_response_text: str) -> Reconstructed:
        """Process reconstruction results and categorize clips."""
        result = self._categorize_clips(masked_video, recon_caps)

        debug_data = None
        if result.failed or result.changed_unmasked:
            debug_data = {
                "ok": result.ok,
                "failed": result.failed,
                "changed_unmasked": result.changed_unmasked,
                "llm_response_text": llm_response_text
            }

        return Reconstructed(
            video_id=masked_video.video_id,
            reconstructed_captions=result.reconstructed_dict,
            debug_data=debug_data
        )

    class CategorizationResult(BaseModel):
        ok: list[int] = []
        failed: list[int] = []
        changed_unmasked: list[int] = []
        reconstructed_dict: dict[int, str] = {}

    @staticmethod
    def _categorize_clips(masked_video: CaptionedVideo, recon_caps: dict[int, str]) -> "LLMStrategy.CategorizationResult":
        """Categorize clips into successful, failed, and changed categories."""
        ok = []
        failed = []
        changed_unmasked = []
        reconstructed_dict: dict[int, str] = {}

        for c in masked_video.clips:
            if c.is_masked():
                if new_cap := recon_caps.get(c.index):
                    ok.append(c.index)
                    reconstructed_dict[c.index] = new_cap
                else:
                    failed.append(c.index)
                    reconstructed_dict[c.index] = ""
            elif c.index in recon_caps and c.caption != recon_caps.get(c.index):
                changed_unmasked.append(c.index)

        return LLMStrategy.CategorizationResult(
            ok=ok,
            failed=failed,
            changed_unmasked=changed_unmasked,
            reconstructed_dict=reconstructed_dict
        )

class IterativeReconstructionStrategy(ReconstructionStrategy):
    """
    Strategy using an iterative/autoregressive approach to fill gaps.
    Commonly used with smaller local models via HuggingFaceModelAdapter.
    """
    def __init__(self, name: str, model_adapter: HuggingFaceModelAdapter, prompt_builder: ClozePromptBuilder, config: dict):
        super().__init__(name)
        self.model_adapter = model_adapter
        self.prompt_builder = prompt_builder
        self.config = config
        self.temperature = config.get("temperature", 0.2)
        self.repetition_penalty = config.get("repetition_penalty", 1.2)
        self.default_max_new_tokens = config.get("max_new_tokens", 60) 

    def _clean_output(self, text: str) -> str:
        """
        Cleans the model output by removing quotes, timestamps, and whitespace.
        """
        # Remove timestamps like [00:00] or [00:00:00]
        text = re.sub(r'\[\d{2}:\d{2}(?::\d{2})?\]', '', text)
        
        # Remove surrounding quotes
        text = text.strip().strip('"\'')
        
        # Remove double spaces
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text

    def reconstruct(self, masked_video: CaptionedVideo) -> Reconstructed:
        reconstructed_captions = {}
        # We work on a copy of clips to update context as we go
        working_clips = [c.model_copy() for c in masked_video.clips]

        for i, clip in enumerate(working_clips):
            if clip.is_masked():
                # 1. Build Context
                # Context Before: All prior captions (or a window)
                # Context After: All future captions (or a window)
                WINDOW_SIZE = 500 # Effectively unlimited context
                
                def format_ts(ts) -> str:
                    # timestamps are floats (seconds). Convert to [SS] or [MM:SS]
                    m, s = divmod(ts.start, 60)
                    return f"[{int(m):02d}:{int(s):02d}]"

                start_before = max(0, i - WINDOW_SIZE)
                context_before_clips = working_clips[start_before:i]
                context_before_str = "\n".join(
                    f"{format_ts(c.timestamp)} {c.caption}" for c in context_before_clips if c.caption
                )

                end_after = min(len(working_clips), i + 1 + WINDOW_SIZE)
                context_after_clips = working_clips[i+1:end_after]
                context_after_str = "\n".join(
                    f"{format_ts(c.timestamp)} {c.caption}" for c in context_after_clips if c.caption
                )
                
                # Context is passed as-is. Builder handles conditional prompts for empty context.
                if not context_before_str.strip():
                    context_before_end_hint = "START"
                else:
                    context_before_end_hint = context_before_str.split()[-1]

                if not context_after_str.strip():
                    context_after_start_hint = "END"
                else:
                    context_after_start_hint = context_after_str.split()[0]
                
                target_timestamp = format_ts(clip.timestamp)
                target_duration = f"{clip.timestamp.duration:.1f}"

                
                prompt_context = {
                    "TARGET_TIMESTAMP": target_timestamp,
                    "TARGET_DURATION": target_duration,
                    "CONTEXT_BEFORE": context_before_str,
                    "CONTEXT_AFTER": context_after_str, 
                    "CONTEXT_BEFORE_END_HINT": context_before_end_hint,
                    "CONTEXT_AFTER_START_HINT": context_after_start_hint
                }
                
                # Dynamic max_new_tokens calculation
                lengths = [len(str(c.caption).split()) for c in context_before_clips + context_after_clips if c.caption]
                if lengths:
                    avg_len = sum(lengths) / len(lengths)
                    computed_max_tokens = int(avg_len * 2.5) 
                    max_new_tokens = max(20, min(computed_max_tokens, 100)) 
                else:
                    max_new_tokens = self.default_max_new_tokens

                try:
                    messages = self.prompt_builder.build_prompt(prompt_context)
                    
                    generated_text = self.model_adapter.call(
                        messages=messages,
                        temperature=self.temperature,
                        repetition_penalty=self.repetition_penalty,
                        max_new_tokens=max_new_tokens
                    )
                    
                    generated_text = self._clean_output(generated_text)
                    reconstructed_captions[clip.index] = generated_text
                    
                    # Update the working clip so it serves as context for subsequent gaps
                    working_clips[i] = clip.model_copy(update={'caption': generated_text})

                except Exception as e:
                    logging.error(f"Local LLM failed for {masked_video.video_id} index {i}: {e}")
                    reconstructed_captions[clip.index] = "" # Failed

        return Reconstructed(
            video_id=masked_video.video_id,
            reconstructed_captions=reconstructed_captions
        )

class TextReconstructionStrategyBuilder:
    """
    A builder class responsible for creating reconstruction strategy objects.
    """
    def __init__(self, llm_cache, master_seed:int, llm_client:genai.Client, block_llm: bool = False):
        self.master_seed = master_seed
        self.block_llm = block_llm
        self.llm_manager_builder = LLM_Manager_Builder(llm_client, llm_cache)
        self._local_model_cache: dict[str, HuggingFaceModelAdapter | Any] = {}       
        self.prompts_dir = Path("prompts").resolve()

    def get_strategy(self, strategy_config: dict) -> ReconstructionStrategy:
        """
        Builds and returns a specific strategy instance based on the config.
        """
        from pydantic import TypeAdapter
        from .config_models import StrategyConfig, LLMStrategyConfig, LocalLLMStrategyConfig, BaselineRepeatConfig

        # Validate using Pydantic
        try:
            config_model = TypeAdapter(StrategyConfig).validate_python(strategy_config)
        except Exception as e:
           # Fallback for now or raise detailed error?
           # Re-raising nicely
           print(f"DEBUG: Invalid config received: {strategy_config}")
           raise UserFacingError(f"Invalid strategy configuration: {e}")

        if isinstance(config_model, LLMStrategyConfig):
            llm_conf = config_model.llm.copy()
            llm_conf['seed'] = llm_conf.get('seed',0) + self.master_seed
            return LLMStrategy(
                name=config_model.name,
                llm_model=self.llm_manager_builder.from_config(llm_conf),
                prompt_builder=JSONPromptBuilder.from_config(llm_conf)
            )

        elif isinstance(config_model, LocalLLMStrategyConfig):
            model_key = config_model.model_key
            
            # System-level decision for backend
            backend = device_setup.get_llm_backend()
            cache_key = f"{model_key}_{backend}"
            
            if cache_key not in self._local_model_cache:
                if backend == "keras_llm":
                    if KerasLLM is None:
                        raise UserFacingError("KerasLLM backend selected but dependencies not found.")
                    logging.info(f"Initializing KerasLLM for {model_key}...")
                    self._local_model_cache[cache_key] = KerasLLM(model_key=model_key, block_llm=self.block_llm)
                else:
                    self._local_model_cache[cache_key] = HuggingFaceModelAdapter(model_key=model_key, block_llm=self.block_llm)
            
            prompt_path = self.prompts_dir / config_model.prompt_dir
            if prompt_path.is_dir():
                prompt_builder = ClozePromptBuilder.from_directory(prompt_path)
            else:
                raise UserFacingError(f"Prompt directory '{prompt_path}' does not exist.")

            # Convert back to dict for the strategy constructor if it expects a dict
            # or update Strategy to take the model?
            # Existing Strategy expects a dict 'config'
            # We can dump model to dict
            
            return IterativeReconstructionStrategy(
                name=config_model.name,
                model_adapter=self._local_model_cache[cache_key],
                prompt_builder=prompt_builder,
                config=config_model.model_dump()
            )

        elif isinstance(config_model, BaselineRepeatConfig):
            return BaselineRepeatStrategy()

        else:
            raise NotImplementedError(f"Strategy type '{config_model.type}' is not implemented.")


class BatchGridSearchStrategy(ReconstructionStrategy):
    """
    Simulates running N independent experiments (search grid) in parallel for one video.
    Each configuration in the grid maintains its own context/state.
    """
    def __init__(self, name: str, model_adapter: HuggingFaceModelAdapter, prompt_builder: ClozePromptBuilder, configs: list[dict]):
        super().__init__(name)
        self.model_adapter = model_adapter
        self.prompt_builder = prompt_builder
        self.configs = configs
        
        # Extract vectorized params
        # Fallback defaults if missing in config
        self.temperatures = [c.get("temperature", 0.2) for c in configs]
        self.penalties = [c.get("repetition_penalty", 1.2) for c in configs]
        
        # We assume max_new_tokens logic is similar or we pick the max requested?
        # For simplicity, we'll calculate per-batch-item but we must pass a single scalar to generate() 
        # as the 'cutoff'. The model adapter can stop earlier if EOS is generated, 
        # OR we pick a reasonable safe max.
        self.default_max_new_tokens = configs[0].get("max_new_tokens", 60)

    def reconstruct(self, masked_video: CaptionedVideo, active_indices: list[int] | None = None) -> list[Reconstructed]:
        """
        Returns a LIST of Reconstructed objects, one for each configuration.
        
        Args:
            masked_video: The source video
            active_indices: Optional list of indices in `self.configs` to actually run. 
                            If provided, others will be skip/None.
        """
        num_configs = len(self.configs)
        # 1. Initialize N copies of video state
        # We need independent mutable "working clips" for each config to maintain context divergence.
        # Structure: batch_states[config_idx][clip_idx]
        batch_states = [[c.model_copy() for c in masked_video.clips] for _ in range(num_configs)]
        
        # We also need a place to store results
        results_captions = [{} for _ in range(num_configs)]
        
        # If active_indices is None, run all
        if active_indices is None:
            active_indices = list(range(num_configs))
            
        active_set = set(active_indices)
        
        # 2. Iterate through clips (time-step)
        # All states have same number of clips
        num_clips = len(masked_video.clips)
        
        for clip_idx in range(num_clips):
            # Check if this clip needs filling (it is masked in the original)
            # Since all states start as copies of 'masked_video', we check the first one
            if not batch_states[0][clip_idx].is_masked():
                continue
                
            # 3. Prepare Batch Inputs (Only for active configs)
            prompts = []
            valid_batch_indices = [] # Map local batch idx -> global config idx
            
            for config_idx in active_indices:
                # Build context for this specific state
                current_clips = batch_states[config_idx]
                prompt_messages, computed_max = self._build_prompt_for_state(current_clips, clip_idx)
                prompts.append(prompt_messages)
                valid_batch_indices.append(config_idx)
                
            if not prompts:
                continue
                
            # 4. Run Batch Inference
            # Gather params for the active batch
            active_temps = [self.temperatures[i] for i in valid_batch_indices]
            active_pens = [self.penalties[i] for i in valid_batch_indices]
            
            # Simple max token heuristic: use default
            # A more complex one would be max(computed_max for all)
            
            try:
                responses = self.model_adapter.generate_batch(
                    messages_list=prompts,
                    temperatures=active_temps,
                    penalties=active_pens,
                    max_new_tokens=self.default_max_new_tokens
                )
                
                # 5. Update States
                for local_i, global_i in enumerate(valid_batch_indices):
                    text = self._clean_output(responses[local_i])
                    
                    # Save result
                    results_captions[global_i][batch_states[global_i][clip_idx].index] = text
                    
                    # Update context
                    old_clip = batch_states[global_i][clip_idx]
                    batch_states[global_i][clip_idx] = old_clip.model_copy(update={'caption': text})
            
            except Exception as e:
                logging.error(f"Batch generation failed at clip {clip_idx}: {e}")
                # Fail gracefully for this step? or crash?
                # For now, we record empty strings for this step to keep states valid-ish
                for global_i in valid_batch_indices:
                     results_captions[global_i][batch_states[global_i][clip_idx].index] = ""

        # 6. Return List of Results
        # For inactive indices, we return None or a 'skipped' result?
        # The runner expects one result per config.
        final_results = []
        for i in range(num_configs):
            if i in active_set:
                final_results.append(Reconstructed(
                    video_id=masked_video.video_id,
                    reconstructed_captions=results_captions[i]
                ))
            else:
                # Return a dummy skip result for bookkeeping
                final_results.append(Reconstructed(
                     video_id=masked_video.video_id,
                     reconstructed_captions={},
                     skip_reason="batch_inactive"
                ))
                
        return final_results

    def _build_prompt_for_state(self, clips: list[CaptionedClip], target_idx: int):
        # reuse logic from IterativeReconstructionStrategy
        # Ideally refactor `IterativeReconstructionStrategy` to extract `build_context`
        # But for now, I'll duplicate/adapt logic to avoid massive refactor risk.
        
        WINDOW_SIZE = 500
        i = target_idx
        
        def format_ts(ts) -> str:
            m, s = divmod(ts.start, 60)
            return f"[{int(m):02d}:{int(s):02d}]"

        start_before = max(0, i - WINDOW_SIZE)
        context_before_clips = clips[start_before:i]
        context_before_str = "\n".join(
            f"{format_ts(c.timestamp)} {c.caption}" for c in context_before_clips if c.caption
        )

        end_after = min(len(clips), i + 1 + WINDOW_SIZE)
        context_after_clips = clips[i+1:end_after]
        context_after_str = "\n".join(
            f"{format_ts(c.timestamp)} {c.caption}" for c in context_after_clips if c.caption
        )
        
        if not context_before_str.strip():
            context_before_end_hint = "START"
        else:
            context_before_end_hint = context_before_str.split()[-1]

        if not context_after_str.strip():
            context_after_start_hint = "END"
        else:
            context_after_start_hint = context_after_str.split()[0]
        
        target_clip = clips[i]
        target_timestamp = format_ts(target_clip.timestamp)
        target_duration = f"{target_clip.timestamp.duration:.1f}"

        prompt_context = {
            "TARGET_TIMESTAMP": target_timestamp,
            "TARGET_DURATION": target_duration,
            "CONTEXT_BEFORE": context_before_str,
            "CONTEXT_AFTER": context_after_str, 
            "CONTEXT_BEFORE_END_HINT": context_before_end_hint,
            "CONTEXT_AFTER_START_HINT": context_after_start_hint
        }
        
        # Max tokens
        lengths = [len(str(c.caption).split()) for c in context_before_clips + context_after_clips if c.caption]
        if lengths:
            avg_len = sum(lengths) / len(lengths)
            computed_max = int(avg_len * 2.5) 
            max_new = max(20, min(computed_max, 100)) 
        else:
            max_new = self.default_max_new_tokens
            
        return self.prompt_builder.build_prompt(prompt_context), max_new

    def _clean_output(self, text: str) -> str:
        # Duplicated from IterativeReconstructionStrategy
        text = re.sub(r'\[\d{2}:\d{2}(?::\d{2})?\]', '', text)
        text = text.strip().strip('"\'')
        text = re.sub(r'\s+', ' ', text).strip()
        return text
