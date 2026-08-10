
import logging
import torch
import os
import yaml
from pathlib import Path

logger = logging.getLogger(__name__)

_DEVICE = None
_IS_TPU = False
_LLM_BACKEND = None

def _init_device():
    global _DEVICE, _IS_TPU
    
    # Check for TPU/XLA
    # Usually requires `import torch_xla.core.xla_model as xm`
    # We guard the import to avoid crashing on non-TPU systems
    try:
        import torch_xla.core.xla_model as xm
        devices = xm.get_xla_supported_devices()
        if devices:
            _DEVICE = xm.xla_device()
            _IS_TPU = True
            logger.info(f"TPU detected. Using device: {_DEVICE}")
            return
    except ImportError:
        pass
    except Exception as e:
        logger.warning(f"Error checking for XLA device: {e}")

    # Fallback to CUDA or CPU
    if torch.cuda.is_available():
        _DEVICE = torch.device("cuda")
        logger.info("CUDA GPU detected. Using device: cuda")
    elif torch.backends.mps.is_available():
        _DEVICE = torch.device("mps")
        logger.info("MPS (Mac) detected. Using device: mps")
    else:
        _DEVICE = torch.device("cpu")
        logger.info("No accelerator detected. Using device: cpu")    

def get_device() -> torch.device:
    if _DEVICE is None:
        _init_device()
    return _DEVICE

def is_tpu() -> bool:
    if _DEVICE is None:
        _init_device()
    return _IS_TPU

def get_llm_backend() -> str:
    """
    Determines the system-wide LLM backend to use.
    Returns: 'keras_llm' or 'pytorch'
    """
    global _LLM_BACKEND
    if _LLM_BACKEND is not None:
        return _LLM_BACKEND

    # 1. Check system.yaml for override
    try:
        if Path("config/system.yaml").exists():
            with open("config/system.yaml", "r") as f:
                sys_conf = yaml.safe_load(f) or {}
                # Look for top-level key 'llm_backend'
                if "llm_backend" in sys_conf:
                    requested = sys_conf["llm_backend"]
                    logger.info(f"System config overrides LLM backend to: {requested}")
                    _LLM_BACKEND = requested
                    return _LLM_BACKEND
    except Exception as e:
        logger.warning(f"Failed to read system.yaml for backend override: {e}")

    # 2. Auto-Detect
    # If TPU is present AND Keras/JAX libraries are importable, prefer KerasLLM
    # (assuming Keras 3 + JAX is the optimized path for TPU v5e)
    if is_tpu():
        try:
            import keras
            import keras_nlp
            import jax
            logger.info("TPU detected and Keras/JAX installed. Defaulting to 'keras_llm' backend.")
            _LLM_BACKEND = "keras_llm"
            return _LLM_BACKEND
        except ImportError:
            logger.info("TPU detected but Keras/JAX missing. Falling back to 'pytorch' backend.")
    
    # Default fallback
    _LLM_BACKEND = "pytorch"
    return _LLM_BACKEND

def get_compute_dtype() -> torch.dtype:
    """
    Returns the preferred floating point type for the current device.
    TPU -> bfloat16
    CUDA -> float16 (or bfloat16 if supported, but defaulting to float16 for T4 compatibility)
    CPU -> float32
    """
    if is_tpu():
        return torch.bfloat16
    
    if is_cuda():
        # Check for Ampere or newer (SM >= 80) for BF16, otherwise FP16
        # T4 is SM 7.5, so FP16.
        if torch.cuda.get_device_capability()[0] >= 8:
            return torch.bfloat16
        return torch.float16
        
    return torch.float32

def is_cuda() -> bool:
    d = get_device()
    return d.type == "cuda"

def clear_cache():
    if is_cuda():
        torch.cuda.empty_cache()
    # XLA doesn't exactly have an empty_cache equivalent in the same way, 
    # but memory management is handled by the XLA runtime.

def calculate_optimal_batch_size(requested_batch_size: int, default_min: int = 1, gb_per_sequence: float = 1.5) -> int:
    """
    Dynamically computes optimal batch size based on available GPU VRAM.
    Defaults to requested_batch_size on CPU, TPU, or if CUDA mem_get_info is unavailable.
    """
    if requested_batch_size <= 1:
        return max(default_min, requested_batch_size)

    if not is_cuda():
        return requested_batch_size

    try:
        free_bytes, total_bytes = torch.cuda.mem_get_info()
        free_vram_gb = free_bytes / (1024 ** 3)
        safe_batch = max(default_min, int(free_vram_gb // gb_per_sequence))
        optimal = max(default_min, min(requested_batch_size, safe_batch))
        logger.debug(f"Dynamic VRAM Batch Sizing: free_vram={free_vram_gb:.2f}GB -> batch_size={optimal} (requested={requested_batch_size})")
        return optimal
    except Exception as e:
        logger.warning(f"Could not query CUDA memory info ({e}). Using requested batch size {requested_batch_size}")
        return requested_batch_size

