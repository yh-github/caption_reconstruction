
import logging
import torch
import os

logger = logging.getLogger(__name__)

_DEVICE = None
_IS_TPU = False

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
