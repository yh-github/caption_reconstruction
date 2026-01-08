
import logging
import torch
import time
from common_utils import device_setup

logger = logging.getLogger(__name__)

def run_sanity_check():
    """
    Performs a quick sanity check to verify:
    1. Device detection (TPU vs CUDA vs CPU)
    2. Tensor creation on device
    3. Small matrix multiplication (hardware health check)
    """
    device = device_setup.get_device()
    is_tpu = device_setup.is_tpu()
    
    logger.info("=" * 40)
    logger.info("   STARTING HARDWARE SANITY CHECK")
    logger.info("=" * 40)
    logger.info(f"Target Device: {device}")
    logger.info(f"Is TPU: {is_tpu}")
    
    try:
        # 1. Create Tensor
        t0 = time.time()
        dtype = device_setup.get_compute_dtype()
        x = torch.randn(1024, 1024, device=device, dtype=dtype)
        y = torch.randn(1024, 1024, device=device, dtype=dtype)
        logger.info(f"✓ Tensor allocation successful ({dtype})")
        
        # 2. Compute (Matrix Mul)
        # On TPU, this triggers XLA compilation for this graph
        z = torch.matmul(x, y)
        
        # 3. Synchronize / Materialize
        # XLA is lazy, so we need to print or access data to force execution
        if is_tpu:
            import torch_xla.core.xla_model as xm
            xm.mark_step()
            
        res_sum = z.sum().item()
        t1 = time.time()
        
        logger.info(f"✓ Matrix multiplication successful (Sum: {res_sum:.2f})")
        logger.info(f"✓ Tensors moved to device and computed in {t1-t0:.4f}s")
        
        if is_tpu:
             logger.info("🚀 TPU is ONLINE and functioning!")
        elif device.type == "cuda":
             logger.info("🟢 CUDA GPU is ONLINE and functioning!")
        else:
             logger.warning("⚠️  Running on CPU. Performance will be slow.")
             
    except Exception as e:
        logger.error("❌ SANITY CHECK FAILED!")
        logger.error(f"Error details: {e}")
        # Build strict failure if it's supposed to be TPU but failed
        if is_tpu:
             raise RuntimeError("TPU Sanity Check Failed. See logs for details.") from e
        raise e
        
    logger.info("=" * 40)
