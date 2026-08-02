"""Optional fused CUDA kernels with numerically safe PyTorch fallbacks."""

from src.kernels.triton_fused import (  # noqa: F401
    fused_gate_residual,
    fused_qk_norm_rope,
    fused_rms_modulate,
    fused_swiglu7,
)
from src.kernels.varlen_attn import segment_ids, varlen_attention  # noqa: F401
