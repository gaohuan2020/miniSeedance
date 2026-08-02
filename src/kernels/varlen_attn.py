"""Block-diagonal attention over a packed token sequence.

Sequence packing puts several samples in one flat sequence, so attention has to
stop at segment boundaries. FlashAttention-4's varlen entry point does exactly
that, but its Python interface is not dynamo-traceable (the CuTe JIT queries the
device and its kernel cache while tracing), so under torch.compile it breaks the
graph in every block. Wrapping FA4's fwd/bwd kernels in a pair of opaque custom
ops keeps FA4 *and* full inductor fusion around it: measured 144 ms/24 blocks
versus 226 ms eager and 281 ms for a flex_attention block mask, at a 12k-token
bin (1280x24, 20 heads).

`cu_seqlens` may end in repeated values; FA4 treats the resulting zero-length
segments as empty, which is what lets the packer keep a fixed-size cu_seqlens
buffer (and therefore a single compiled graph) regardless of how many samples
landed in the bin.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

try:
    from flash_attn.cute.interface import _flash_attn_bwd, _flash_attn_fwd

    _FA4_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only without flash-attn-4
    _flash_attn_bwd = _flash_attn_fwd = None
    _FA4_AVAILABLE = False


if _FA4_AVAILABLE:

    @torch.library.custom_op("mini_agnes::varlen_attn", mutates_args=(), device_types="cuda")
    def _varlen_attn(
        q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
        cu_seqlens: torch.Tensor, max_seqlen: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        out, lse, *_ = _flash_attn_fwd(
            q, k, v, cu_seqlens_q=cu_seqlens, cu_seqlens_k=cu_seqlens,
            max_seqlen_q=max_seqlen, max_seqlen_k=max_seqlen, return_lse=True,
        )
        return out, lse

    @_varlen_attn.register_fake
    def _(q, k, v, cu_seqlens, max_seqlen):
        total, heads, _ = q.shape
        return torch.empty_like(q), q.new_empty((heads, total), dtype=torch.float32)

    @torch.library.custom_op(
        "mini_agnes::varlen_attn_bwd", mutates_args=(), device_types="cuda"
    )
    def _varlen_attn_bwd(
        q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, out: torch.Tensor,
        dout: torch.Tensor, lse: torch.Tensor, cu_seqlens: torch.Tensor, max_seqlen: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return _flash_attn_bwd(
            q, k, v, out, dout, lse, None, False, 0.0,
            cu_seqlens_q=cu_seqlens, cu_seqlens_k=cu_seqlens,
            max_seqlen_q=max_seqlen, max_seqlen_k=max_seqlen,
        )

    @_varlen_attn_bwd.register_fake
    def _(q, k, v, out, dout, lse, cu_seqlens, max_seqlen):
        return torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)

    def _setup_context(ctx, inputs, output):
        q, k, v, cu_seqlens, max_seqlen = inputs
        ctx.save_for_backward(q, k, v, output[0], output[1], cu_seqlens)
        ctx.max_seqlen = max_seqlen

    def _backward(ctx, grad_out, grad_lse):
        q, k, v, out, lse, cu_seqlens = ctx.saved_tensors
        dq, dk, dv = _varlen_attn_bwd(
            q, k, v, out, grad_out.contiguous(), lse, cu_seqlens, ctx.max_seqlen
        )
        return dq, dk, dv, None, None

    _varlen_attn.register_autograd(_backward, setup_context=_setup_context)


def segment_ids(cu_seqlens: torch.Tensor, total: int) -> torch.Tensor:
    """(total,) segment index of every token in a packed sequence."""
    pos = torch.arange(total, device=cu_seqlens.device)
    return torch.searchsorted(cu_seqlens[1:].contiguous(), pos, right=True)


def varlen_attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
    cu_seqlens: torch.Tensor, max_seqlen: int,
) -> torch.Tensor:
    """Attention restricted to the segments described by `cu_seqlens`.

    q/k/v: (total, heads, head_dim). Falls back to SDPA with an explicit
    block-diagonal mask when FA4 is unavailable or the inputs are not bf16/fp16
    CUDA tensors (tests, CPU).
    """
    if _FA4_AVAILABLE and q.is_cuda and q.dtype in (torch.float16, torch.bfloat16):
        return _varlen_attn(q, k, v, cu_seqlens, max_seqlen)[0]

    seg = segment_ids(cu_seqlens, q.shape[0])
    mask = seg[:, None] == seg[None, :]
    qt, kt, vt = (t.transpose(0, 1).unsqueeze(0) for t in (q, k, v))
    out = F.scaled_dot_product_attention(qt, kt, vt, attn_mask=mask)
    return out.squeeze(0).transpose(0, 1)
