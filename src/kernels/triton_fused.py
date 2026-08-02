"""Small, shape-polymorphic Triton fusions used by the DiT.

The kernels specialize only on channel dimensions, not batch or sequence
length, so the five data buckets share the same binaries. Set
``MINI_AGNES_TRITON=0`` to force the eager PyTorch paths.
"""

from __future__ import annotations

import os

import torch
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl

    _TRITON_AVAILABLE = True
except ImportError:
    triton = None
    tl = None
    _TRITON_AVAILABLE = False


def _enabled(*tensors: torch.Tensor) -> bool:
    return (
        _TRITON_AVAILABLE
        and os.environ.get("MINI_AGNES_TRITON", "1") != "0"
        and all(t.is_cuda and t.dtype in (torch.float16, torch.bfloat16) for t in tensors)
    )


def _uniform_rows(x: torch.Tensor) -> bool:
    """Whether leading dimensions can be flattened without changing row stride."""
    if x.ndim < 2 or x.stride(-1) != 1:
        return False
    row_stride = x.stride(-2)
    expected = row_stride * x.shape[-2]
    for dim in range(x.ndim - 3, -1, -1):
        if x.stride(dim) != expected:
            return False
        expected *= x.shape[dim]
    return True


if _TRITON_AVAILABLE:

    @triton.jit
    def _swiglu_fwd(
        x_ptr,
        out_ptr,
        n_elements,
        alpha: tl.constexpr,
        limit: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n_elements
        even = tl.load(x_ptr + 2 * offs, mask=mask).to(tl.float32)
        odd = tl.load(x_ptr + 2 * offs + 1, mask=mask).to(tl.float32)
        a = tl.minimum(even, limit)
        b = tl.maximum(tl.minimum(odd, limit), -limit)
        gate = a * tl.sigmoid(alpha * a)
        tl.store(out_ptr + offs, gate * (b + 1.0), mask=mask)


    @triton.jit
    def _swiglu_bwd(
        grad_ptr,
        x_ptr,
        dx_ptr,
        n_elements,
        alpha: tl.constexpr,
        limit: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n_elements
        grad = tl.load(grad_ptr + offs, mask=mask).to(tl.float32)
        even = tl.load(x_ptr + 2 * offs, mask=mask).to(tl.float32)
        odd = tl.load(x_ptr + 2 * offs + 1, mask=mask).to(tl.float32)
        a = tl.minimum(even, limit)
        b = tl.maximum(tl.minimum(odd, limit), -limit)
        sig = tl.sigmoid(alpha * a)
        gate = a * sig
        dgate = sig + alpha * a * sig * (1.0 - sig)
        da = grad * (b + 1.0) * dgate
        db = grad * gate
        da = tl.where(even <= limit, da, 0.0)
        db = tl.where((odd >= -limit) & (odd <= limit), db, 0.0)
        tl.store(dx_ptr + 2 * offs, da, mask=mask)
        tl.store(dx_ptr + 2 * offs + 1, db, mask=mask)


    @triton.jit
    def _rms_mod_fwd(
        x_ptr,
        shift_ptr,
        scale_ptr,
        out_ptr,
        rstd_ptr,
        x_row_stride,
        shift_row_stride,
        scale_row_stride,
        n_rows,
        n_cols: tl.constexpr,
        eps: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        row = tl.program_id(0)
        cols = tl.arange(0, BLOCK)
        mask = (row < n_rows) & (cols < n_cols)
        x = tl.load(x_ptr + row * x_row_stride + cols, mask=mask, other=0.0).to(tl.float32)
        mean_sq = tl.sum(x * x, axis=0) / n_cols
        rstd = tl.rsqrt(mean_sq + eps)
        shift = tl.load(
            shift_ptr + row * shift_row_stride + cols, mask=mask, other=0.0
        ).to(tl.float32)
        scale = tl.load(
            scale_ptr + row * scale_row_stride + cols, mask=mask, other=0.0
        ).to(tl.float32)
        tl.store(out_ptr + row * n_cols + cols, x * rstd * (1.0 + scale) + shift, mask=mask)
        tl.store(rstd_ptr + row, rstd)


    @triton.jit
    def _rms_mod_bwd(
        grad_ptr,
        x_ptr,
        scale_ptr,
        rstd_ptr,
        dx_ptr,
        dshift_ptr,
        dscale_ptr,
        x_row_stride,
        scale_row_stride,
        n_rows,
        n_cols: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        row = tl.program_id(0)
        cols = tl.arange(0, BLOCK)
        mask = (row < n_rows) & (cols < n_cols)
        grad = tl.load(grad_ptr + row * n_cols + cols, mask=mask, other=0.0).to(tl.float32)
        x = tl.load(x_ptr + row * x_row_stride + cols, mask=mask, other=0.0).to(tl.float32)
        scale = tl.load(
            scale_ptr + row * scale_row_stride + cols, mask=mask, other=0.0
        ).to(tl.float32)
        rstd = tl.load(rstd_ptr + row).to(tl.float32)
        norm = x * rstd
        z = grad * (1.0 + scale)
        dot = tl.sum(z * x, axis=0) / n_cols
        dx = rstd * (z - x * dot * rstd * rstd)
        tl.store(dx_ptr + row * n_cols + cols, dx, mask=mask)
        tl.store(dshift_ptr + row * n_cols + cols, grad, mask=mask)
        tl.store(dscale_ptr + row * n_cols + cols, grad * norm, mask=mask)


    @triton.jit
    def _gate_residual_fwd(
        x_ptr,
        y_ptr,
        gate_ptr,
        out_ptr,
        x_row_stride,
        y_row_stride,
        gate_row_stride,
        n_rows,
        n_cols: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        row = tl.program_id(0)
        cols = tl.arange(0, BLOCK)
        mask = (row < n_rows) & (cols < n_cols)
        x = tl.load(x_ptr + row * x_row_stride + cols, mask=mask)
        y = tl.load(y_ptr + row * y_row_stride + cols, mask=mask)
        gate = tl.load(gate_ptr + row * gate_row_stride + cols, mask=mask)
        tl.store(out_ptr + row * n_cols + cols, x + y * gate, mask=mask)


    @triton.jit
    def _gate_residual_bwd(
        grad_ptr,
        y_ptr,
        gate_ptr,
        dx_ptr,
        dy_ptr,
        dgate_ptr,
        y_row_stride,
        gate_row_stride,
        n_rows,
        n_cols: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        row = tl.program_id(0)
        cols = tl.arange(0, BLOCK)
        mask = (row < n_rows) & (cols < n_cols)
        grad = tl.load(grad_ptr + row * n_cols + cols, mask=mask)
        y = tl.load(y_ptr + row * y_row_stride + cols, mask=mask)
        gate = tl.load(gate_ptr + row * gate_row_stride + cols, mask=mask)
        tl.store(dx_ptr + row * n_cols + cols, grad, mask=mask)
        tl.store(dy_ptr + row * n_cols + cols, grad * gate, mask=mask)
        tl.store(dgate_ptr + row * n_cols + cols, grad * y, mask=mask)


    @triton.jit
    def _qk_norm_rope_fwd(
        q_ptr,
        k_ptr,
        qw_ptr,
        kw_ptr,
        cos_ptr,
        sin_ptr,
        qo_ptr,
        ko_ptr,
        qrstd_ptr,
        krstd_ptr,
        qsb,
        qsl,
        qsh,
        ksb,
        ksl,
        ksh,
        csb,
        csl,
        n_rows,
        seq_len,
        n_heads: tl.constexpr,
        head_dim: tl.constexpr,
        rope_half: tl.constexpr,
        eps: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        row = tl.program_id(0)
        cols = tl.arange(0, BLOCK)
        mask = (row < n_rows) & (cols < head_dim)
        head = row % n_heads
        tmp = row // n_heads
        pos = tmp % seq_len
        batch = tmp // seq_len
        qbase = batch * qsb + pos * qsl + head * qsh
        kbase = batch * ksb + pos * ksl + head * ksh
        q = tl.load(q_ptr + qbase + cols, mask=mask, other=0.0).to(tl.float32)
        k = tl.load(k_ptr + kbase + cols, mask=mask, other=0.0).to(tl.float32)
        qrstd = tl.rsqrt(tl.sum(q * q, axis=0) / head_dim + eps)
        krstd = tl.rsqrt(tl.sum(k * k, axis=0) / head_dim + eps)
        qw = tl.load(qw_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        kw = tl.load(kw_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        qn = q * qrstd * qw
        kn = k * krstd * kw

        rope_dim = 2 * rope_half
        pair_col = tl.where(cols < rope_half, cols + rope_half, cols - rope_half)
        pair_mask = mask & (cols < rope_dim)
        qpair = tl.load(q_ptr + qbase + pair_col, mask=pair_mask, other=0.0).to(tl.float32)
        kpair = tl.load(k_ptr + kbase + pair_col, mask=pair_mask, other=0.0).to(tl.float32)
        qwpair = tl.load(qw_ptr + pair_col, mask=pair_mask, other=0.0).to(tl.float32)
        kwpair = tl.load(kw_ptr + pair_col, mask=pair_mask, other=0.0).to(tl.float32)
        qnpair = qpair * qrstd * qwpair
        knpair = kpair * krstd * kwpair
        rope_col = tl.where(cols < rope_half, cols, cols - rope_half)
        c = tl.load(cos_ptr + batch * csb + pos * csl + rope_col, mask=pair_mask, other=0.0).to(
            tl.float32
        )
        s = tl.load(sin_ptr + batch * csb + pos * csl + rope_col, mask=pair_mask, other=0.0).to(
            tl.float32
        )
        sign = tl.where(cols < rope_half, -1.0, 1.0)
        qo = tl.where(cols < rope_dim, qn * c + sign * qnpair * s, qn)
        ko = tl.where(cols < rope_dim, kn * c + sign * knpair * s, kn)
        tl.store(qo_ptr + row * head_dim + cols, qo, mask=mask)
        tl.store(ko_ptr + row * head_dim + cols, ko, mask=mask)
        tl.store(qrstd_ptr + row, qrstd)
        tl.store(krstd_ptr + row, krstd)


    @triton.jit
    def _qk_norm_rope_dx(
        dqo_ptr,
        dko_ptr,
        q_ptr,
        k_ptr,
        qw_ptr,
        kw_ptr,
        cos_ptr,
        sin_ptr,
        qrstd_ptr,
        krstd_ptr,
        dq_ptr,
        dk_ptr,
        qsb,
        qsl,
        qsh,
        ksb,
        ksl,
        ksh,
        csb,
        csl,
        n_rows,
        seq_len,
        n_heads: tl.constexpr,
        head_dim: tl.constexpr,
        rope_half: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        row = tl.program_id(0)
        cols = tl.arange(0, BLOCK)
        mask = (row < n_rows) & (cols < head_dim)
        head = row % n_heads
        tmp = row // n_heads
        pos = tmp % seq_len
        batch = tmp // seq_len
        qbase = batch * qsb + pos * qsl + head * qsh
        kbase = batch * ksb + pos * ksl + head * ksh
        rope_dim = 2 * rope_half
        pair_col = tl.where(cols < rope_half, cols + rope_half, cols - rope_half)
        pair_mask = mask & (cols < rope_dim)
        rope_col = tl.where(cols < rope_half, cols, cols - rope_half)
        c = tl.load(cos_ptr + batch * csb + pos * csl + rope_col, mask=pair_mask, other=0.0).to(
            tl.float32
        )
        s = tl.load(sin_ptr + batch * csb + pos * csl + rope_col, mask=pair_mask, other=0.0).to(
            tl.float32
        )
        dqo = tl.load(dqo_ptr + row * head_dim + cols, mask=mask, other=0.0).to(tl.float32)
        dko = tl.load(dko_ptr + row * head_dim + cols, mask=mask, other=0.0).to(tl.float32)
        dqop = tl.load(
            dqo_ptr + row * head_dim + pair_col, mask=pair_mask, other=0.0
        ).to(tl.float32)
        dkop = tl.load(
            dko_ptr + row * head_dim + pair_col, mask=pair_mask, other=0.0
        ).to(tl.float32)
        inv_sign = tl.where(cols < rope_half, 1.0, -1.0)
        dqn = tl.where(cols < rope_dim, dqo * c + inv_sign * dqop * s, dqo)
        dkn = tl.where(cols < rope_dim, dko * c + inv_sign * dkop * s, dko)

        q = tl.load(q_ptr + qbase + cols, mask=mask, other=0.0).to(tl.float32)
        k = tl.load(k_ptr + kbase + cols, mask=mask, other=0.0).to(tl.float32)
        qw = tl.load(qw_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        kw = tl.load(kw_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        qrstd = tl.load(qrstd_ptr + row).to(tl.float32)
        krstd = tl.load(krstd_ptr + row).to(tl.float32)
        qz = dqn * qw
        kz = dkn * kw
        qdot = tl.sum(qz * q, axis=0) / head_dim
        kdot = tl.sum(kz * k, axis=0) / head_dim
        dq = qrstd * (qz - q * qdot * qrstd * qrstd)
        dk = krstd * (kz - k * kdot * krstd * krstd)
        tl.store(dq_ptr + row * head_dim + cols, dq, mask=mask)
        tl.store(dk_ptr + row * head_dim + cols, dk, mask=mask)


    @triton.jit
    def _qk_norm_rope_dw(
        dqo_ptr,
        dko_ptr,
        q_ptr,
        k_ptr,
        cos_ptr,
        sin_ptr,
        qrstd_ptr,
        krstd_ptr,
        dqw_ptr,
        dkw_ptr,
        qsb,
        qsl,
        qsh,
        ksb,
        ksl,
        ksh,
        csb,
        csl,
        n_rows,
        seq_len,
        n_heads: tl.constexpr,
        head_dim: tl.constexpr,
        rope_half: tl.constexpr,
        ROW_BLOCK: tl.constexpr,
    ):
        col = tl.program_id(0)
        row_block = tl.program_id(1)
        rows = row_block * ROW_BLOCK + tl.arange(0, ROW_BLOCK)
        mask = rows < n_rows
        head = rows % n_heads
        tmp = rows // n_heads
        pos = tmp % seq_len
        batch = tmp // seq_len
        qbase = batch * qsb + pos * qsl + head * qsh
        kbase = batch * ksb + pos * ksl + head * ksh
        rope_dim = 2 * rope_half
        pair_col = tl.where(col < rope_half, col + rope_half, col - rope_half)
        dqo = tl.load(dqo_ptr + rows * head_dim + col, mask=mask, other=0.0).to(tl.float32)
        dko = tl.load(dko_ptr + rows * head_dim + col, mask=mask, other=0.0).to(tl.float32)
        pair_mask = mask & (col < rope_dim)
        dqop = tl.load(
            dqo_ptr + rows * head_dim + pair_col, mask=pair_mask, other=0.0
        ).to(tl.float32)
        dkop = tl.load(
            dko_ptr + rows * head_dim + pair_col, mask=pair_mask, other=0.0
        ).to(tl.float32)
        rope_col = tl.where(col < rope_half, col, col - rope_half)
        c = tl.load(
            cos_ptr + batch * csb + pos * csl + rope_col, mask=pair_mask, other=0.0
        ).to(tl.float32)
        s = tl.load(
            sin_ptr + batch * csb + pos * csl + rope_col, mask=pair_mask, other=0.0
        ).to(tl.float32)
        inv_sign = tl.where(col < rope_half, 1.0, -1.0)
        dqo = tl.where(col < rope_dim, dqo * c + inv_sign * dqop * s, dqo)
        dko = tl.where(col < rope_dim, dko * c + inv_sign * dkop * s, dko)
        q = tl.load(q_ptr + qbase + col, mask=mask, other=0.0).to(tl.float32)
        k = tl.load(k_ptr + kbase + col, mask=mask, other=0.0).to(tl.float32)
        qrstd = tl.load(qrstd_ptr + rows, mask=mask, other=0.0).to(tl.float32)
        krstd = tl.load(krstd_ptr + rows, mask=mask, other=0.0).to(tl.float32)
        qpart = tl.sum(dqo * q * qrstd, axis=0)
        kpart = tl.sum(dko * k * krstd, axis=0)
        tl.atomic_add(dqw_ptr + col, qpart)
        tl.atomic_add(dkw_ptr + col, kpart)


class _SwiGLU7(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, alpha: float, limit: float):
        out = torch.empty((*x.shape[:-1], x.shape[-1] // 2), device=x.device, dtype=x.dtype)
        n = out.numel()
        _swiglu_fwd[(triton.cdiv(n, 256),)](x, out, n, alpha, limit, BLOCK=256)
        ctx.save_for_backward(x)
        ctx.alpha = alpha
        ctx.limit = limit
        return out

    @staticmethod
    def backward(ctx, grad: torch.Tensor):
        (x,) = ctx.saved_tensors
        grad = grad.contiguous()
        dx = torch.empty_like(x)
        n = grad.numel()
        _swiglu_bwd[(triton.cdiv(n, 256),)](
            grad, x, dx, n, ctx.alpha, ctx.limit, BLOCK=256
        )
        return dx, None, None


class _RMSModulate(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, shift, scale, eps):
        n_cols = x.shape[-1]
        n_rows = x.numel() // n_cols
        out = torch.empty_like(x, memory_format=torch.contiguous_format)
        rstd = torch.empty(n_rows, device=x.device, dtype=torch.float32)
        block = triton.next_power_of_2(n_cols)
        _rms_mod_fwd[(n_rows,)](
            x,
            shift,
            scale,
            out,
            rstd,
            x.stride(-2),
            shift.stride(-2),
            scale.stride(-2),
            n_rows,
            n_cols,
            eps,
            BLOCK=block,
            num_warps=8,
        )
        ctx.save_for_backward(x, scale, rstd)
        return out

    @staticmethod
    def backward(ctx, grad):
        x, scale, rstd = ctx.saved_tensors
        grad = grad.contiguous()
        dx = torch.empty_like(x, memory_format=torch.contiguous_format)
        dshift = torch.empty_like(x, memory_format=torch.contiguous_format)
        dscale = torch.empty_like(x, memory_format=torch.contiguous_format)
        n_cols = x.shape[-1]
        n_rows = x.numel() // n_cols
        _rms_mod_bwd[(n_rows,)](
            grad,
            x,
            scale,
            rstd,
            dx,
            dshift,
            dscale,
            x.stride(-2),
            scale.stride(-2),
            n_rows,
            n_cols,
            BLOCK=triton.next_power_of_2(n_cols),
            num_warps=8,
        )
        return dx, dshift, dscale, None


class _GateResidual(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, y, gate):
        n_cols = x.shape[-1]
        n_rows = x.numel() // n_cols
        out = torch.empty_like(x, memory_format=torch.contiguous_format)
        _gate_residual_fwd[(n_rows,)](
            x,
            y,
            gate,
            out,
            x.stride(-2),
            y.stride(-2),
            gate.stride(-2),
            n_rows,
            n_cols,
            BLOCK=triton.next_power_of_2(n_cols),
            num_warps=8,
        )
        ctx.save_for_backward(y, gate)
        return out

    @staticmethod
    def backward(ctx, grad):
        y, gate = ctx.saved_tensors
        grad = grad.contiguous()
        dx = torch.empty_like(grad)
        dy = torch.empty_like(grad)
        dgate = torch.empty_like(grad)
        n_cols = grad.shape[-1]
        n_rows = grad.numel() // n_cols
        _gate_residual_bwd[(n_rows,)](
            grad,
            y,
            gate,
            dx,
            dy,
            dgate,
            y.stride(-2),
            gate.stride(-2),
            n_rows,
            n_cols,
            BLOCK=triton.next_power_of_2(n_cols),
            num_warps=8,
        )
        return dx, dy, dgate


class _QKNormRoPE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, q_weight, k_weight, cos, sin, eps):
        batch, seq_len, n_heads, head_dim = q.shape
        rope_half = cos.shape[-1]
        n_rows = batch * seq_len * n_heads
        qo = torch.empty_like(q, memory_format=torch.contiguous_format)
        ko = torch.empty_like(k, memory_format=torch.contiguous_format)
        qrstd = torch.empty(n_rows, device=q.device, dtype=torch.float32)
        krstd = torch.empty_like(qrstd)
        _qk_norm_rope_fwd[(n_rows,)](
            q,
            k,
            q_weight,
            k_weight,
            cos,
            sin,
            qo,
            ko,
            qrstd,
            krstd,
            q.stride(0),
            q.stride(1),
            q.stride(2),
            k.stride(0),
            k.stride(1),
            k.stride(2),
            cos.stride(0),
            cos.stride(1),
            n_rows,
            seq_len,
            n_heads,
            head_dim,
            rope_half,
            eps,
            BLOCK=triton.next_power_of_2(head_dim),
            num_warps=4,
        )
        ctx.save_for_backward(q, k, q_weight, k_weight, cos, sin, qrstd, krstd)
        return qo, ko

    @staticmethod
    def backward(ctx, dqo, dko):
        q, k, q_weight, k_weight, cos, sin, qrstd, krstd = ctx.saved_tensors
        dqo, dko = dqo.contiguous(), dko.contiguous()
        batch, seq_len, n_heads, head_dim = q.shape
        rope_half = cos.shape[-1]
        n_rows = batch * seq_len * n_heads
        dq = torch.empty_like(q, memory_format=torch.contiguous_format)
        dk = torch.empty_like(k, memory_format=torch.contiguous_format)
        common = (
            dqo,
            dko,
            q,
            k,
            q_weight,
            k_weight,
            cos,
            sin,
            qrstd,
            krstd,
            dq,
            dk,
            q.stride(0),
            q.stride(1),
            q.stride(2),
            k.stride(0),
            k.stride(1),
            k.stride(2),
            cos.stride(0),
            cos.stride(1),
            n_rows,
            seq_len,
            n_heads,
            head_dim,
            rope_half,
        )
        _qk_norm_rope_dx[(n_rows,)](
            *common,
            BLOCK=triton.next_power_of_2(head_dim),
            num_warps=4,
        )
        dqw = torch.zeros_like(q_weight, dtype=torch.float32)
        dkw = torch.zeros_like(k_weight, dtype=torch.float32)
        row_block = 256
        _qk_norm_rope_dw[(head_dim, triton.cdiv(n_rows, row_block))](
            dqo,
            dko,
            q,
            k,
            cos,
            sin,
            qrstd,
            krstd,
            dqw,
            dkw,
            q.stride(0),
            q.stride(1),
            q.stride(2),
            k.stride(0),
            k.stride(1),
            k.stride(2),
            cos.stride(0),
            cos.stride(1),
            n_rows,
            seq_len,
            n_heads,
            head_dim,
            rope_half,
            ROW_BLOCK=row_block,
            num_warps=4,
        )
        return dq, dk, dqw.to(q_weight.dtype), dkw.to(k_weight.dtype), None, None, None


def fused_swiglu7(
    x: torch.Tensor, alpha: float = 1.702, limit: float = 7.0
) -> torch.Tensor:
    if _enabled(x) and x.is_contiguous() and x.shape[-1] % 2 == 0:
        return _SwiGLU7.apply(x, alpha, limit)
    x_glu = x.view(*x.shape[:-1], -1, 2)
    a = x_glu[..., 0].clamp(max=limit)
    b = x_glu[..., 1].clamp(min=-limit, max=limit)
    return a * torch.sigmoid(alpha * a) * (b + 1.0)


def fused_rms_modulate(
    x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor, eps: float = 1e-6
) -> torch.Tensor:
    if (
        _enabled(x, shift, scale)
        and x.shape == shift.shape == scale.shape
        and _uniform_rows(x)
        and _uniform_rows(shift)
        and _uniform_rows(scale)
        and x.shape[-1] <= 65536
    ):
        return _RMSModulate.apply(x, shift, scale, eps)
    return F.rms_norm(x, (x.shape[-1],), eps=eps) * (1.0 + scale) + shift


def fused_gate_residual(
    x: torch.Tensor, y: torch.Tensor, gate: torch.Tensor
) -> torch.Tensor:
    if (
        _enabled(x, y, gate)
        and x.shape == y.shape == gate.shape
        and _uniform_rows(x)
        and _uniform_rows(y)
        and _uniform_rows(gate)
        and x.shape[-1] <= 65536
    ):
        return _GateResidual.apply(x, y, gate)
    return x + gate * y


def fused_qk_norm_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Return ``None`` when the fused fast path cannot preserve semantics."""
    valid = (
        _enabled(q, k)
        and q.shape == k.shape
        and q.ndim == 4
        and q.stride(-1) == k.stride(-1) == 1
        and q_weight.is_cuda
        and k_weight.is_cuda
        and q_weight.shape == k_weight.shape == (q.shape[-1],)
        and cos.is_cuda
        and sin.is_cuda
        and cos.shape == sin.shape
        and cos.ndim == 3
        and cos.shape[:2] == q.shape[:2]
        and cos.stride(-1) == sin.stride(-1) == 1
        and 2 * cos.shape[-1] <= q.shape[-1]
        and q.shape[-1] <= 256
    )
    if not valid:
        return None
    return _QKNormRoPE.apply(q, k, q_weight, k_weight, cos, sin, eps)
