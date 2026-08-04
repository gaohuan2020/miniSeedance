"""
Minimal text-to-video DiT.

Architecture follows daVinci-MagiHuman's DiT (inference/model/dit/dit_module.py):
- single-stream joint sequence: [video tokens; text tokens] with per-modality
  embedders (Adapter) and joint full 3D spatio-temporal attention
- 3D RoPE from (t, h, w) coordinates (ElementWiseFourierEmbed / freq_bands,
  rotate_half, apply_rotary_emb_torch copied from MagiHuman)
- RMSNorm-based pre-norm, q/k RMSNorm inside attention
- SwiGLU7 MLP (copied from MagiHuman)
- video-only final head

MagiHuman's inference DiT is a distilled model with no timestep conditioning,
which Self-Flow training requires. So per-token adaLN-Zero timestep modulation
and the self-distillation projector (SimpleHead) are grafted in from Self-Flow.

The forward pass keeps Self-Flow's compatibility convention (`timesteps = 1 -
timesteps`, output negated) so denoise_loop in sampling.py works unchanged.
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

from src.kernels import (
    fused_gate_residual,
    fused_qk_norm_rope,
    fused_rms_modulate,
    fused_swiglu7,
    varlen_attention,
)
from src.utils import AUDIO, PAD, TEXT, VIDEO, PackPlan

class TimestepEmbedder(nn.Module):
    """Embeds scalar timesteps into vector representations (from Self-Flow)."""

    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(start=0, end=half, dtype=torch.float32)
            / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
            )
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        return self.mlp(t_freq)


class SimpleHead(nn.Module):
    """Simple projection head for self-distillation (from Self-Flow)."""

    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear1 = nn.Linear(in_dim, in_dim + out_dim)
        self.linear2 = nn.Linear(in_dim + out_dim, out_dim)
        self.act = nn.SiLU()

    def forward(self, x):
        x = self.linear1(x)
        x = self.linear2(self.act(x))
        return x


#################################################################################
#                       3D RoPE (from daVinci-MagiHuman)                        #
#################################################################################

def freq_bands(num_bands: int, temperature: float = 10000.0, step: int = 2) -> torch.Tensor:
    exp = torch.arange(0, num_bands, step, dtype=torch.int64).to(torch.float32) / num_bands
    bands = 1.0 / (temperature**exp)
    return bands


def rotate_half(x, interleaved=False):
    if not interleaved:
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)
    else:
        x1, x2 = x[..., ::2], x[..., 1::2]
        return rearrange(torch.stack((-x2, x1), dim=-1), "... d two -> ... (d two)", two=2)


def apply_rotary_emb_torch(x, cos, sin, interleaved=False):
    """
    x: (batch_size, seqlen, nheads, headdim)
    cos, sin: (seqlen, rotary_dim / 2) or (batch_size, seqlen, rotary_dim / 2)
    """
    ro_dim = cos.shape[-1] * 2
    assert ro_dim <= x.shape[-1]
    cos = repeat(cos, "... d -> ... 1 (2 d)" if not interleaved else "... d -> ... 1 (d 2)")
    sin = repeat(sin, "... d -> ... 1 (2 d)" if not interleaved else "... d -> ... 1 (d 2)")
    return torch.cat([x[..., :ro_dim] * cos + rotate_half(x[..., :ro_dim], interleaved) * sin, x[..., ro_dim:]], dim=-1)


class Rope3D(nn.Module):
    """Axis-split Fourier embedding of ``(t, h, w)`` coordinates.

    Historical configs learn ``head_dim // 8`` shared bands. H3-style configs
    pass ``freq_dim`` to use that many fixed frequencies per axis. The resulting
    rotary layout is ``[T | H | W]`` in each half, leaving remaining head
    channels unrotated.
    """

    def __init__(
        self,
        head_dim: int,
        temperature: float = 10000.0,
        freq_dim: Optional[int] = None,
    ):
        super().__init__()
        self.head_dim = head_dim
        self.freq_dim = head_dim // 8 if freq_dim is None else freq_dim
        bands = freq_bands(self.freq_dim, temperature=temperature, step=1)
        if freq_dim is None:
            # Preserve historical state-dict behavior for old checkpoint configs.
            self.bands = nn.Parameter(bands)
        else:
            self.register_buffer("bands", bands, persistent=True)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """coords: (..., L, 3) with (t, h, w) -> rope (..., L, 6 * n_bands),
        laid out as [sin | cos] for apply_rotary_emb_torch."""
        proj = coords.float().unsqueeze(-1) * self.bands  # (..., L, 3, B)
        return torch.cat((proj.sin(), proj.cos()), dim=-2).flatten(-2)


#################################################################################
#                       SwiGLU7 MLP (from daVinci-MagiHuman)                    #
#################################################################################

def swiglu7(x, alpha: float = 1.702, limit: float = 7.0, out_dtype: Optional[torch.dtype] = None):
    out_dtype = x.dtype if out_dtype is None else out_dtype
    return fused_swiglu7(x, alpha=alpha, limit=limit).to(out_dtype)


class SwiGLU7MLP(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        intermediate_size = int(hidden_size * 4 * 2 / 3) // 4 * 4  # MagiHuman sizing
        self.up_gate_proj = nn.Linear(hidden_size, intermediate_size * 2, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x):
        return self.down_proj(swiglu7(self.up_gate_proj(x)))


class MoESwiGLU7MLP(nn.Module):
    """Top-k routed SwiGLU experts plus one always-on shared expert."""

    def __init__(self, hidden_size, num_experts=8, top_k=2, shared_expert=True):
        super().__init__()
        if num_experts < 1:
            raise ValueError("num_experts must be positive")
        if not 1 <= top_k <= num_experts:
            raise ValueError(f"top_k must be in [1, {num_experts}], got {top_k}")
        if not shared_expert:
            raise ValueError("MoE requires one shared expert")
        self.num_experts = num_experts
        self.top_k = top_k
        self.router = nn.Linear(hidden_size, num_experts, bias=False)
        self.experts = nn.ModuleList([SwiGLU7MLP(hidden_size) for _ in range(num_experts)])
        self.shared_expert = SwiGLU7MLP(hidden_size)

    def forward(self, x, token_mask=None, return_router_stats=False):
        shape = x.shape
        flat = x.reshape(-1, shape[-1])
        valid = (
            torch.ones(flat.shape[0], device=x.device, dtype=torch.bool)
            if token_mask is None else token_mask.expand(shape[:-1]).reshape(-1)
        )

        # Routing stays in fp32 even under bf16 autocast. Renormalizing the
        # selected gates makes each token's routed contribution sum to one.
        probs = self.router(flat).float().softmax(dim=-1)
        top_weights, top_indices = probs.topk(self.top_k, dim=-1)
        top_weights = top_weights / top_weights.sum(dim=-1, keepdim=True)

        # Linear autocast can produce bf16 expert outputs even when the residual
        # stream is fp32. Use the shared expert to establish the accumulation
        # dtype so index_add never mixes fp32 and bf16.
        shared = self.shared_expert(flat)
        routed = torch.zeros_like(shared)
        for expert_idx, expert in enumerate(self.experts):
            token_idx, slot_idx = torch.where(
                (top_indices == expert_idx) & valid.unsqueeze(-1)
            )
            expert_out = expert(flat.index_select(0, token_idx))
            weighted = expert_out * top_weights[token_idx, slot_idx, None].to(expert_out.dtype)
            routed = routed.index_add(0, token_idx, weighted)

        out = shared + routed
        if not return_router_stats:
            return out.reshape(shape)

        # Switch-style balancing: importance is differentiable through all
        # router probabilities; load is the (non-differentiable) Top-k usage.
        valid_f = valid.float()
        n_valid = valid_f.sum().clamp(min=1.0)
        importance = (probs * valid_f.unsqueeze(-1)).sum(dim=0) / n_valid
        assignments = torch.stack(
            [((top_indices == i).float() * valid_f.unsqueeze(-1)).sum()
             for i in range(self.num_experts)]
        )
        load = assignments / (n_valid * self.top_k)
        aux_loss = self.num_experts * (importance * load).sum()
        return out.reshape(shape), aux_loss, load


#################################################################################
#                Attention with q/k RMSNorm and 3D RoPE                         #
#################################################################################

try:  # FlashAttention-4 (CuTe DSL, Hopper/Blackwell); pure-Python JIT package
    from flash_attn.cute import flash_attn_func as _fa4_func
except ImportError:
    _fa4_func = None


def linear_attention(q, k, v, eps=1e-6):
    """Non-causal kernelized linear attention (feature map elu(x) + 1),
    O(L * d^2) instead of softmax's O(L^2 * d). q, k, v: (b, h, l, d).

    Matmuls stay in the input dtype (tensor cores accumulate in fp32);
    only the normalizer is computed in fp32 to avoid bf16 overflow."""
    q = F.elu(q) + 1
    k = F.elu(k) + 1
    kv = torch.einsum("bhld,bhle->bhde", k, v)                              # (b, h, d, d)
    z = torch.einsum("bhld,bhd->bhl", q.float(), k.sum(dim=2).float())      # (b, h, l)
    return torch.einsum("bhld,bhde->bhle", q, kv) / (z.unsqueeze(-1).to(v.dtype) + eps)


class RopeAttention(nn.Module):
    """Joint attention over the whole sequence, MagiHuman style:
    bias-free qkv/proj, per-head q/k RMSNorm, rotary embedding on q/k.

    `attention` selects the kernel (all three share the same parameters, so
    checkpoints are structurally interchangeable):
    - "flash" (default): exact softmax attention via FlashAttention-4
      (flash-attn-4 package, Blackwell/Hopper); falls back to SDPA auto if the
      package is unavailable. fp32 inputs are computed in bf16 and cast back.
    - "softmax": exact softmax attention, SDPA auto backend selection.
    - "linear": kernelized linear attention, O(L) instead of O(L^2)."""

    def __init__(
        self,
        hidden_size,
        num_heads,
        attention="flash",
        attention_head_dim=None,
    ):
        super().__init__()
        if attention_head_dim is None:
            assert hidden_size % num_heads == 0
        assert attention in ("flash", "softmax", "linear"), f"unknown attention {attention!r}"
        self.num_heads = num_heads
        self.head_dim = (
            hidden_size // num_heads if attention_head_dim is None else attention_head_dim
        )
        self.inner_dim = num_heads * self.head_dim
        self.expanded = attention_head_dim is not None
        self.attention = attention

        if self.expanded:
            self.to_q = nn.Linear(hidden_size, self.inner_dim, bias=False)
            self.to_k = nn.Linear(hidden_size, self.inner_dim, bias=False)
            self.to_v = nn.Linear(hidden_size, self.inner_dim, bias=False)
            self.to_out = nn.Linear(self.inner_dim, hidden_size, bias=False)
        else:
            self.linear_qkv = nn.Linear(hidden_size, hidden_size * 3, bias=False)
            self.linear_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.q_norm = nn.RMSNorm(self.head_dim, eps=1e-6)
        self.k_norm = nn.RMSNorm(self.head_dim, eps=1e-6)

    def forward(self, x, rope, cu_seqlens=None, max_seqlen=None):
        b, l, _ = x.shape
        if self.expanded:
            q, k, v = (
                rearrange(proj(x), "b l (h d) -> b l h d", h=self.num_heads)
                for proj in (self.to_q, self.to_k, self.to_v)
            )
            project = self.to_out
        else:
            qkv = self.linear_qkv(x)
            q, k, v = rearrange(
                qkv,
                "b l (three h d) -> three b l h d",
                three=3,
                h=self.num_heads,
            )
            project = self.linear_proj

        sin_emb, cos_emb = rope.tensor_split(2, -1)
        fused_qk = fused_qk_norm_rope(
            q,
            k,
            self.q_norm.weight,
            self.k_norm.weight,
            cos_emb,
            sin_emb,
            eps=self.q_norm.eps,
        )
        if fused_qk is None:
            q = apply_rotary_emb_torch(self.q_norm(q), cos_emb, sin_emb)
            k = apply_rotary_emb_torch(self.k_norm(k), cos_emb, sin_emb)
        else:
            q, k = fused_qk

        # Packed bins: block-diagonal attention via FA4 varlen (custom op is
        # compile-safe). Dense path kept for single-segment sampling.
        if cu_seqlens is not None:
            dtype = q.dtype
            if dtype not in (torch.float16, torch.bfloat16):
                q, k, v = (t.to(torch.bfloat16) for t in (q, k, v))
            qkv_flat = (rearrange(t, "b l h d -> (b l) h d") for t in (q, k, v))
            out = varlen_attention(*qkv_flat, cu_seqlens, max_seqlen).to(dtype)
            return project(rearrange(out, "(b l) h d -> b l (h d)", b=b))

        # FA4's Python interface is not dynamo-traceable (graph breaks in every
        # block); under torch.compile fall through to SDPA, which is the same
        # math and gets fused by inductor.
        if self.attention == "flash" and _fa4_func is not None and not torch.compiler.is_compiling():
            # FA4 consumes (b, l, h, d) directly -- no transposes needed
            dtype = q.dtype
            if dtype not in (torch.float16, torch.bfloat16):
                q, k, v = (t.to(torch.bfloat16) for t in (q, k, v))
            out = _fa4_func(q, k, v, causal=False)[0].to(dtype)
            return project(out.reshape(b, l, -1))

        q, k, v = (rearrange(t, "b l h d -> b h l d") for t in (q, k, v))
        if self.attention == "linear":
            out = linear_attention(q, k, v)
        elif self.attention == "flash":  # FA4 unavailable: same math via SDPA
            dtype = q.dtype
            if dtype not in (torch.float16, torch.bfloat16):
                q, k, v = (t.to(torch.bfloat16) for t in (q, k, v))
            out = F.scaled_dot_product_attention(q, k, v).to(dtype)
        else:
            out = F.scaled_dot_product_attention(q, k, v)
        out = rearrange(out, "b h l d -> b l (h d)")
        return project(out)


class ContextRefinerBlock(nn.Module):
    """Dense H3-style text-token refiner without timestep conditioning."""

    def __init__(
        self,
        hidden_size,
        num_heads,
        attention="flash",
        attention_head_dim=None,
    ):
        super().__init__()
        self.norm1 = nn.RMSNorm(hidden_size, eps=1e-6)
        self.attn = RopeAttention(
            hidden_size,
            num_heads,
            attention=attention,
            attention_head_dim=attention_head_dim,
        )
        self.norm2 = nn.RMSNorm(hidden_size, eps=1e-6)
        self.mlp = SwiGLU7MLP(hidden_size)

    def forward(self, x, rope):
        x = x + self.attn(self.norm1(x), rope)
        return x + self.mlp(self.norm2(x))


#################################################################################
#          mHC: Manifold-Constrained Hyper-Connections (DeepSeek)               #
#################################################################################

def sinkhorn_knopp(logits, iters=8):
    """Project (..., n, n) logits onto the Birkhoff polytope (doubly stochastic
    matrices): elementwise exp, then alternate row/column normalization."""
    m = torch.exp(logits)
    for _ in range(iters):
        m = m / m.sum(dim=-1, keepdim=True)
        m = m / m.sum(dim=-2, keepdim=True)
    return m


class MHCResidual(nn.Module):
    """One mHC-wrapped sublayer connection (mHC, arXiv:2512.24880).

    The residual state carries n streams (B, L, n, C). Per token:
        H_pre  = sigmoid(a_pre * proj(RMSNorm(vec x)) + b_pre)        (n,)
        H_post = 2 sigmoid(...)                                       (n,)
        H_res  = SinkhornKnopp(a_res * proj(...) + b_res)             (n, n)
        x_out  = H_res @ x + H_post^T * f(H_pre @ x)
    H_res is doubly stochastic, so stream mixing is a convex combination
    (norm-preserving, composition-closed) -- the identity mapping property of
    plain residuals is retained. Init makes the whole connection *exactly*
    equivalent to the standard pre-norm residual: sigma(b_pre) = 1/n reads the
    stream mean, H_post = 1 writes f() to every stream, and any doubly
    stochastic H_res is an identity on identical streams.
    """

    def __init__(self, hidden_size, n, sinkhorn_iters=8):
        super().__init__()
        self.n = n
        self.sinkhorn_iters = sinkhorn_iters
        # Consolidated dynamic projection: [pre (n) | post (n) | res (n^2)].
        # The paper's RMSNorm before phi carries no affine here: a per-channel
        # scale would be absorbed into the bias-free phi.weight anyway.
        self.phi = nn.Linear(n * hidden_size, n * n + 2 * n, bias=False)
        self.alpha = nn.Parameter(torch.full((3,), 0.01))              # (pre, post, res) gates
        self.b_pre = nn.Parameter(torch.full((n,), -math.log(n - 1.0)))  # sigmoid -> 1/n
        self.b_post = nn.Parameter(torch.zeros(n))                     # 2 sigmoid(0) = 1
        self.b_res = nn.Parameter(4.0 * torch.eye(n))                  # Sinkhorn(exp) ~ I
        nn.init.zeros_(self.phi.weight)

    def forward(self, xs, f):
        """xs: (B, L, n, C) streams; f: sublayer callable (B, L, C) -> (B, L, C)."""
        n = self.n
        flat = xs.flatten(-2)
        # phi(RMSNorm(flat)) == phi(flat) * inv_rms(flat) for an affine-free
        # norm and bias-free phi: run the GEMM first and scale its tiny (n^2 +
        # 2n)-dim output by the per-token scalar. This avoids materializing
        # the normalized (B, L, n*C) tensor, whose fwd+bwd reduction kernels
        # dominated the mHC overhead.
        inv_rms = torch.rsqrt(flat.float().pow(2).mean(-1, keepdim=True) + 1e-6)
        d = self.phi(flat) * inv_rms.to(flat.dtype)
        d_pre, d_post, d_res = d.split([n, n, n * n], dim=-1)
        h_pre = torch.sigmoid(self.alpha[0].to(xs.dtype) * d_pre + self.b_pre.to(xs.dtype))
        h_post = 2 * torch.sigmoid(self.alpha[1].to(xs.dtype) * d_post + self.b_post.to(xs.dtype))
        logits = self.alpha[2] * d_res.float().unflatten(-1, (n, n)) + self.b_res.float()
        h_res = sinkhorn_knopp(logits, self.sinkhorn_iters).to(xs.dtype)

        u = (h_pre.unsqueeze(-1) * xs).sum(dim=-2)                    # (B, L, C)
        y = f(u)
        # H_res @ xs written as n^2 fused multiply-adds: cuBLAS treats the
        # (B*L, n, n) x (B*L, n, C) batched GEMM with M=K=4 as thousands of
        # tiny matmuls (~10x slower); this form fuses into one elementwise
        # kernel under torch.compile.
        streams = xs.unbind(dim=-2)
        mixed = torch.stack(
            [sum(h_res[..., i, j, None] * streams[j] for j in range(n)) for i in range(n)],
            dim=-2,
        )
        return mixed + h_post.unsqueeze(-1) * y.unsqueeze(-2)


#################################################################################
#                 Transformer block with per-token adaLN-Zero                   #
#################################################################################

def modulate_per_token(x, shift, scale):
    return x * (1 + scale) + shift


class VideoDiTBlock(nn.Module):
    """MagiHuman-style layer (RoPE attention + SwiGLU7 MLP) with Self-Flow's
    per-token adaLN-Zero conditioning.

    `adaln` selects how the 6 modulation vectors are produced:
    - "per_block": each block owns a hidden -> 6*hidden Linear (DiT classic).
      ~33% of model params and a large per-token GEMM in every block.
    - "single" (PixArt-alpha style): the DiT computes one shared modulation
      tensor from the timestep embedding; each block only adds a learnable
      (6*hidden,) offset. Same per-token conditioning, 24x fewer adaLN GEMMs.
    """

    def __init__(
        self, hidden_size, num_heads, attention="flash", adaln="per_block", mhc=0,
        moe_num_experts=0, moe_top_k=2, moe_shared_expert=True,
        attention_head_dim=None, modality_adaln=False,
    ):
        super().__init__()
        self.norm1 = nn.RMSNorm(hidden_size, eps=1e-6, elementwise_affine=False)
        self.attn = RopeAttention(
            hidden_size,
            num_heads,
            attention=attention,
            attention_head_dim=attention_head_dim,
        )
        self.norm2 = nn.RMSNorm(hidden_size, eps=1e-6, elementwise_affine=False)
        self.is_moe = moe_num_experts > 0
        self.mlp = (
            MoESwiGLU7MLP(hidden_size, moe_num_experts, moe_top_k, moe_shared_expert)
            if self.is_moe else SwiGLU7MLP(hidden_size)
        )
        self.adaln = adaln
        if adaln == "single":
            self.scale_shift_table = nn.Parameter(torch.zeros(6 * hidden_size))
        else:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size, bias=True)
            )
        self.modality_adaln = modality_adaln
        if modality_adaln:
            # VIDEO/AUDIO/TEXT get small per-block offsets on top of the shared
            # timestep/text modulation. PAD receives no modality offset.
            self.modality_scale_shift = nn.Parameter(torch.zeros(3, 6 * hidden_size))
        self.mhc = mhc
        if mhc > 1:  # one connection per sublayer (attention / MLP)
            self.mhc_attn = MHCResidual(hidden_size, mhc)
            self.mhc_mlp = MHCResidual(hidden_size, mhc)

    def forward(
        self, x, c, rope, cu_seqlens=None, max_seqlen=None,
        token_mask=None, modality_ids=None, return_router_stats=False,
    ):
        # c: timestep embedding (per_block) or precomputed shared modulation (single)
        mod = c + self.scale_shift_table if self.adaln == "single" else self.adaLN_modulation(c)
        if self.modality_adaln:
            if modality_ids is None:
                raise ValueError("modality_ids are required when modality_adaln is enabled")
            valid_modality = modality_ids < PAD
            safe_ids = modality_ids.clamp(min=VIDEO, max=TEXT)
            modality_offset = F.embedding(safe_ids, self.modality_scale_shift)
            mod = mod + modality_offset * valid_modality.unsqueeze(-1)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = mod.chunk(6, dim=-1)
        f_attn = lambda u: gate_msa * self.attn(
            fused_rms_modulate(u, shift_msa, scale_msa, eps=self.norm1.eps),
            rope, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen,
        )
        router_aux = router_load = None

        def run_mlp(u):
            nonlocal router_aux, router_load
            u = fused_rms_modulate(u, shift_mlp, scale_mlp, eps=self.norm2.eps)
            if self.is_moe:
                result = self.mlp(u, token_mask, return_router_stats=return_router_stats)
                if return_router_stats:
                    out, router_aux, router_load = result
                else:
                    out = result
            else:
                out = self.mlp(u)
            return out

        f_mlp = lambda u: gate_mlp * run_mlp(u)

        if self.mhc > 1:  # x: (B, L, n, C) hyper-connection streams
            x = self.mhc_attn(x, f_attn)
            x = self.mhc_mlp(x, f_mlp)
            return (x, router_aux, router_load) if return_router_stats else x
        attn_out = self.attn(
            fused_rms_modulate(x, shift_msa, scale_msa, eps=self.norm1.eps),
            rope, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen,
        )
        x = fused_gate_residual(x, attn_out, gate_msa)
        mlp_out = run_mlp(x)
        x = fused_gate_residual(x, mlp_out, gate_mlp)
        return (x, router_aux, router_load) if return_router_stats else x


class VideoFinalLayer(nn.Module):
    """Per-token adaLN final layer projecting back to video token channels."""

    def __init__(self, hidden_size, out_dim):
        super().__init__()
        self.norm_final = nn.RMSNorm(hidden_size, eps=1e-6, elementwise_affine=False)
        self.linear = nn.Linear(hidden_size, out_dim, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        x = fused_rms_modulate(x, shift, scale, eps=self.norm_final.eps)
        return self.linear(x)


#################################################################################
#                              Video DiT                                        #
#################################################################################

class TextToVideoDiT(nn.Module):
    """Text-conditioned DiT over a joint token sequence.

    With `audio_in_channels` set, the noised sequence carries two generation
    modalities packed into one tensor: [video tokens | audio tokens], each
    zero-padded on the channel dim to a common width. The 4th column of
    `x_ids` marks the modality (0 = video, 1 = audio); each modality has its
    own embedder and output head, sharing the transformer blocks.
    """

    def __init__(
        self,
        video_in_channels=192,   # z_dim 48 * patch 2 * 2
        text_in_channels=512,    # CLIP hidden size
        audio_in_channels=None,  # Stable Audio latent dim (64); None = video only
        hidden_size=384,
        depth=12,
        num_heads=6,
        attention="flash",       # "flash" | "softmax" | "linear"
        adaln="per_block",       # "per_block" | "single" (PixArt-alpha shared adaLN)
        mhc=0,                   # 0/1 = plain residual; n>=2 = mHC with n streams
        pooled_text=False,       # add pooled text to the adaLN vector (SD3/Flux)
        moe_num_experts=0,       # 0 = dense MLP; otherwise routed expert count
        moe_top_k=2,
        moe_shared_expert=True,
        attention_head_dim=None, # None = residual-width legacy attention
        mm_rope_freq_dim=None,   # fixed frequencies per T/H/W axis when set
        mm_rope_theta=10000.0,
        modality_adaln=False,
        context_refiner_layers=0,
    ):
        super().__init__()
        self.video_in_channels = video_in_channels
        self.audio_in_channels = audio_in_channels
        self.adaln = adaln
        self.mhc = mhc
        self.pooled_text = pooled_text
        self.moe_num_experts = moe_num_experts
        # Static video/audio token split (set by the trainer / inference from
        # TokenLayout.l_v) so the forward pass is fully torch.compile-traceable
        self.seq_l_v = None
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.attention_head_dim = (
            hidden_size // num_heads if attention_head_dim is None else attention_head_dim
        )
        self.modality_adaln = modality_adaln
        self.context_refiner_layers = context_refiner_layers

        # Per-modality embedders (MagiHuman Adapter)
        self.video_embedder = nn.Linear(video_in_channels, hidden_size, bias=True)
        self.text_embedder = nn.Linear(text_in_channels, hidden_size, bias=True)
        if audio_in_channels:
            self.audio_embedder = nn.Linear(audio_in_channels, hidden_size, bias=True)
        self.rope = Rope3D(
            self.attention_head_dim,
            temperature=mm_rope_theta,
            freq_dim=mm_rope_freq_dim,
        )
        self.context_refiner = nn.ModuleList([
            ContextRefinerBlock(
                hidden_size,
                num_heads,
                attention=attention,
                attention_head_dim=attention_head_dim,
            )
            for _ in range(context_refiner_layers)
        ])
        if context_refiner_layers:
            self.context_refiner_norm = nn.RMSNorm(hidden_size, eps=1e-6)

        # Timestep conditioning (Self-Flow)
        self.t_embedder = TimestepEmbedder(hidden_size)
        if pooled_text:
            # Text also reaches modulation as one vector per sample, the way
            # SD3/Flux/PixArt do it. Without this the only route from caption to
            # video token is attention over 48 text tokens in a ~2400-token
            # sequence, which a run this small never learns to use.
            self.text_pooler = nn.Sequential(
                nn.LayerNorm(text_in_channels),
                nn.Linear(text_in_channels, hidden_size, bias=True),
                nn.SiLU(),
                nn.Linear(hidden_size, hidden_size, bias=True),
            )
        if adaln == "single":
            self.adaln_head = nn.Sequential(
                nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size, bias=True)
            )

        self.blocks = nn.ModuleList([
            VideoDiTBlock(
                hidden_size, num_heads, attention=attention, adaln=adaln, mhc=mhc,
                moe_num_experts=moe_num_experts, moe_top_k=moe_top_k,
                moe_shared_expert=moe_shared_expert,
                attention_head_dim=attention_head_dim,
                modality_adaln=modality_adaln,
            )
            for _ in range(depth)
        ])
        self.final_layer = VideoFinalLayer(hidden_size, video_in_channels)
        if audio_in_channels:
            self.audio_final_layer = VideoFinalLayer(hidden_size, audio_in_channels)

        # Self-distillation projector (Self-Flow)
        self.projector = SimpleHead(hidden_size, hidden_size)

        self.initialize_weights()

    def _refine_text(self, text, max_seqs=1):
        """Project cached features and refine each caption independently."""
        h = self.text_embedder(text)
        if not self.context_refiner:
            return h

        b, total_len, hidden = h.shape
        text_len = total_len // max_seqs
        h = h.reshape(b * max_seqs, text_len, hidden)
        pos = torch.arange(text_len, device=h.device, dtype=torch.float32)
        zeros = torch.zeros_like(pos)
        coords = torch.stack((pos, zeros, zeros), dim=-1)
        coords = coords.unsqueeze(0).expand(h.shape[0], -1, -1)
        rope = self.rope(coords)
        for block in self.context_refiner:
            h = block(h, rope)
        h = self.context_refiner_norm(h)
        return h.reshape(b, total_len, hidden)

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # mHC dynamic projections start at zero (the xavier pass above touched
        # them); with the static biases this makes every connection an exact
        # standard residual at init.
        for m in self.modules():
            if isinstance(m, MHCResidual):
                nn.init.zeros_(m.phi.weight)

        # The pooled-text path starts at exactly zero, so a run with it enabled
        # begins from the same function as one without it.
        if self.pooled_text:
            nn.init.constant_(self.text_pooler[-1].weight, 0)
            nn.init.constant_(self.text_pooler[-1].bias, 0)

        # adaLN-Zero init: gates start at zero in both modes ("single" tables
        # are created as zeros; zeroing the shared head keeps blocks identity)
        if self.adaln == "single":
            nn.init.constant_(self.adaln_head[-1].weight, 0)
            nn.init.constant_(self.adaln_head[-1].bias, 0)
        else:
            for block in self.blocks:
                nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
                nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        final_layers = [self.final_layer]
        if self.audio_in_channels:
            final_layers.append(self.audio_final_layer)
        for fl in final_layers:
            nn.init.constant_(fl.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(fl.adaLN_modulation[-1].bias, 0)
            nn.init.constant_(fl.linear.weight, 0)
            nn.init.constant_(fl.linear.bias, 0)

    def _embed_noised(self, x, modality):
        """Per-row video/audio embedding; PAD rows stay zero."""
        b, l, _ = x.shape
        h = x.new_zeros(b, l, self.hidden_size)
        v_mask = (modality == VIDEO).unsqueeze(-1)
        h = h + v_mask * self.video_embedder(x[..., : self.video_in_channels])
        if self.audio_in_channels:
            a_mask = (modality == AUDIO).unsqueeze(-1)
            h = h + a_mask * self.audio_embedder(x[..., : self.audio_in_channels])
        return h

    def _pooled_packed(self, text, plan: PackPlan):
        """Each joint token's own sample's pooled text, laid out like the blocks
        see it: gathered through `plan.joint_src` exactly as `h` is, so the pad
        tail reads a zero row instead of some other sample's caption."""
        b = text.shape[0]
        text_len = text.shape[1] // plan.max_seqs
        pooled = self.text_pooler(
            text.reshape(b, plan.max_seqs, text_len, -1).mean(dim=2)
        )                                                    # (B, max_seqs, H)
        pooled = torch.cat([pooled, pooled.new_zeros(b, 1, pooled.shape[-1])], dim=1)
        device = text.device
        src_segment = torch.cat([
            plan.segment,
            torch.arange(plan.max_seqs, device=device).repeat_interleave(text_len),
            torch.tensor([plan.max_seqs], device=device),
        ])
        return pooled[:, src_segment[plan.joint_src]]

    def _heads(self, h_x, c_x, modality, width):
        """Per-modality output heads scattered back onto the noised grid."""
        out = h_x.new_zeros(h_x.shape[0], h_x.shape[1], width)
        v_mask = (modality == VIDEO).unsqueeze(-1)
        v_pred = self.final_layer(h_x, c_x)
        out = out + v_mask * F.pad(v_pred, (0, width - v_pred.shape[-1]))
        if self.audio_in_channels:
            a_mask = (modality == AUDIO).unsqueeze(-1)
            a_pred = self.audio_final_layer(h_x, c_x)
            out = out + a_mask * F.pad(a_pred, (0, width - a_pred.shape[-1]))
        return out

    def _forward_packed(
        self, x, t, text, plan: PackPlan,
        return_features=False, return_raw_features=False, features_only=False,
        return_router_stats=False,
    ):
        """Packed-bin forward: joint sequence via plan.joint_src / plan.x_slot."""
        assert not (return_raw_features and return_features)
        b, x_len, width = x.shape
        joint_len = plan.joint_len
        modality = plan.modality  # (x_len,)

        h_x = self._embed_noised(x, modality)
        h_text = self._refine_text(text, plan.max_seqs)
        zero = h_x.new_zeros(b, 1, self.hidden_size)
        source = torch.cat([h_x, h_text, zero], dim=1)  # (B, x_len + text + 1, H)
        h = source[:, plan.joint_src]  # (B, joint_len, H)

        rope = self.rope(plan.x_ids[..., :3].float().unsqueeze(0).expand(b, -1, -1))

        if t.ndim == 1:
            t_x = t.unsqueeze(1).expand(b, x_len)
        else:
            t_x = t
        # Text tokens (and the gather-source zero row) are clean (t=1); PAD
        # noised slots already carry t=0 from the objective.
        t_source = torch.cat([
            t_x,
            torch.ones(b, text.shape[1], device=x.device, dtype=t_x.dtype),
            torch.ones(b, 1, device=x.device, dtype=t_x.dtype),
        ], dim=1)
        t_all = t_source[:, plan.joint_src]
        c = self.t_embedder(t_all.reshape(-1)).reshape(b, joint_len, -1)
        if self.pooled_text:
            c = c + self._pooled_packed(text, plan)
        cond = self.adaln_head(c) if self.adaln == "single" else c

        if self.mhc > 1:
            h = h.unsqueeze(-2).repeat(1, 1, self.mhc, 1)
        collapse = (lambda z: z.mean(dim=-2)) if self.mhc > 1 else (lambda z: z)

        zs = None
        router_aux = h.new_zeros((), dtype=torch.float32)
        router_load = h.new_zeros(self.moe_num_experts, dtype=torch.float32)
        token_mask = plan.x_ids[:, 3] != PAD
        modality_ids = plan.x_ids[:, 3].long()
        for i, block in enumerate(self.blocks):
            result = block(
                h, cond, rope, cu_seqlens=plan.cu_seqlens, max_seqlen=plan.max_seqlen,
                token_mask=token_mask, modality_ids=modality_ids,
                return_router_stats=return_router_stats,
            )
            if return_router_stats:
                h, block_aux, block_load = result
                router_aux = router_aux + block_aux
                router_load = router_load + block_load
            else:
                h = result
            if (i + 1) == return_features:
                zs = self.projector(collapse(h)[:, plan.x_slot])
            elif (i + 1) == return_raw_features:
                zs = collapse(h)[:, plan.x_slot]
            if features_only and zs is not None:
                return None, zs
        h = collapse(h)
        h_x = h[:, plan.x_slot]
        c_x = c[:, plan.x_slot]
        out = self._heads(h_x, c_x, modality, width)
        if return_features or return_raw_features:
            result = (out, zs)
        else:
            result = out
        if return_router_stats:
            scale = 1.0 / len(self.blocks)
            if return_features or return_raw_features:
                return result[0], result[1], router_aux * scale, router_load * scale
            return result, router_aux * scale, router_load * scale
        return result

    def _forward(self, x, t, text, x_ids, return_features=False, return_raw_features=False,
                 features_only=False, return_router_stats=False):
        """Dense (single-segment) forward used by sampling.

        Args:
            x: (B, L_x, C) noised tokens: video only, or [video | audio] with
               each modality zero-padded on the channel dim to a common width
            t: (B,) or (B, L_x) timesteps, 0 = noise .. 1 = data
            text: (B, L_t, C_t) text token features (CLIP last hidden state)
            x_ids: (B, L_x, 4) (t, h, w, modality) coords; modality 0 = video, 1 = audio
            features_only: stop right after the feature layer and skip the
                remaining blocks + output heads (EMA-teacher fast path)
        """
        assert not (return_raw_features and return_features)
        b, l_x, _ = x.shape
        l_t = text.shape[1]

        # Joint sequence: [video (; audio); text] (MagiHuman single-stream design)
        if self.audio_in_channels:
            # Prefer the static split injected by the caller (TokenLayout.l_v):
            # reading it from x_ids is data-dependent and breaks torch.compile's graph
            l_v = self.seq_l_v if self.seq_l_v is not None else int((x_ids[0, :, 3] == 0).sum())
            h_video = self.video_embedder(x[:, :l_v, : self.video_in_channels])
            h_audio = self.audio_embedder(x[:, l_v:, : self.audio_in_channels])
            h = torch.cat([h_video, h_audio, self._refine_text(text)], dim=1)
        else:
            l_v = l_x
            h_video = self.video_embedder(x)
            h = torch.cat([h_video, self._refine_text(text)], dim=1)

        # 3D RoPE coords: video/audio from x_ids; text placed after the
        # noised-token time range on the t axis (h = w = 0)
        v_coords = x_ids[..., :3].float()
        t_max = v_coords[..., 0].amax(dim=1, keepdim=True)  # (B, 1)
        txt_time = t_max + 1 + torch.arange(l_t, device=x.device).float().unsqueeze(0)
        t_coords = torch.stack(
            [txt_time, torch.zeros_like(txt_time), torch.zeros_like(txt_time)], dim=-1
        )
        rope = self.rope(torch.cat([v_coords, t_coords], dim=1))

        # Per-token timestep conditioning; text tokens are always clean (t=1)
        if t.ndim == 1:
            t_x = t.unsqueeze(1).expand(b, l_x)
        else:
            t_x = t
        t_all = torch.cat([t_x, torch.ones(b, l_t, device=x.device, dtype=t_x.dtype)], dim=1)
        c = self.t_embedder(t_all.reshape(-1)).reshape(b, l_x + l_t, -1)
        if self.pooled_text:
            c = c + self.text_pooler(text.mean(dim=1)).unsqueeze(1)
        # "single": one shared modulation for all blocks (each adds its table)
        cond = self.adaln_head(c) if self.adaln == "single" else c

        if self.mhc > 1:  # replicate into n identical hyper-connection streams
            h = h.unsqueeze(-2).repeat(1, 1, self.mhc, 1)
        collapse = (lambda z: z.mean(dim=-2)) if self.mhc > 1 else (lambda z: z)

        zs = None
        router_aux = h.new_zeros((), dtype=torch.float32)
        router_load = h.new_zeros(self.moe_num_experts, dtype=torch.float32)
        token_mask = torch.ones((b, l_x + l_t), device=h.device, dtype=torch.bool)
        text_modality = torch.full(
            (b, l_t), TEXT, device=x.device, dtype=torch.long
        )
        modality_ids = torch.cat([x_ids[..., 3].long(), text_modality], dim=1)
        for i, block in enumerate(self.blocks):
            result = block(
                h, cond, rope, token_mask=token_mask, modality_ids=modality_ids,
                return_router_stats=return_router_stats,
            )
            if return_router_stats:
                h, block_aux, block_load = result
                router_aux = router_aux + block_aux
                router_load = router_load + block_load
            else:
                h = result
            if (i + 1) == return_features:
                zs = self.projector(collapse(h)[:, :l_x])
            elif (i + 1) == return_raw_features:
                zs = collapse(h)[:, :l_x]
            if features_only and zs is not None:
                return None, zs
        h = collapse(h)

        # Per-modality heads (MagiHuman: final_norm_video + final_linear_video);
        # outputs are zero-padded back to the packed channel width
        out = self.final_layer(h[:, :l_v], c[:, :l_v])
        if self.audio_in_channels:
            out_audio = self.audio_final_layer(h[:, l_v:l_x], c[:, l_v:l_x])
            width = x.shape[-1]
            out = torch.cat([
                F.pad(out, (0, width - out.shape[-1])),
                F.pad(out_audio, (0, width - out_audio.shape[-1])),
            ], dim=1)

        if return_features or return_raw_features:
            result = (out, zs)
        else:
            result = out
        if return_router_stats:
            scale = 1.0 / len(self.blocks)
            if return_features or return_raw_features:
                return result[0], result[1], router_aux * scale, router_load * scale
            return result, router_aux * scale, router_load * scale
        return result

    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        vector: torch.Tensor,
        x_ids: torch.Tensor = None,
        pack_plan: PackPlan = None,
        return_features=False,
        return_raw_features=False,
        features_only=False,
        return_router_stats=False,
    ):
        """Self-Flow compatibility convention: flips timesteps, negates output,
        so sampling.py's denoise_loop works unchanged. `vector` carries the
        text token features — (B, L_t, C_t) for dense sampling, or
        (B, max_seqs * L_t, C_t) when `pack_plan` is set."""
        timesteps = 1 - timesteps

        run = (
            (lambda: self._forward_packed(
                x, timesteps, vector, pack_plan,
                return_features=return_features,
                return_raw_features=return_raw_features,
                features_only=features_only,
                return_router_stats=return_router_stats,
            ))
            if pack_plan is not None else
            (lambda: self._forward(
                x, timesteps, vector, x_ids,
                return_features=return_features,
                return_raw_features=return_raw_features,
                features_only=features_only,
                return_router_stats=return_router_stats,
            ))
        )

        if return_router_stats:
            if return_features or return_raw_features:
                out, zs, router_aux, router_load = run()
                return (None if out is None else -out), zs, router_aux, router_load
            out, router_aux, router_load = run()
            return -out, router_aux, router_load
        if return_features or return_raw_features:
            out, zs = run()
            return (None if out is None else -out), zs
        return -run()
