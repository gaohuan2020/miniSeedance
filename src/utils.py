"""Token sequence geometry shared by training and inference."""

import itertools
from typing import NamedTuple, Optional

import torch
import torch.nn.functional as F
from einops import rearrange

# x_ids modality column
VIDEO, AUDIO, TEXT, PAD = 0, 1, 2, 3


class TokenLayout:
    """Geometry of the packed [video | audio] token sequence.

    Owns everything both training and inference need to agree on: video
    patchify/unpatchify, channel zero-padding to a common width, position ids
    with the modality column (0 = video, 1 = audio), and modality slices for
    per-modality losses. Video tokens are row-major over (t, h, w).

    `patch_size_t` groups consecutive latent frames into one token along the
    time axis (spatial `patch_size` works on h/w as before). Latent T that is
    not divisible by patch_size_t is zero-padded at the end; unpack crops the
    pad back off. Pad tokens take part in attention and the loss (they learn
    to predict zeros), which costs a little capacity but keeps the packed
    tensor rectangular.
    """

    def __init__(self, latent_shape, patch_size, patch_size_t=1, audio_channels=0, audio_len=0):
        self.latent_shape = tuple(latent_shape)  # (C, T, H, W)
        self.patch_size = patch_size
        self.patch_size_t = patch_size_t
        self.audio_channels = audio_channels
        self.audio_len = audio_len

        c, t, h, w = self.latent_shape
        p, pt = patch_size, patch_size_t
        assert h % p == 0 and w % p == 0
        self.t_pad = (-t) % pt
        self.t_tokens = (t + self.t_pad) // pt
        self.token_dim = c * pt * p * p
        self.l_v = self.t_tokens * (h // p) * (w // p)
        self.width = max(self.token_dim, audio_channels) if audio_channels else self.token_dim
        self.seq_len = self.l_v + audio_len
        self._joint_ids = {}

    @property
    def has_audio(self):
        return self.audio_channels > 0

    def x_ids(self, device):
        """(seq_len, 4) coords (t, h, w, modality)."""
        c, t, h, w = self.latent_shape
        p = self.patch_size
        v_ids = torch.cartesian_prod(
            torch.arange(self.t_tokens), torch.arange(h // p), torch.arange(w // p),
            torch.arange(1),
        ).to(device).float()
        if not self.has_audio:
            return v_ids
        # Audio time coords in video token-frame units, aligned to the clip
        a_time = torch.arange(self.audio_len, device=device).float() * (self.t_tokens / self.audio_len)
        ones = torch.ones_like(a_time)
        a_ids = torch.stack([a_time, -ones, -ones, ones], dim=-1)
        return torch.cat([v_ids, a_ids], dim=0)

    def joint_ids(self, device, text_len):
        """(seq_len + text_len, 4) coords of one sample's whole attention segment.

        Text tokens sit after the noised tokens' time range on the t axis with
        h = w = 0, which is where the pre-packing forward pass placed them, so
        checkpoints keep their positional semantics.
        """
        key = (str(device), text_len)
        cached = self._joint_ids.get(key)
        if cached is not None:
            return cached
        ids = self.x_ids(device)
        t_text = ids[:, 0].max() + 1 + torch.arange(text_len, device=device).float()
        zeros = torch.zeros_like(t_text)
        text_ids = torch.stack([t_text, zeros, zeros, zeros + TEXT], dim=-1)
        joint = torch.cat([ids, text_ids], dim=0)
        self._joint_ids[key] = joint
        return joint

    def pack(self, video_latents, audio_latents=None):
        """(B, C, T, H, W) [+ (B, C_a, L_a)] -> (B, seq_len, width) tokens."""
        p, pt = self.patch_size, self.patch_size_t
        if self.t_pad:
            video_latents = F.pad(video_latents, (0, 0, 0, 0, 0, self.t_pad))
        x = rearrange(
            video_latents, "b c (t pt) (h p) (w q) -> b (t h w) (c pt p q)", pt=pt, p=p, q=p
        )
        if not self.has_audio:
            return x
        a = audio_latents.permute(0, 2, 1)  # (B, L_a, C_a)
        return torch.cat([
            F.pad(x, (0, self.width - self.token_dim)),
            F.pad(a, (0, self.width - self.audio_channels)),
        ], dim=1)

    def unpack(self, tokens):
        """(B, seq_len, width) -> ((B, C, T, H, W), (B, C_a, L_a) | None)."""
        c, t, h, w = self.latent_shape
        p, pt = self.patch_size, self.patch_size_t
        video = rearrange(
            tokens[:, : self.l_v, : self.token_dim],
            "b (t h w) (c pt p q) -> b c (t pt) (h p) (w q)",
            t=self.t_tokens, h=h // p, w=w // p, c=c, pt=pt, p=p, q=p,
        )
        if self.t_pad:
            video = video[:, :, :t]
        if not self.has_audio:
            return video, None
        audio = tokens[:, self.l_v :, : self.audio_channels].permute(0, 2, 1)
        return video, audio


class PackPlan(NamedTuple):
    """Index geometry of one packed bin of samples.

    Sequence packing puts several samples of different shapes into one flat
    sequence rather than padding them all up to a common rectangle. Each
    sample's text tokens are spliced in directly behind its own noised tokens so
    that one sample is one contiguous attention segment, and `cu_seqlens` stops
    attention at the segment boundaries.

    The noised token tensor the trainer and the sampler work with is *not* the
    joint sequence: it holds only the [video | audio] rows, packed back to back
    to a fixed budget `x_len`. `joint_src` gathers it (together with the text
    features and one zero row) into the joint sequence the blocks see, `x_slot`
    gathers the block outputs back. Every index tensor here has a static shape,
    so a bin holding four samples and one holding six share a compiled graph.

    `cu_seqlens` is None when each batch row already is exactly one segment
    (the sampling case), which lets attention take the plain dense path.

    Every field is a tensor except `max_seqlen` and `max_seqs`, which are
    dataset-wide constants: a plan is a torch.compile input, and an int or tuple
    field that tracked the bin contents would guard a fresh graph per bin
    composition. How many samples actually landed in the bin is therefore never
    a Python int — reading it back would both re-specialize the graph and cost a
    device sync every step.
    """

    x_ids: torch.Tensor                 # (L_joint, 4) coords (t, h, w, modality)
    joint_src: torch.Tensor             # (L_joint,) into [noised | text | zero row]
    x_slot: torch.Tensor                # (x_len,) joint position of each noised row
    modality: torch.Tensor              # (x_len,) VIDEO / AUDIO / PAD per noised row
    segment: torch.Tensor               # (x_len,) sample index per noised row
    cu_seqlens: Optional[torch.Tensor]  # (max_seqs + 2,) int32 joint segment bounds
    max_seqlen: int                     # static upper bound on one segment's length
    max_seqs: int                       # static bin capacity in samples

    @property
    def joint_len(self):
        return self.x_ids.shape[0]


def build_pack_plan(layouts, text_len, x_len, max_seqs, max_seqlen, device, batch=1):
    """Index geometry for a bin holding one sample per entry of `layouts`.

    `x_len`, `max_seqs` and `max_seqlen` are the *static* packing budget: they
    must not depend on what actually landed in the bin, or torch.compile
    re-specializes every step. `batch` replicates the segment bounds over a
    batch dimension (sampling packs one sample per row).
    """
    n = len(layouts)
    assert 0 < n <= max_seqs, f"{n} samples exceed the bin capacity of {max_seqs}"
    x_lens = [lo.seq_len for lo in layouts]
    assert sum(x_lens) <= x_len, f"{sum(x_lens)} tokens exceed the bin budget of {x_len}"

    joint_len = x_len + max_seqs * text_len
    seg_lens = [xl + text_len for xl in x_lens]
    x_off = list(itertools.accumulate(x_lens, initial=0))
    joint_off = list(itertools.accumulate(seg_lens, initial=0))

    ids = [lo.joint_ids(device, text_len) for lo in layouts]
    n_pad = joint_len - joint_off[-1]
    if n_pad:
        pad_ids = torch.zeros(n_pad, 4, device=device)
        pad_ids[:, 3] = PAD
        ids.append(pad_ids)
    x_ids = torch.cat(ids, dim=0)

    def as_index(seq):
        return torch.tensor(seq, device=device, dtype=torch.long)

    x_lens_t, x_off_t, joint_off_t = as_index(x_lens), as_index(x_off), as_index(joint_off)
    bounds_joint = joint_off_t[1:].contiguous()
    bounds_x = x_off_t[1:].contiguous()

    pos = torch.arange(joint_len, device=device)
    seg = torch.searchsorted(bounds_joint, pos, right=True)  # n on the pad tail
    seg_c = seg.clamp(max=n - 1)
    local = pos - joint_off_t[seg_c]
    in_text = local >= x_lens_t[seg_c]
    joint_src = torch.where(
        in_text,
        x_len + seg_c * text_len + (local - x_lens_t[seg_c]),
        x_off_t[seg_c] + local,
    )
    joint_src = torch.where(seg < n, joint_src, joint_len)  # the zero row

    x_pos = torch.arange(x_len, device=device)
    x_seg = torch.searchsorted(bounds_x, x_pos, right=True)
    x_seg_c = x_seg.clamp(max=n - 1)
    # Unused rows of the noised tensor read back a pad row of the joint
    # sequence; the loss masks them out via `modality`.
    x_slot = torch.where(
        x_seg < n, joint_off_t[x_seg_c] + (x_pos - x_off_t[x_seg_c]), joint_len - 1
    )

    cu_seqlens = None
    if max_seqs > 1 or n_pad:
        cu_seqlens = torch.full((max_seqs + 2,), joint_len, device=device, dtype=torch.int32)
        cu_seqlens[: n + 1] = joint_off_t.to(torch.int32)
        if batch > 1:
            offsets = torch.arange(batch, device=device, dtype=torch.int32) * joint_len
            # Duplicate bounds at the row seams become zero-length segments,
            # which FA4 skips.
            cu_seqlens = (cu_seqlens.unsqueeze(0) + offsets.unsqueeze(1)).flatten()

    return PackPlan(
        x_ids=x_ids,
        joint_src=joint_src,
        x_slot=x_slot,
        modality=x_ids[x_slot, 3],
        segment=x_seg.clamp(max=n - 1),  # PAD rows inherit the last sample's t
        cu_seqlens=cu_seqlens,
        max_seqlen=max_seqlen,
        max_seqs=max_seqs,
    )


def pack_noised(tokens_list, x_len):
    """Stack per-sample (L_i, C) tokens into one (1, x_len, C) bin; tail is PAD."""
    assert tokens_list, "empty bin"
    width = tokens_list[0].shape[-1]
    device, dtype = tokens_list[0].device, tokens_list[0].dtype
    out = torch.zeros(1, x_len, width, device=device, dtype=dtype)
    cursor = 0
    for tok in tokens_list:
        n = tok.shape[0]
        out[0, cursor : cursor + n] = tok
        cursor += n
    assert cursor <= x_len
    return out


def pack_text(text_list, max_seqs, text_len):
    """Stack per-sample (L_t, D) text features into (1, max_seqs * L_t, D)."""
    assert text_list, "empty bin"
    dim = text_list[0].shape[-1]
    device, dtype = text_list[0].device, text_list[0].dtype
    out = torch.zeros(1, max_seqs * text_len, dim, device=device, dtype=dtype)
    for i, y in enumerate(text_list):
        out[0, i * text_len : (i + 1) * text_len] = y
    return out


def timeshift(t, alpha):
    """Timeshift s(a, u) = a * u / (1 + (a - 1) * u) (Esser et al.).

    Self-Flow states it on the noise level u, where u = 1 is pure noise; this
    codebase's t = 1 is clean data, so the shift applies to u = 1 - t. alpha > 1
    moves mass toward high noise, alpha == 1 is the identity, and both endpoints
    are fixed -- so a shifted grid still spans the whole path. Used for the
    training timestep distribution (trainshift) and for the sampling grid
    (sampleshift).
    """
    if alpha == 1.0:
        return t
    u = 1 - t
    return 1 - alpha * u / (1 + (alpha - 1) * u)


def masked_mean(err, mask, channels):
    """Mean of `err[..., :channels]` over positions where `mask` is True.

    `err`: (B, L, C), `mask`: (B, L) bool/float. Empty mask -> 0 (no grad).
    Uses a sum/count form so PAD slots contribute neither value nor gradient,
    and so torch.compile stays sync-free (no boolean advanced indexing).
    """
    m = mask.to(err.dtype).unsqueeze(-1)
    num = (err[..., :channels] * m).sum()
    den = m.sum() * channels
    return num / den.clamp(min=1.0)


def masked_per_sample_losses(err, mask, channels, segment, max_seqs):
    """Masked mean of `err[..., :channels]` for each sample of a packed bin.

    Returns (per_sample (B, max_seqs), valid (B, max_seqs)) where `valid` marks
    the slots that actually received tokens. `err`: (B, L, C); `mask`: (B, L) or
    (L,); `segment`: (L,) sample index per row. `max_seqs` is the static bin
    capacity, so unused slots score zero count and fall out (one compiled graph,
    no host sync).
    """
    b, seq = err.shape[0], err.shape[1]
    m = mask.to(err.dtype).expand(b, seq).contiguous()       # (B, L)
    num = (err[..., :channels] * m.unsqueeze(-1)).sum(-1)    # (B, L)
    seg = segment.unsqueeze(0).expand(b, seq)
    sums = err.new_zeros(b, max_seqs).scatter_add_(1, seg, num)
    counts = err.new_zeros(b, max_seqs).scatter_add_(1, seg, m)
    return sums / (counts * channels).clamp(min=1.0), counts > 0


def masked_per_sample_mean(err, mask, channels, segment, max_seqs):
    """Per-sample masked mean over `channels`, then averaged across samples.

    Every sample in the bin gets weight 1/n regardless of how many tokens it
    contributed (Wan / LoongForge packing behaviour). A plain global token mean
    lets a 4k-token clip dominate a 1.8k-token one, so the loss scale — and its
    gradient — swings with whatever the packer happened to put in the bin.
    """
    per_sample, valid = masked_per_sample_losses(err, mask, channels, segment, max_seqs)
    v = valid.to(err.dtype)
    return (per_sample * v).sum() / v.sum().clamp(min=1.0)
