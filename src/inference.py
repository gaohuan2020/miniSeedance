"""
Sampling entry points shared by the CLI, the eval scripts and the in-training
sampler.

- sample_latents: token-level (precomputed text features -> latents); used
  during training where no text encoder is loaded.
- sample_videos: prompt-level (text encoder + VAE decode -> uint8 frames and
  waveforms); used by src/sample.py and src/evaluation/.
"""

import math

import numpy as np
import torch

from src.log import get_logger
from src.media import to_uint8_frames
from src.sampling import denoise_loop
from src.utils import TokenLayout

logger = get_logger("inference")


def parse_aspect(spec):
    """'16:9' | '4:3' | '1.78' -> a width/height ratio."""
    if ":" in spec:
        w, h = spec.split(":", 1)
        ratio = float(w) / float(h)
    else:
        ratio = float(spec)
    assert ratio > 0, f"aspect ratio must be positive, got {spec!r}"
    return ratio


def resolve_aspect(spec, ckpt):
    """Snap a requested aspect ratio onto the nearest bucket the model trained on.

    Returns (latent_shape, audio_len). Only a handful of shapes were ever
    trained, and RoPE uses absolute coordinates, so an off-bucket grid is pure
    extrapolation; snapping keeps generation inside the trained regime.

    Duration follows the ratio: it is kept at the checkpoint default whenever
    that ratio was trained at that duration, and otherwise moves to the closest
    duration available for the ratio (portrait, say, may only exist at 10s).
    Audio length moves with it, since the two must stay aligned.
    """
    ratio = parse_aspect(spec)
    default_shape = tuple(ckpt["latent_shape"])
    shapes = [tuple(s) for s in ckpt.get("latent_shapes") or [default_shape]]
    t_default = default_shape[1]

    def distance(shape):
        _, t, h, w = shape
        return round(abs(math.log((w / h) / ratio)), 6), abs(t - t_default)

    shape = min(shapes, key=distance)
    _, t, h, w = shape
    logger.info(
        "aspect %s -> trained bucket %s (%dx%d px, %d frames, ratio %.3f)",
        spec, shape, w * 16, h * 16, t * 4 - 3, w / h,
    )
    if abs(math.log((w / h) / ratio)) > 1e-3:
        logger.warning(
            "no bucket matches %s exactly; nearest trained ratio is %.3f", spec, w / h
        )

    audio = ckpt.get("audio")
    if audio is None:
        return shape, 0
    lens = audio.get("latent_lens")
    if lens is not None:
        return shape, lens[shapes.index(shape)]
    # Checkpoints saved before latent_lens: audio length is proportional to a
    # bucket's frame count, which pins it to (T - 1).
    audio_len = round(audio["latent_len"] * (t - 1) / (t_default - 1))
    if audio_len != audio["latent_len"]:
        logger.warning(
            "checkpoint has no per-bucket audio lengths; scaled %d -> %d for T=%d",
            audio["latent_len"], audio_len, t,
        )
    return shape, audio_len


@torch.no_grad()
def sample_latents(
    model, cond, null, layout: TokenLayout, device,
    num_steps=50, cfg_scale=1.0, guidance_low=0.0, guidance_high=1.0,
    mode="SDE", shift=1.0, seed=None,
):
    """Denoise from pure noise with cached text features.

    cond: (B, L_t, D) conditioning features; null: (1, L_t, D).
    Returns (video_latents (B, C, T, H, W), audio_latents (B, C_a, L_a) | None).
    """
    if seed is not None:
        torch.manual_seed(seed)
    if getattr(model, "seq_l_v", None) is None and hasattr(model, "seq_l_v"):
        model.seq_l_v = layout.l_v
    b = cond.shape[0]
    c, t, h, w = layout.latent_shape

    noise = torch.randn(b, c, t, h, w, device=device, dtype=torch.bfloat16)
    a_noise = None
    if layout.has_audio:
        a_noise = torch.randn(
            b, layout.audio_channels, layout.audio_len, device=device, dtype=torch.bfloat16
        )
    x = layout.pack(noise, a_noise)
    x_ids = layout.x_ids(device).unsqueeze(0).expand(b, -1, -1)

    if cfg_scale > 1.0:
        x = torch.cat([x, x], dim=0)
        x_ids = torch.cat([x_ids, x_ids], dim=0)
        vector = torch.cat([null.expand(b, -1, -1), cond], dim=0)  # [uncond, cond]
    else:
        vector = cond

    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        samples = denoise_loop(
            model=model, num_steps=num_steps,
            cfg_scale=cfg_scale if cfg_scale > 1.0 else None,
            guidance_low=guidance_low, guidance_high=guidance_high,
            mode=mode, shift=shift, x=x, x_ids=x_ids, vector=vector,
        )
    if cfg_scale > 1.0:
        samples = samples[b:]

    video_latents, audio_latents = layout.unpack(samples.float())
    return video_latents, audio_latents


@torch.no_grad()
def sample_videos(
    model, vae, text_encoder, prompts, latent_shape,
    num_steps=50, cfg_scale=1.0, guidance_low=0.0, guidance_high=1.0,
    mode="SDE", shift=1.0, device="cuda", seed=None, patch_size=2, patch_size_t=1,
    text_len=16, audio_info=None, audio_vae=None,
):
    """Prompts -> (videos uint8 (B, T, H, W, 3), waveforms (B, 2, N) | None)."""
    layout = TokenLayout(
        latent_shape, patch_size, patch_size_t=patch_size_t,
        audio_channels=audio_info["in_channels"] if audio_info else 0,
        audio_len=audio_info["latent_len"] if audio_info else 0,
    )
    cond = text_encoder.encode_tokens(prompts, max_length=text_len)
    null = text_encoder.encode_tokens([""], max_length=text_len)

    video_latents, audio_latents = sample_latents(
        model, cond, null, layout, device,
        num_steps=num_steps, cfg_scale=cfg_scale,
        guidance_low=guidance_low, guidance_high=guidance_high, mode=mode, shift=shift,
        seed=seed,
    )

    waveforms = None
    if audio_latents is not None:
        audio_latents = audio_latents * audio_info["std"].to(device) + audio_info["mean"].to(device)
        waveforms = audio_vae.decode(audio_latents).float().clamp(-1, 1).cpu().numpy()

    videos = to_uint8_frames(vae.decode(video_latents))
    return videos, waveforms


def load_dit_from_ckpt(ckpt, config, mk, device, text_dim, use_ema=False):
    """Instantiate the DiT described by a checkpoint and load its weights."""
    from src.models.dit import TextToVideoDiT

    audio_info = ckpt.get("audio")
    p = config["model"]["patch_size"]
    pt = config["model"].get("patch_size_t", 1)
    model = TextToVideoDiT(
        video_in_channels=ckpt["latent_shape"][0] * pt * p * p,
        text_in_channels=text_dim,
        audio_in_channels=audio_info["in_channels"] if audio_info else None,
        **mk,
    ).to(device)
    model.load_state_dict(ckpt["ema" if use_ema else "model"])
    model.eval()
    return model
