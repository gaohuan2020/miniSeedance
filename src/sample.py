#!/usr/bin/env python3
"""
Generate videos from text prompts with a trained checkpoint.

The checkpoint carries the full config, so the matching text encoder, video
VAE and (for joint AV checkpoints) audio VAE are rebuilt automatically.
Outputs a .mp4 (with audio when available), a .gif and a frame-strip .png
per prompt.

Usage:
    python -m src.sample --ckpt results/video/ckpt_last.pt \
        --prompts "a red circle moving right on a white background" --cfg-scale 3.0

Pass --aspect 16:9 (or 4:3, 9:16) to generate at a different shape; it snaps to
the nearest shape the checkpoint trained on. --width/--height override the grid
directly instead, including shapes never trained.
"""

import argparse
from pathlib import Path

import torch

from src.config import resolve_ckpt_config
from src.inference import load_dit_from_ckpt, resolve_aspect, sample_videos
from src.log import get_logger
from src.media import save_gif, save_mp4, save_strip, save_wav
from src.models import build_audio_vae, build_text_encoder, build_video_vae

logger = get_logger("sample")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--prompts", type=str, nargs="+", required=True)
    parser.add_argument("--aspect", type=str, default=None,
                        help="aspect ratio, e.g. 16:9 or 4:3; snaps to the nearest shape the "
                             "model trained on (duration and audio length follow)")
    parser.add_argument("--width", type=int, default=None,
                        help="output width in pixels (multiple of 32; default: training resolution)")
    parser.add_argument("--height", type=int, default=None,
                        help="output height in pixels (multiple of 32; default: training resolution)")
    parser.add_argument("--output-dir", type=str, default="samples_video")
    parser.add_argument("--num-steps", type=int, default=None,
                        help="default: checkpoint's qualitative sample setting")
    parser.add_argument("--cfg-scale", type=float, default=None,
                        help="default: checkpoint's qualitative sample setting")
    parser.add_argument("--mode", type=str, default=None, choices=["SDE", "ODE"],
                        help="default: checkpoint's training sample mode")
    parser.add_argument("--shift", type=float, default=None,
                        help="sampleshift alpha on the timestep grid (ODE only); > 1 spends "
                             "more steps at high noise; default: checkpoint setting")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--use-ema", action="store_true",
                        help="sample with EMA weights (default: checkpoint sample setting)")
    parser.add_argument("--no-ema", action="store_false", dest="use_ema")
    parser.set_defaults(use_ema=None)
    args = parser.parse_args()

    device = "cuda"
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=True)
    mk, config = resolve_ckpt_config(ckpt)
    sample_cfg = config["train"]["sample"]
    num_steps = sample_cfg["num_steps"] if args.num_steps is None else args.num_steps
    cfg_scale = sample_cfg["cfg_scale"] if args.cfg_scale is None else args.cfg_scale
    mode = config["train"]["sample_mode"] if args.mode is None else args.mode
    shift = config["train"]["sample_shift"] if args.shift is None else args.shift
    use_ema = sample_cfg["use_ema"] if args.use_ema is None else args.use_ema
    p = config["model"]["patch_size"]
    c_lat, t_lat, h_lat, w_lat = ckpt["latent_shape"]
    audio_info = ckpt.get("audio")

    if args.aspect is not None:
        assert args.width is None and args.height is None, \
            "--aspect picks a trained shape; pass --width/--height instead to override it freely"
        (c_lat, t_lat, h_lat, w_lat), audio_len = resolve_aspect(args.aspect, ckpt)
        if audio_info is not None:
            audio_info = {**audio_info, "latent_len": audio_len}

    # Optional output resolution override (RoPE + conv VAE handle any grid;
    # quality outside the training resolution relies on RoPE extrapolation)
    if args.width is not None:
        assert args.width % 16 == 0 and (args.width // 16) % p == 0, \
            f"width must be divisible by {16 * p}"
        w_lat = args.width // 16
    if args.height is not None:
        assert args.height % 16 == 0 and (args.height // 16) % p == 0, \
            f"height must be divisible by {16 * p}"
        h_lat = args.height // 16
    latent_shape = (c_lat, t_lat, h_lat, w_lat)
    logger.info("latent grid: %dx%dx%d -> %d frames of %dx%d (WxH)",
                t_lat, h_lat, w_lat, t_lat * 4 - 3, w_lat * 16, h_lat * 16)

    text_encoder = build_text_encoder(config, device)
    vae = build_video_vae(config, device)
    audio_vae = build_audio_vae(config, device) if audio_info is not None else None

    model = load_dit_from_ckpt(ckpt, config, mk, device, text_encoder.embed_dim, use_ema=use_ema)
    logger.info("Loaded %s weights from step %d", "EMA" if use_ema else "raw", ckpt["step"])

    videos, waveforms = sample_videos(
        model, vae, text_encoder, args.prompts, latent_shape,
        num_steps=num_steps, cfg_scale=cfg_scale,
        mode=mode, shift=shift, device=device, seed=args.seed, patch_size=p,
        patch_size_t=config["model"].get("patch_size_t", 1),
        text_len=config["text_encoder"]["text_len"],
        audio_info=audio_info, audio_vae=audio_vae,
    )

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    for i, (video, prompt) in enumerate(zip(videos, args.prompts)):
        slug = prompt.replace(" ", "_")[:60]
        save_gif(video, out / f"{i:02d}_{slug}.gif")
        save_strip(video, out / f"{i:02d}_{slug}_strip.png")
        wav_path = None
        if waveforms is not None:
            wav_path = out / f"{i:02d}_{slug}.wav"
            save_wav(waveforms[i], wav_path, audio_info["sample_rate"])
        save_mp4(video, out / f"{i:02d}_{slug}.mp4", wav_path=wav_path)
        logger.info("Saved %s (+ .gif%s", out / f"{i:02d}_{slug}.mp4",
                    " + .wav)" if wav_path is not None else ")")


if __name__ == "__main__":
    main()
