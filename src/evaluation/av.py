#!/usr/bin/env python3
"""
Evaluate audio-video alignment of a joint AV checkpoint on the toy dataset.

For every (color, shape, direction, background) combination, generate a clip
and classify:
- video direction: object centroid displacement (from evaluation.toy_video)
- audio direction: dominant frequency in the first / last 0.4 s windows
  (660 Hz constant = right, 330 Hz constant = left, rising = up, falling = down)

Reports video accuracy, audio accuracy, and AV consistency.

Usage:
    python -m src.evaluation.av --ckpt results/av_sf/ckpt_last.pt --seeds 0 --cfg-scale 3.0
"""

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import torch

from src.evaluation.toy_video import classify_video
from src.config import resolve_ckpt_config
from src.data.synthesis.toy_video import AUDIO_RATE, BACKGROUNDS, COLORS, DIRECTIONS, SHAPES
from src.inference import load_dit_from_ckpt, sample_videos
from src.log import get_logger
from src.models import build_audio_vae, build_text_encoder, build_video_vae

logger = get_logger("eval_av")


def dominant_freq(mono, sample_rate=AUDIO_RATE):
    spec = np.abs(np.fft.rfft(mono * np.hanning(len(mono))))
    freqs = np.fft.rfftfreq(len(mono), 1 / sample_rate)
    band = (freqs > 100) & (freqs < 2000)
    return float(freqs[band][spec[band].argmax()])


def classify_audio(waveform, sample_rate=AUDIO_RATE):
    """waveform (2, N) float -> direction guess from the pitch trajectory."""
    mono = waveform.mean(axis=0)
    w = int(0.4 * sample_rate)
    f_start = dominant_freq(mono[:w], sample_rate)
    f_end = dominant_freq(mono[-w:], sample_rate)

    if f_end > f_start * 1.25:
        return "up", f_start, f_end
    if f_end < f_start / 1.25:
        return "down", f_start, f_end
    mid = (f_start + f_end) / 2
    return ("right" if abs(mid - 660) < abs(mid - 330) else "left"), f_start, f_end


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--num-steps", type=int, default=50)
    parser.add_argument("--cfg-scale", type=float, default=1.0)
    parser.add_argument("--chunk", type=int, default=54)
    parser.add_argument("--subset-every", type=int, default=1)
    parser.add_argument("--output-dir", type=str, default=None, help="optionally dump wav/gif samples")
    parser.add_argument("--use-ema", action="store_true", help="evaluate EMA weights (default: raw)")
    parser.add_argument("--no-ema", action="store_false", dest="use_ema")
    args = parser.parse_args()

    device = "cuda"
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=True)
    assert ckpt.get("audio") is not None, "checkpoint was not trained with audio"
    audio_info = ckpt["audio"]
    latent_shape = ckpt["latent_shape"]
    mk, config = resolve_ckpt_config(ckpt)
    p = config["model"]["patch_size"]

    text_encoder = build_text_encoder(config, device)
    vae = build_video_vae(config, device)
    audio_vae = build_audio_vae(config, device)
    model = load_dit_from_ckpt(ckpt, config, mk, device, text_encoder.embed_dim, use_ema=args.use_ema)
    objective = ckpt.get("args", {}).get("objective") or config["train"]["objective"]
    logger.info("Loaded %s (step %d, objective %s)", args.ckpt, ckpt["step"], objective)

    combos = list(itertools.product(COLORS, SHAPES, DIRECTIONS, BACKGROUNDS))[:: args.subset_every]
    prompts = [f"a {c} {s} moving {d} on a {b} background" for c, s, d, b in combos]
    logger.info("%d combinations x %d seeds", len(combos), len(args.seeds))

    stats = {"video_all": 0, "video_dir": 0, "audio_dir": 0, "av_consistent": 0, "n": 0}
    records = []
    for seed in args.seeds:
        for i in range(0, len(prompts), args.chunk):
            chunk_prompts = prompts[i : i + args.chunk]
            chunk_combos = combos[i : i + args.chunk]
            videos, waves = sample_videos(
                model, vae, text_encoder, chunk_prompts, latent_shape,
                num_steps=args.num_steps, cfg_scale=args.cfg_scale,
                device=device, seed=seed * 10000 + i, patch_size=p,
                patch_size_t=config["model"].get("patch_size_t", 1),
                audio_info=audio_info, audio_vae=audio_vae,
            )
            for j, ((c, s, d, b), frames, wav) in enumerate(zip(chunk_combos, videos, waves)):
                pc, ps, pb, pd = classify_video(frames)
                ad, f0, f1 = classify_audio(wav, audio_info["sample_rate"])
                stats["video_all"] += (pc, ps, pb, pd) == (c, s, b, d)
                stats["video_dir"] += pd == d
                stats["audio_dir"] += ad == d
                stats["av_consistent"] += (pd is not None) and (ad == pd)
                stats["n"] += 1
                records.append(dict(
                    prompt=chunk_prompts[j], seed=seed,
                    video=dict(color=pc, shape=ps, bg=pb, direction=pd),
                    audio=dict(direction=ad, f_start=round(f0, 1), f_end=round(f1, 1)),
                ))

    n = stats["n"]
    logger.info("video all-correct: %.1f%% | video direction: %.1f%% | "
                "audio direction: %.1f%% | AV consistent: %.1f%% (n=%d)",
                100 * stats["video_all"] / n, 100 * stats["video_dir"] / n,
                100 * stats["audio_dir"] / n, 100 * stats["av_consistent"] / n, n)

    if args.output_dir:
        out = Path(args.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        with open(out / "av_results.json", "w") as f:
            json.dump({"stats": stats, "records": records}, f, indent=2)
        logger.info("Saved records to %s", out / "av_results.json")


if __name__ == "__main__":
    main()
