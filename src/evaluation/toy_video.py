#!/usr/bin/env python3
"""
Evaluate prompt fidelity of the toy text-to-video model.

For every (color, shape, direction, background) combination, generate a video
and check with pixel heuristics:
- background / shape color: nearest palette color (border pixels / blob median)
- shape type: occupancy of the four bbox corners on the middle frame
- motion direction: centroid displacement between first and last frame

Usage:
    python -m src.evaluation.toy_video \
        --ckpts selfflow=results/video/ckpt_last.pt fm=results/video_fm/ckpt_last.pt \
        --seeds 0 --output-dir eval_video
"""

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageFilter

from src.config import resolve_ckpt_config
from src.data.synthesis.toy_video import BACKGROUNDS, COLORS, DIRECTIONS, SHAPES
from src.inference import load_dit_from_ckpt, sample_videos
from src.log import get_logger
from src.models import build_audio_vae, build_text_encoder, build_video_vae

logger = get_logger("eval")


def nearest(rgb, palette):
    dists = {name: float(np.linalg.norm(np.array(rgb) - np.array(val))) for name, val in palette.items()}
    return min(dists, key=dists.get)


def frame_mask(frame: np.ndarray, bg_rgb: np.ndarray) -> np.ndarray:
    dist = np.linalg.norm(frame.astype(np.float32) - bg_rgb, axis=-1)
    mask = dist > 60
    mask_img = Image.fromarray((mask * 255).astype(np.uint8)).filter(ImageFilter.MedianFilter(size=5))
    return np.array(mask_img) > 127


def classify_frame(frame: np.ndarray):
    """(H, W, 3) uint8 -> (color, shape, bg); color/shape None if no object."""
    f = frame.astype(np.float32)
    border = np.concatenate([
        f[:8].reshape(-1, 3), f[-8:].reshape(-1, 3), f[:, :8].reshape(-1, 3), f[:, -8:].reshape(-1, 3),
    ])
    bg_rgb = np.median(border, axis=0)
    bg = nearest(bg_rgb, BACKGROUNDS)

    mask = frame_mask(frame, bg_rgb)
    if mask.sum() < 200:
        return None, None, bg

    color = nearest(np.median(f[mask], axis=0), COLORS)

    rows, cols = np.where(mask)
    r0, r1 = rows.min(), rows.max()
    c0, c1 = cols.min(), cols.max()
    kh = max(1, int(0.2 * (r1 - r0 + 1)))
    kw = max(1, int(0.2 * (c1 - c0 + 1)))
    tl = mask[r0:r0 + kh, c0:c0 + kw].mean()
    tr = mask[r0:r0 + kh, c1 - kw + 1:c1 + 1].mean()
    bl = mask[r1 - kh + 1:r1 + 1, c0:c0 + kw].mean()
    br = mask[r1 - kh + 1:r1 + 1, c1 - kw + 1:c1 + 1].mean()
    top, bottom = (tl + tr) / 2, (bl + br) / 2

    if top < 0.4 and bottom > 0.55:
        shape = "triangle"
    elif top > 0.6 and bottom > 0.6:
        shape = "square"
    else:
        shape = "circle"

    return color, shape, bg


def classify_video(frames: np.ndarray):
    """(T, H, W, 3) uint8 -> (color, shape, bg, direction)."""
    mid = frames[len(frames) // 2]
    color, shape, bg = classify_frame(mid)

    # Direction from centroid displacement between first and last frame
    f = frames[0].astype(np.float32)
    border = np.concatenate([
        f[:8].reshape(-1, 3), f[-8:].reshape(-1, 3), f[:, :8].reshape(-1, 3), f[:, -8:].reshape(-1, 3),
    ])
    bg_rgb = np.median(border, axis=0)

    cents = []
    for fr in (frames[0], frames[-1]):
        mask = frame_mask(fr, bg_rgb)
        if mask.sum() < 50:
            return color, shape, bg, None
        ys, xs = np.where(mask)
        cents.append((xs.mean(), ys.mean()))
    dx = cents[1][0] - cents[0][0]
    dy = cents[1][1] - cents[0][1]
    if abs(dx) > abs(dy):
        direction = "right" if dx > 0 else "left"
    else:
        direction = "down" if dy > 0 else "up"
    return color, shape, bg, direction


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpts", type=str, nargs="+", required=True, help="label=path pairs")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--num-steps", type=int, default=50)
    parser.add_argument("--cfg-scale", type=float, default=1.0)
    parser.add_argument("--chunk", type=int, default=54, help="prompts per sampling batch")
    parser.add_argument("--subset-every", type=int, default=1, help="evaluate every Nth combination")
    parser.add_argument("--output-dir", type=str, default="eval_video")
    parser.add_argument("--use-ema", action="store_true", help="evaluate EMA weights (default: raw)")
    parser.add_argument("--no-ema", action="store_false", dest="use_ema")
    args = parser.parse_args()

    device = "cuda"
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    combos = list(itertools.product(COLORS, SHAPES, DIRECTIONS, BACKGROUNDS))[:: args.subset_every]
    prompts = [f"a {c} {s} moving {d} on a {b} background" for c, s, d, b in combos]
    logger.info("%d combinations x %d seeds per model", len(combos), len(args.seeds))

    # Components are rebuilt per checkpoint config, cached by identity
    encoders, vaes, audio_vaes = {}, {}, {}
    results = {}
    for pair in args.ckpts:
        label, ckpt_path = pair.split("=", 1)
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        latent_shape = ckpt["latent_shape"]
        mk, config = resolve_ckpt_config(ckpt)
        p = config["model"]["patch_size"]
        audio_info = ckpt.get("audio")

        te_key = json.dumps(config["text_encoder"], sort_keys=True)
        if te_key not in encoders:
            encoders[te_key] = build_text_encoder(config, device)
        text_encoder = encoders[te_key]
        vae_key = json.dumps(config["video_vae"], sort_keys=True)
        if vae_key not in vaes:
            vaes[vae_key] = build_video_vae(config, device)
        vae = vaes[vae_key]
        audio_vae = None
        if audio_info is not None:
            av_key = json.dumps(config["audio_vae"], sort_keys=True)
            if av_key not in audio_vaes:
                audio_vaes[av_key] = build_audio_vae(config, device)
            audio_vae = audio_vaes[av_key]

        model = load_dit_from_ckpt(ckpt, config, mk, device, text_encoder.embed_dim, use_ema=args.use_ema)
        objective = ckpt.get("args", {}).get("objective") or config["train"]["objective"]
        logger.info("[%s] loaded %s (step %d, objective %s)",
                    label, ckpt_path, ckpt["step"], objective)

        stats = {"color": 0, "shape": 0, "bg": 0, "direction": 0, "all": 0, "no_object": 0, "n": 0}
        records = []

        for seed in args.seeds:
            for i in range(0, len(prompts), args.chunk):
                chunk_prompts = prompts[i : i + args.chunk]
                chunk_combos = combos[i : i + args.chunk]
                videos, _ = sample_videos(
                    model, vae, text_encoder, chunk_prompts, latent_shape,
                    num_steps=args.num_steps, cfg_scale=args.cfg_scale,
                    device=device, seed=seed * 10000 + i, patch_size=p,
                    patch_size_t=config["model"].get("patch_size_t", 1),
                    audio_info=audio_info, audio_vae=audio_vae,
                )
                for (c, s, d, b), frames in zip(chunk_combos, videos):
                    pc, ps, pb, pd = classify_video(frames)
                    ok = {"color": pc == c, "shape": ps == s, "bg": pb == b, "direction": pd == d}
                    for k, v in ok.items():
                        stats[k] += v
                    stats["all"] += all(ok.values())
                    stats["no_object"] += pc is None
                    stats["n"] += 1
                    records.append(dict(
                        prompt=f"a {c} {s} moving {d} on a {b} background", seed=seed,
                        pred=dict(color=pc, shape=ps, bg=pb, direction=pd),
                        ok=bool(all(ok.values())),
                    ))

        n = stats["n"]
        summary = {k: stats[k] / n for k in ["color", "shape", "bg", "direction", "all", "no_object"]}
        summary["n"] = n
        results[label] = {"summary": summary, "records": records}
        logger.info("[%s] color %.1f%% | shape %.1f%% | bg %.1f%% | direction %.1f%% "
                    "| all %.1f%% | no_object %.1f%% (n=%d)",
                    label, 100 * summary["color"], 100 * summary["shape"], 100 * summary["bg"],
                    100 * summary["direction"], 100 * summary["all"],
                    100 * summary["no_object"], n)

    with open(out_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Saved results to %s", out_dir / "results.json")


if __name__ == "__main__":
    main()
