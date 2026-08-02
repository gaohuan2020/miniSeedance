"""Toy text-to-video dataset: colored shapes moving in a direction, with a
direction-coded audio track for joint audio-video training.

Each video is T frames of a single shape translating at constant velocity.
Caption: "a {color} {shape} moving {direction} on a {bg} background".

Audio (stereo 44.1 kHz, aligned to the clip duration) encodes the motion:
    right -> constant 660 Hz        left -> constant 330 Hz
    up    -> sweep 330 -> 660 Hz    down -> sweep 660 -> 330 Hz

Videos are saved as .npz: `frames` uint8 (T, H, W, 3), `audio` int16 (2, N),
plus a metadata.jsonl with {"file_name", "text"} per line.
"""

import json
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from src.log import get_logger

logger = get_logger(__name__)

COLORS = {
    "red": (220, 40, 40),
    "green": (40, 180, 60),
    "blue": (40, 80, 220),
    "yellow": (235, 200, 30),
    "purple": (150, 60, 200),
    "orange": (240, 140, 30),
}

BACKGROUNDS = {
    "white": (245, 245, 245),
    "black": (20, 20, 20),
    "gray": (128, 128, 128),
}

SHAPES = ["circle", "square", "triangle"]

DIRECTIONS = {
    "right": (1, 0),
    "left": (-1, 0),
    "down": (0, 1),
    "up": (0, -1),
}

AUDIO_RATE = 44100
# Audio length is a multiple of the Stable Audio VAE downsampling ratio (2048)
# closest to the clip duration (17 frames @ 8 fps = 2.125 s): 46 * 2048 samples.
AUDIO_SAMPLES = 46 * 2048

# Direction -> (start_freq, end_freq) in Hz; constant tone when equal
DIRECTION_TONES = {
    "right": (660.0, 660.0),
    "left": (330.0, 330.0),
    "up": (330.0, 660.0),
    "down": (660.0, 330.0),
}


def make_audio(direction):
    """Stereo int16 tone whose pitch trajectory encodes the motion direction."""
    f0, f1 = DIRECTION_TONES[direction]
    t = np.arange(AUDIO_SAMPLES) / AUDIO_RATE
    dur = AUDIO_SAMPLES / AUDIO_RATE
    # Linear frequency sweep: phase = 2*pi * (f0*t + (f1-f0)*t^2 / (2*dur))
    phase = 2 * np.pi * (f0 * t + (f1 - f0) * t * t / (2 * dur))
    wave = 0.5 * np.sin(phase)
    fade = min(int(0.05 * AUDIO_RATE), AUDIO_SAMPLES // 4)
    env = np.ones(AUDIO_SAMPLES)
    env[:fade] = np.linspace(0, 1, fade)
    env[-fade:] = np.linspace(1, 0, fade)
    wave = (wave * env * 32767).astype(np.int16)
    return np.stack([wave, wave])  # (2, N) stereo


def draw_frame(width, height, bg_rgb, color_rgb, shape, cx, cy, half):
    img = Image.new("RGB", (width, height), bg_rgb)
    draw = ImageDraw.Draw(img)
    if shape == "circle":
        draw.ellipse([cx - half, cy - half, cx + half, cy + half], fill=color_rgb)
    elif shape == "square":
        draw.rectangle([cx - half, cy - half, cx + half, cy + half], fill=color_rgb)
    else:  # triangle
        draw.polygon(
            [(cx, cy - half), (cx - half, cy + half), (cx + half, cy + half)],
            fill=color_rgb,
        )
    return np.asarray(img, dtype=np.uint8)


def make_video(width, height, num_frames, rng):
    color = rng.choice(list(COLORS))
    shape = rng.choice(SHAPES)
    bg = rng.choice(list(BACKGROUNDS))
    direction = rng.choice(list(DIRECTIONS))
    dx, dy = DIRECTIONS[direction]

    short = min(width, height)
    half = rng.randint(short // 8, short // 5)
    speed = rng.uniform(2.0, 4.0)  # pixels per frame
    travel = speed * (num_frames - 1)

    # Start position such that the shape stays fully inside the frame
    margin = half + 2

    def start(lo, hi, d):
        if d > 0:
            return rng.uniform(lo, hi - travel)
        if d < 0:
            return rng.uniform(lo + travel, hi)
        return rng.uniform(lo, hi)

    x0 = start(margin, width - margin, dx)
    y0 = start(margin, height - margin, dy)

    frames = []
    for i in range(num_frames):
        cx = int(round(x0 + dx * speed * i))
        cy = int(round(y0 + dy * speed * i))
        frames.append(draw_frame(width, height, BACKGROUNDS[bg], COLORS[color], shape, cx, cy, half))

    caption = f"a {color} {shape} moving {direction} on a {bg} background"
    return np.stack(frames), make_audio(direction), caption


def generate_dataset(output_dir, num_videos, num_frames=17, width=128, height=128, seed=0):
    assert (num_frames - 1) % 4 == 0, "num_frames must be 1 + 4k for the Wan VAE"
    assert width % 16 == 0 and height % 16 == 0, "width/height must be divisible by 16"
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)

    with open(out / "metadata.jsonl", "w") as f:
        for i in range(num_videos):
            frames, audio, caption = make_video(width, height, num_frames, rng)
            name = f"{i:05d}.npz"
            np.savez_compressed(out / name, frames=frames, audio=audio, audio_rate=AUDIO_RATE)
            f.write(json.dumps({"file_name": name, "text": caption}) + "\n")

    logger.info("Wrote %d videos (%dx%dx%d) to %s", num_videos, num_frames, height, width, out)


def main():
    """CLI: python -m src.data.synthesis.toy_video --output-dir data/toy_video --num-videos 3000"""
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=str, default="data/toy_video")
    parser.add_argument("--num-videos", type=int, default=3000)
    parser.add_argument("--num-frames", type=int, default=17, help="must be 1 + 4k for the Wan VAE")
    parser.add_argument("--size", type=int, default=128)
    parser.add_argument("--width", type=int, default=None, help="frame width (default: --size)")
    parser.add_argument("--height", type=int, default=None, help="frame height (default: --size)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    generate_dataset(
        args.output_dir, args.num_videos, num_frames=args.num_frames,
        width=args.width or args.size, height=args.height or args.size, seed=args.seed,
    )


if __name__ == "__main__":
    main()
