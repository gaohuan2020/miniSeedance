"""
Build a small multi-resolution training subset from the raw AVCaps dataset.

Picks the N shortest clips, optionally ensuring every available resolution is
represented, trims each to one or more configured durations, and resizes by
aspect ratio into resolution buckets (all divisible by 16 * patch_size = 32):
    landscape 16:9-ish -> 512x288 | landscape 4:3-ish -> 512x384 | portrait -> 288x512

Per item, outputs <out-dir>/:
    XXXXX.npz         frames (T, H, W, 3) uint8 @ 8 fps + audio (2, N) int16 @ 44.1 kHz
    gt_XXXXX.mp4/.wav ground-truth preview at the training resolution
    metadata.jsonl    {"file_name", "text" (first audio-visual caption), "source"}

CLI: python -m src.data.processing.avcaps_subset --avcaps-dir data/avcaps --out-dir data/avcaps10
"""

import argparse
import json
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
from imageio_ffmpeg import get_ffmpeg_exe

from src.log import get_logger
from src.media import save_mp4, save_wav

logger = get_logger(__name__)

SAMPLE_RATE = 44100


def bucket_resolution(w, h):
    ar = w / h
    if ar < 1.0:
        return 288, 512
    if ar > 1.55:
        return 512, 288
    return 512, 384


def select_clips(stats, count, ensure_all_resolutions=False):
    """Select the shortest clips, replacing duplicates to cover all buckets."""
    picked = list(stats[:count])
    if not ensure_all_resolutions:
        return picked

    available = {bucket_resolution(w, h) for _, _, w, h in stats}
    if count < len(available):
        raise ValueError(
            f"--count={count} cannot cover all {len(available)} resolution buckets"
        )

    counts = {}
    for _, _, w, h in picked:
        size = bucket_resolution(w, h)
        counts[size] = counts.get(size, 0) + 1

    for missing in sorted(available - counts.keys()):
        candidate = next(item for item in stats if bucket_resolution(item[2], item[3]) == missing)
        replace = max(
            (i for i, item in enumerate(picked)
             if counts[bucket_resolution(item[2], item[3])] > 1),
            key=lambda i: picked[i][0],
        )
        old_size = bucket_resolution(picked[replace][2], picked[replace][3])
        counts[old_size] -= 1
        counts[missing] = 1
        picked[replace] = candidate

    return sorted(picked)


def read_frames(path, num_frames, fps, size):
    """First num_frames at `fps`, resized to size=(W, H), RGB uint8."""
    cap = cv2.VideoCapture(str(path))
    src_fps = cap.get(cv2.CAP_PROP_FPS)
    need = {round(i / fps * src_fps): i for i in range(num_frames)}  # src idx -> out idx
    frames = [None] * num_frames
    src_idx = 0
    while any(f is None for f in frames):
        ok, frame = cap.read()
        if not ok:
            break
        if src_idx in need:
            frame = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
            frames[need[src_idx]] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        src_idx += 1
    cap.release()
    last = next(f for f in reversed(frames) if f is not None)
    return np.stack([f if f is not None else last for f in frames])


def read_audio(path, duration, start=0.0):
    """`duration` seconds from `start` as (2, N) int16 @ 44.1 kHz (zero-padded)."""
    n = int(duration * SAMPLE_RATE)
    cmd = [get_ffmpeg_exe()]
    if start:
        cmd += ["-ss", f"{start}"]        # before -i: seek without decoding
    cmd += [ "-i", str(path), "-t", f"{duration}", "-vn",
           "-ac", "2", "-ar", str(SAMPLE_RATE), "-f", "s16le", "-"]
    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg audio failed: {proc.stderr.decode()[-300:]}")
    pcm = np.frombuffer(proc.stdout, dtype=np.int16).reshape(-1, 2).T  # (2, N)
    out = np.zeros((2, n), dtype=np.int16)
    out[:, : min(n, pcm.shape[1])] = pcm[:, :n]
    return out


# AVCaps describes each clip several times over. Ordered so the first entry is
# the caption the single-caption pipeline always used, which keeps evaluation
# prompts unchanged. Audio-only captions are left out by default: as a text
# prompt "the people were speaking together" says nothing about what is on
# screen, so it adds conditioning ambiguity rather than removing it.
CAPTION_FIELDS = ("audio_visual_captions", "GPT_AV_captions", "visual_captions")
ALL_CAPTION_FIELDS = CAPTION_FIELDS + ("audio_captions",)


def collect_captions(entry, fields=CAPTION_FIELDS):
    """Every distinct caption of one clip, primary first."""
    seen = {}
    for key in fields:
        for caption in entry.get(key) or ():
            caption = caption.strip()
            if caption:
                seen.setdefault(caption, None)
    if not seen:
        raise ValueError("no caption")
    return list(seen)


def pick_caption(entry):
    """The one caption used where a single prompt is needed."""
    return collect_captions(entry, ALL_CAPTION_FIELDS)[0]


def probe_video(path):
    """Return (duration, path, width, height) without decoding the clip."""
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 0
    duration = cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps if fps else 0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return duration, path, width, height


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--avcaps-dir", type=str, default="data/avcaps")
    parser.add_argument("--out-dir", type=str, default="data/avcaps10")
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--duration", type=float, default=20.0)
    parser.add_argument(
        "--durations",
        type=str,
        help="comma-separated clip durations assigned round-robin, e.g. 10,20",
    )
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4,
                        help="parallel video probing/decoding workers")
    parser.add_argument(
        "--ensure-all-resolutions",
        action="store_true",
        help="replace duplicate short clips so every available resolution bucket is represented",
    )
    args = parser.parse_args()

    avcaps = Path(args.avcaps_dir)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    captions = json.load(open(avcaps / "train_captions.json"))
    durations = (
        [float(value) for value in args.durations.split(",")]
        if args.durations else [args.duration]
    )
    if not durations or any(duration <= 0 for duration in durations):
        raise ValueError("durations must be positive")
    if args.workers < 1:
        raise ValueError("--workers must be at least 1")
    for duration in durations:
        sampled_frames = duration * args.fps
        if not sampled_frames.is_integer() or int(sampled_frames) % 4:
            raise ValueError(
                f"duration {duration:g}s at {args.fps} fps does not produce 1 + 4k frames"
            )
    required_duration = max(durations)

    # Shortest clips that still cover every requested target duration
    candidates = [
        path for path in sorted((avcaps / "train_videos").iterdir())
        if path.stem in captions
    ]
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        probed = pool.map(probe_video, candidates)
        stats = [item for item in probed if item[0] >= required_duration]
    stats.sort()
    picked = select_clips(stats, args.count, args.ensure_all_resolutions)
    logger.info("picked %d clips, durations %.1f-%.1fs", len(picked), picked[0][0], picked[-1][0])

    def process_clip(index_item):
        i, (dur, f, w, h) = index_item
        target_duration = durations[i % len(durations)]
        num_frames = int(target_duration * args.fps) + 1  # 4k+1 for the causal VAE
        size = bucket_resolution(w, h)
        frames = read_frames(f, num_frames, args.fps, size)
        audio = read_audio(f, target_duration)
        text = pick_caption(captions[f.stem])
        np.savez_compressed(out / f"{i:05d}.npz", frames=frames, audio=audio)
        wav_path = out / f"gt_{i:05d}.wav"
        save_wav(audio.astype(np.float32) / 32767.0, wav_path, SAMPLE_RATE)
        save_mp4(frames, out / f"gt_{i:05d}.mp4", wav_path=wav_path, fps=args.fps)
        item_meta = {
            "file_name": f"{i:05d}.npz",
            "text": text,
            "source": f.name,
            "duration": target_duration,
        }
        return item_meta, (i, f, target_duration, w, h, size, text)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        processed = list(pool.map(process_clip, enumerate(picked)))

    meta = []
    for item_meta, (i, f, target_duration, w, h, size, text) in processed:
        meta.append(item_meta)
        logger.info(
            "[%d/%d] %s %.1fs %dx%d -> %dx%d | %s",
            i + 1, len(picked), f.name, target_duration, w, h, *size, text[:80],
        )

    with open(out / "metadata.jsonl", "w") as fh:
        for m in meta:
            fh.write(json.dumps(m, ensure_ascii=False) + "\n")
    logger.info("wrote %d items -> %s", len(meta), out)


if __name__ == "__main__":
    main()
