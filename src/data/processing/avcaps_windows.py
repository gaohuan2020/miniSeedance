"""Cut every AVCaps source video to one fixed length, from its start.

Built for training a duration curriculum: cut the same 1661 sources at 5 s, then
10 s, then 20 s, and train each stage from the previous stage's weights. Every
source is at least 20 s, so all three stages hold the same clips and differ only
in how much of each one the model has to account for -- 5 s is 1152 video tokens
against 4032 at 20 s, so the short stage fits three times as many clips into the
same packing budget and sees three times as many captions per step.

Unlike avcaps_semantic_split.py this puts one duration in every bucket (three
spatial buckets instead of five mixed ones) and can also take several windows
per source (--max-windows) when the goal is more clips rather than a curriculum.

Output layout matches avcaps_semantic_split.py (npz per clip + metadata.jsonl),
so latent_caching.py and the trainer take it unchanged. Train clips come from
AVCaps `train_videos` and test clips from `test_videos`, so the two splits never
share a source.

Usage:
    for s in 5 10 20; do
        python -m src.data.processing.avcaps_windows --clip-seconds $s \
            --max-windows 1 --out-dir data/avcaps_${s}s --workers 24
    done
"""

import argparse
import itertools
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np

from src.data.processing.avcaps_subset import (
    SAMPLE_RATE,
    collect_captions,
    probe_video,
    read_audio,
)
from src.data.processing.avcaps_semantic_split import bucket_key, nearest_resolution
from src.log import get_logger
from src.media import save_mp4, save_wav

logger = get_logger(__name__)


def window_starts(source_duration, clip_seconds, stride, max_windows, margin=0.2):
    """Window start times inside one source, earliest first."""
    starts = []
    start = 0.0
    while start + clip_seconds <= source_duration - margin and len(starts) < max_windows:
        starts.append(round(start, 3))
        start += stride
    return starts


def read_windows(path, starts, num_frames, fps, size):
    """Frames for every window of one source in a single decode pass.

    Seeking per window would re-decode the file from the last keyframe each
    time, so instead this collects which source frame each window needs and
    walks the file once. Returns a list of (num_frames, H, W, 3) uint8 arrays.
    """
    cap = cv2.VideoCapture(str(path))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or fps
    need = {}
    for w, start in enumerate(starts):
        for i in range(num_frames):
            need.setdefault(round((start + i / fps) * src_fps), []).append((w, i))
    grabbed = [[None] * num_frames for _ in starts]
    last_needed = max(need)
    src_idx = 0
    while src_idx <= last_needed:
        ok, frame = cap.read()
        if not ok:
            break
        for w, i in need.get(src_idx, ()):
            if grabbed[w][i] is None:
                resized = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
                grabbed[w][i] = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        src_idx += 1
    cap.release()

    windows = []
    for frames in grabbed:
        last = next((f for f in reversed(frames) if f is not None), None)
        if last is None:
            return []
        windows.append(np.stack([f if f is not None else last for f in frames]))
    return windows


def write_split(sources, out_dir, split, clip_seconds, fps, stride, max_windows,
                workers, dump_media):
    """Cut and write every window of every source. Returns the metadata list.

    Each source is handed a contiguous block of clip indices up front, so the
    workers can compress and write their own npz files: zlib on a 20 MB frame
    array is the bulk of the wall time here, and doing it in one thread would
    take longer than the decoding.
    """
    out_dir.mkdir(parents=True, exist_ok=False)
    num_frames = int(clip_seconds * fps) + 1
    plan, index = [], 0
    for source in sources:
        starts = window_starts(source["source_duration"], clip_seconds, stride,
                               max_windows)
        if starts:
            plan.append((index, starts, source))
            index += len(starts)
    logger.info("%s: %d clips planned from %d sources", split, index, len(plan))
    done = itertools.count(1)

    def process(entry):
        first, starts, source = entry
        frames = read_windows(source["path"], starts, num_frames, fps, source["size"])
        out = []
        for w, (start, clip) in enumerate(zip(starts, frames)):
            audio = read_audio(source["path"], clip_seconds, start=start)
            file_name = f"{first + w:05d}.npz"
            np.savez_compressed(out_dir / file_name, frames=clip, audio=audio)
            if dump_media:
                wav_path = out_dir / f"gt_{first + w:05d}.wav"
                save_wav(audio.astype(np.float32) / 32767.0, wav_path, SAMPLE_RATE)
                save_mp4(clip, out_dir / f"gt_{first + w:05d}.mp4",
                         wav_path=wav_path, fps=fps)
            out.append({
                "file_name": file_name,
                "text": source["text"],
                "texts": source["texts"],
                "source": source["source"],
                "duration": float(clip_seconds),
                "bucket": bucket_key(clip_seconds, source["size"]),
                "split": split,
                "window": w,
                "start": start,
            })
        n = next(done)
        if n % 200 == 0:
            logger.info("%s: %d/%d sources cut", split, n, len(plan))
        return out

    with ThreadPoolExecutor(max_workers=workers) as pool:
        metadata = [item for batch in pool.map(process, plan) for item in batch]

    with (out_dir / "metadata.jsonl").open("w") as handle:
        for item in metadata:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    return metadata


def gather_sources(avcaps_dir, split, clip_seconds, workers):
    """Readable, captioned sources long enough for at least one window."""
    captions = json.loads((avcaps_dir / f"{split}_captions.json").read_text())
    paths, texts = [], {}
    for path in sorted((avcaps_dir / f"{split}_videos").iterdir()):
        if path.stem not in captions:
            continue
        try:
            texts[path.stem] = collect_captions(captions[path.stem])
        except ValueError:
            continue
        paths.append(path)

    logger.info("probing %d captioned %s videos", len(paths), split)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        stats = list(pool.map(probe_video, paths))
    sources = []
    for duration, path, width, height in stats:
        if duration < clip_seconds or width <= 0 or height <= 0:
            continue
        sources.append({
            "path": path,
            "source": path.name,
            "source_duration": duration,
            "size": nearest_resolution(width, height),
            "text": texts[path.stem][0],
            "texts": texts[path.stem],
        })
    return sources


def summarize(metadata, label):
    from collections import Counter

    buckets = Counter(item["bucket"] for item in metadata)
    sources = len({item["source"] for item in metadata})
    hours = sum(item["duration"] for item in metadata) / 3600
    logger.info("%s: %d clips from %d sources (%.2f per source), %.1f h of video",
                label, len(metadata), sources, len(metadata) / max(sources, 1), hours)
    logger.info("%s buckets: %s", label, dict(buckets.most_common()))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--avcaps-dir", default="data/avcaps")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--clip-seconds", type=float, required=True)
    parser.add_argument("--stride", type=float,
                        help="seconds between window starts (default: no overlap)")
    parser.add_argument("--max-windows", type=int, default=6,
                        help="cap per source, so long videos cannot dominate")
    parser.add_argument("--test-windows", type=int, default=1,
                        help="windows per held-out source")
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--dump-media", action="store_true",
                        help="also write gt_*.mp4/.wav previews for train clips")
    args = parser.parse_args()

    avcaps_dir = Path(args.avcaps_dir)
    out_dir = Path(args.out_dir)
    stride = args.stride or args.clip_seconds

    for split, max_windows, dump in (("train", args.max_windows, args.dump_media),
                                     ("test", args.test_windows, True)):
        sources = gather_sources(avcaps_dir, split, args.clip_seconds, args.workers)
        logger.info("%s: %d eligible sources, %.0f s median footage", split,
                    len(sources),
                    np.median([s["source_duration"] for s in sources]) if sources else 0)
        metadata = write_split(
            sources, out_dir / split, split, args.clip_seconds, args.fps, stride,
            max_windows, args.workers, dump,
        )
        summarize(metadata, f"{out_dir / split}")


if __name__ == "__main__":
    main()
