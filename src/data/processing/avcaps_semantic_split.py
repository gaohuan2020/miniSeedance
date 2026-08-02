"""Build a five-bucket AVCaps train/test split.

The training split follows the source aspect-ratio distribution naturally.
The test split comes either from AVCaps' own test division (`--test-source
official`, disjoint source videos) or, with `--test-source semantic`, from
train-pool videos held out so that every test caption is CLIP-similar to a
training caption without being an exact duplicate.

CLI:
    python -m src.data.processing.avcaps_semantic_split \
        --avcaps-dir data/avcaps --out-dir data/avcaps200_semantic
    python -m src.data.processing.avcaps_semantic_split \
        --avcaps-dir data/avcaps --out-dir data/avcaps_full \
        --train-count 0 --test-source official
"""

import argparse
import json
import math
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from src.config import load_config
from src.data.processing.avcaps_subset import (
    SAMPLE_RATE,
    collect_captions,
    probe_video,
    read_audio,
    read_frames,
)
from src.log import get_logger
from src.media import save_mp4, save_wav
from src.models import build_text_encoder


logger = get_logger(__name__)

STANDARD = (512, 384)
WIDE = (512, 288)
PORTRAIT = (288, 512)
SPATIAL_BUCKETS = (STANDARD, WIDE, PORTRAIT)
TARGET_BUCKETS = (
    (10.0, STANDARD),
    (20.0, STANDARD),
    (10.0, WIDE),
    (20.0, WIDE),
    (10.0, PORTRAIT),
)


def nearest_resolution(width, height):
    """Map a source aspect ratio to the closest supported output ratio."""
    source_ar = width / height
    return min(
        SPATIAL_BUCKETS,
        key=lambda size: abs(math.log(source_ar / (size[0] / size[1]))),
    )


def bucket_key(duration, size):
    return f"{int(duration)}s_{size[0]}x{size[1]}"


def normalize_caption(text):
    return " ".join(text.casefold().split())


def encode_texts(encoder, texts, max_length, batch_size):
    features = []
    for start in range(0, len(texts), batch_size):
        pooled = encoder.encode_pooled(
            texts[start : start + batch_size],
            max_length=max_length,
        )
        features.append(F.normalize(pooled, dim=-1).cpu())
    return torch.cat(features)


def eligible_sources(avcaps_dir, split, workers, min_duration=20.0):
    """(probe stats sorted by duration, caption by file stem) for one AVCaps split.

    Keeps only readable sources that carry a caption and are long enough to trim
    to the longest target duration.
    """
    raw_captions = json.loads((avcaps_dir / f"{split}_captions.json").read_text())
    caption_by_stem = {}
    paths = []
    for path in sorted((avcaps_dir / f"{split}_videos").iterdir()):
        if path.stem not in raw_captions:
            continue
        try:
            caption_by_stem[path.stem] = collect_captions(raw_captions[path.stem])
        except ValueError:
            continue
        paths.append(path)

    logger.info("probing %d captioned %s videos with %d workers", len(paths), split, workers)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        stats = [
            item for item in pool.map(probe_video, paths)
            if item[0] >= min_duration and item[2] > 0 and item[3] > 0
        ]
    stats.sort()
    return stats, caption_by_stem


def assign_bucket_items(stats, captions, count=None):
    """Take the shortest sources and alternate durations within each AR bucket.

    `count=None` keeps every source in `stats`.
    """
    if count is None:
        count = len(stats)
    elif len(stats) < count:
        raise ValueError(f"need {count} eligible videos, found {len(stats)}")

    spatial_counts = Counter()
    items = []
    for source_duration, path, width, height in stats[:count]:
        size = nearest_resolution(width, height)
        if size == PORTRAIT:
            duration = 10.0
        else:
            duration = 10.0 if spatial_counts[size] % 2 == 0 else 20.0
        spatial_counts[size] += 1
        items.append({
            "path": path,
            "source": path.name,
            "source_duration": source_duration,
            "source_width": width,
            "source_height": height,
            "duration": duration,
            "size": size,
            "bucket": bucket_key(duration, size),
            "text": captions[path.stem][0],
            "texts": captions[path.stem],
        })
    return items


def select_semantic_test_items(
    candidates,
    train_items,
    encoder,
    text_length,
    embedding_batch_size,
    per_bucket,
    min_similarity,
    max_similarity,
    diversity_weight,
):
    """Greedily select unseen, semantically related test captions per bucket."""
    train_texts = [item["text"] for item in train_items]
    train_normalized = {normalize_caption(text) for text in train_texts}
    candidates = [
        item for item in candidates
        if normalize_caption(item["text"]) not in train_normalized
    ]
    if not candidates:
        raise ValueError("no unseen-caption test candidates")

    train_features = encode_texts(
        encoder, train_texts, text_length, embedding_batch_size,
    )
    candidate_features = encode_texts(
        encoder, [item["text"] for item in candidates],
        text_length, embedding_batch_size,
    )
    similarities = candidate_features @ train_features.T
    best_similarity, best_anchor = similarities.max(dim=1)

    selected = []
    selected_indices = set()
    # Select scarce spatial pools first. The two landscape durations share a
    # pool but always receive distinct source videos.
    selection_order = (
        (10.0, PORTRAIT),
        (10.0, WIDE),
        (20.0, WIDE),
        (10.0, STANDARD),
        (20.0, STANDARD),
    )
    for duration, size in selection_order:
        for _ in range(per_bucket):
            best = None
            for index, item in enumerate(candidates):
                if index in selected_indices or item["size"] != size:
                    continue
                similarity = float(best_similarity[index])
                if not min_similarity <= similarity <= max_similarity:
                    continue
                diversity = 0.0
                if selected_indices:
                    chosen = torch.tensor(sorted(selected_indices), dtype=torch.long)
                    diversity = float(
                        (candidate_features[index] @ candidate_features[chosen].T).max()
                    )
                score = similarity - diversity_weight * max(0.0, diversity)
                candidate = (score, similarity, item["source"], index)
                if best is None or candidate > best:
                    best = candidate
            if best is None:
                key = bucket_key(duration, size)
                raise ValueError(
                    f"not enough semantic candidates for {key}; "
                    "relax --min-similarity/--max-similarity"
                )

            _, similarity, _, index = best
            selected_indices.add(index)
            item = dict(candidates[index])
            anchor_index = int(best_anchor[index])
            item.update({
                "duration": duration,
                "bucket": bucket_key(duration, size),
                "anchor_train_source": train_items[anchor_index]["source"],
                "anchor_train_text": train_items[anchor_index]["text"],
                "clip_similarity": similarity,
            })
            selected.append(item)

    order = {bucket_key(duration, size): i for i, (duration, size) in enumerate(TARGET_BUCKETS)}
    return sorted(selected, key=lambda item: (order[item["bucket"]], item["source"]))


def write_split(items, out_dir, fps, workers, split):
    out_dir.mkdir(parents=True, exist_ok=False)

    def process(index_item):
        index, item = index_item
        duration = item["duration"]
        frames = read_frames(
            item["path"],
            int(duration * fps) + 1,
            fps,
            item["size"],
        )
        audio = read_audio(item["path"], duration)
        file_name = f"{index:05d}.npz"
        np.savez_compressed(out_dir / file_name, frames=frames, audio=audio)
        wav_path = out_dir / f"gt_{index:05d}.wav"
        save_wav(audio.astype(np.float32) / 32767.0, wav_path, SAMPLE_RATE)
        save_mp4(
            frames,
            out_dir / f"gt_{index:05d}.mp4",
            wav_path=wav_path,
            fps=fps,
        )

        metadata = {
            "file_name": file_name,
            "text": item["text"],
            "texts": item.get("texts", [item["text"]]),
            "source": item["source"],
            "duration": duration,
            "bucket": item["bucket"],
            "split": split,
        }
        for key in ("anchor_train_source", "anchor_train_text", "clip_similarity"):
            if key in item:
                metadata[key] = item[key]
        return metadata

    with ThreadPoolExecutor(max_workers=workers) as pool:
        metadata = list(pool.map(process, enumerate(items)))

    with (out_dir / "metadata.jsonl").open("w") as handle:
        for item in metadata:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    return metadata


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--avcaps-dir", default="data/avcaps")
    parser.add_argument("--out-dir", default="data/avcaps200_semantic")
    parser.add_argument("--config", default="configs/xl_fast.json")
    parser.add_argument("--train-count", type=int, default=200,
                        help="0 = every eligible source video in the train division")
    parser.add_argument("--test-source", choices=("semantic", "official"), default="semantic",
                        help="official = AVCaps' own test division (disjoint sources)")
    parser.add_argument("--test-per-bucket", type=int, default=2,
                        help="semantic test source only; official uses every test video")
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--embedding-batch-size", type=int, default=64)
    parser.add_argument("--min-similarity", type=float, default=0.55)
    parser.add_argument("--max-similarity", type=float, default=0.98)
    parser.add_argument("--diversity-weight", type=float, default=0.15)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.workers < 1:
        raise ValueError("--workers must be at least 1")
    if args.test_per_bucket < 1:
        raise ValueError("--test-per-bucket must be at least 1")
    if not 0 <= args.min_similarity < args.max_similarity <= 1:
        raise ValueError("similarity bounds must satisfy 0 <= min < max <= 1")

    torch.manual_seed(args.seed)
    avcaps_dir = Path(args.avcaps_dir)
    out_dir = Path(args.out_dir)
    if out_dir.exists():
        raise FileExistsError(f"output already exists: {out_dir}")

    stats, caption_by_stem = eligible_sources(avcaps_dir, "train", args.workers)
    train_items = assign_bucket_items(stats, caption_by_stem, args.train_count or None)
    train_sources = {item["source"] for item in train_items}

    if args.test_source == "official":
        test_stats, test_captions = eligible_sources(avcaps_dir, "test", args.workers)
        test_items = assign_bucket_items(test_stats, test_captions)
    else:
        test_candidates = []
        for source_duration, path, width, height in stats:
            if path.name in train_sources:
                continue
            size = nearest_resolution(width, height)
            test_candidates.append({
                "path": path,
                "source": path.name,
                "source_duration": source_duration,
                "source_width": width,
                "source_height": height,
                "size": size,
                "text": caption_by_stem[path.stem][0],
                "texts": caption_by_stem[path.stem],
            })

        config = load_config(args.config)
        encoder = build_text_encoder(config, args.device)
        if not hasattr(encoder, "encode_pooled"):
            raise TypeError("semantic split requires a text encoder with encode_pooled()")
        test_items = select_semantic_test_items(
            test_candidates,
            train_items,
            encoder,
            config["text_encoder"]["text_len"],
            args.embedding_batch_size,
            args.test_per_bucket,
            args.min_similarity,
            args.max_similarity,
            args.diversity_weight,
        )
        del encoder
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    train_histogram = Counter(item["bucket"] for item in train_items)
    test_histogram = Counter(item["bucket"] for item in test_items)
    logger.info("train buckets: %s", dict(sorted(train_histogram.items())))
    logger.info("test buckets: %s", dict(sorted(test_histogram.items())))

    out_dir.mkdir(parents=True)
    train_metadata = write_split(train_items, out_dir / "train", args.fps, args.workers, "train")
    test_metadata = write_split(test_items, out_dir / "test", args.fps, args.workers, "test")
    manifest = {
        "seed": args.seed,
        "config": args.config,
        "test_source": args.test_source,
        "target_buckets": [
            bucket_key(duration, size) for duration, size in TARGET_BUCKETS
        ],
        "train_histogram": dict(sorted(train_histogram.items())),
        "test_histogram": dict(sorted(test_histogram.items())),
        "train": train_metadata,
        "test": test_metadata,
    }
    (out_dir / "split_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2)
    )
    logger.info("wrote %d train + %d test items to %s",
                len(train_metadata), len(test_metadata), out_dir)


if __name__ == "__main__":
    main()
