"""
Precompute latents so training never runs the (large) frozen encoders:
video VAE latents, text token features, and (when the dataset has an `audio`
key) normalized audio VAE latents.

Videos are grouped into resolution buckets: every distinct (frames shape,
audio length) combination becomes one bucket, so a dataset can mix
resolutions/durations and training samples each batch from a single bucket.

Outputs in <data-dir>/cache/:
    latents.pt        {"buckets": [(Ni, C, T', H', W') fp16, ...],
                       "index": [(Ni,) original item indices, ...]}
    texts.pt          {"captions", "features" (U, L, D), "index" (N,), "null",
                       "text_encoder": config section used,
                       "caption_ids" (N, K), "caption_counts" (N,) when items
                       carry more than one caption}
    audio_latents.pt  {"buckets": [(Ni, C_a, L_a_i) fp16 normalized, ...],
                       "mean", "std", "sample_rate"}   (aligned with video buckets)
"""

import json
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from src.log import get_logger
from src.models import build_audio_vae, build_text_encoder, build_video_vae

logger = get_logger(__name__)


def build_text_cache(items, config, device, text_len):
    """Encode every caption of every item into the texts.pt payload.

    An item may list several descriptions of the same clip under "texts";
    "text" stays the primary one and is what evaluation prompts with, so its
    metrics never move because the prompt changed. Captions are deduplicated
    across the split, and the per-item id table is padded with the primary id.
    """
    text_encoder = build_text_encoder(config, device)
    per_item = [list(dict.fromkeys([it["text"], *it.get("texts", [])])) for it in items]
    captions = sorted({c for caps in per_item for c in caps})
    cap_to_idx = {c: i for i, c in enumerate(captions)}
    features = []
    for i in range(0, len(captions), 256):
        features.append(
            text_encoder.encode_tokens(captions[i : i + 256], max_length=text_len).cpu()
        )
    index = torch.tensor([cap_to_idx[it["text"]] for it in items], dtype=torch.long)
    out = {
        "captions": captions,
        "features": torch.cat(features),
        "index": index,
        "null": text_encoder.encode_tokens([""], max_length=text_len).cpu(),  # for CFG
        "text_encoder": config["text_encoder"],
    }
    width = max(len(caps) for caps in per_item)
    if width > 1:
        caption_ids = index.unsqueeze(1).repeat(1, width)
        for i, caps in enumerate(per_item):
            caption_ids[i, : len(caps)] = torch.tensor([cap_to_idx[c] for c in caps])
        out["caption_ids"] = caption_ids
        out["caption_counts"] = torch.tensor([len(c) for c in per_item], dtype=torch.long)
    return out


def build_cache(data_dir, config, batch_size=32, device="cuda"):
    text_len = config["text_encoder"]["text_len"]
    data_dir = Path(data_dir)
    cache_dir = data_dir / "cache"
    cache_dir.mkdir(exist_ok=True)

    items = [json.loads(l) for l in (data_dir / "metadata.jsonl").read_text().splitlines() if l.strip()]
    logger.info("%d videos", len(items))

    # ---- bucket by (frames shape, audio length) ----
    has_audio = "audio" in np.load(data_dir / items[0]["file_name"]).files
    buckets = {}  # key -> list of item indices
    for i, it in enumerate(items):
        with np.load(data_dir / it["file_name"]) as z:
            key = (z["frames"].shape, z["audio"].shape if has_audio else None)
        buckets.setdefault(key, []).append(i)
    bucket_items = sorted(buckets.items(), key=lambda kv: -len(kv[1]))  # largest first
    logger.info("%d resolution buckets: %s", len(bucket_items),
                [(k[0], len(v)) for k, v in bucket_items])

    # ---- video VAE, per bucket ----
    vae = build_video_vae(config, device)
    video_buckets, index_buckets = [], []
    for (fshape, _), idxs in tqdm(bucket_items, desc="VAE encode (buckets)"):
        lat = []
        for j in range(0, len(idxs), batch_size):
            videos = []
            for i in idxs[j : j + batch_size]:
                frames = np.load(data_dir / items[i]["file_name"])["frames"]  # (T, H, W, 3) uint8
                v = torch.from_numpy(frames).float() / 127.5 - 1.0
                videos.append(v.permute(3, 0, 1, 2))  # (3, T, H, W)
            lat.append(vae.encode(torch.stack(videos).to(device)).to(torch.float16).cpu())
        video_buckets.append(torch.cat(lat))
        index_buckets.append(torch.tensor(idxs, dtype=torch.long))
    torch.save({"buckets": video_buckets, "index": index_buckets}, cache_dir / "latents.pt")
    logger.info("latents: %s -> %s", [tuple(b.shape) for b in video_buckets],
                cache_dir / "latents.pt")

    # ---- audio VAE, per bucket (joint audio-video training) ----
    if has_audio:
        del vae
        torch.cuda.empty_cache()
        audio_vae = build_audio_vae(config, device)
        audio_buckets = []
        for (_, _), idxs in tqdm(bucket_items, desc="Audio VAE encode (buckets)"):
            lat = []
            for j in range(0, len(idxs), batch_size):
                waves = []
                for i in idxs[j : j + batch_size]:
                    audio = np.load(data_dir / items[i]["file_name"])["audio"]  # (2, N) int16
                    waves.append(torch.from_numpy(audio).float() / 32767.0)
                lat.append(audio_vae.encode(torch.stack(waves).to(device)).to(torch.float16).cpu())
            audio_buckets.append(torch.cat(lat).float())
        # Global per-channel stats over all buckets (weighted by element count)
        flat = torch.cat([b.permute(1, 0, 2).reshape(b.shape[1], -1) for b in audio_buckets], dim=1)
        mean = flat.mean(dim=1).view(1, -1, 1)
        # Relative floor keeps near-empty channels (e.g. silent STFT bins)
        # from being amplified into pure-noise diffusion targets.
        std = flat.std(dim=1).view(1, -1, 1)
        std = std.clamp_min(0.05 * std.max()).clamp_min(1e-4)
        audio_buckets = [((b - mean) / std).to(torch.float16) for b in audio_buckets]
        torch.save(
            {"buckets": audio_buckets, "mean": mean, "std": std,
             "sample_rate": audio_vae.sample_rate},
            cache_dir / "audio_latents.pt",
        )
        logger.info("audio latents: %s -> %s", [tuple(b.shape) for b in audio_buckets],
                    cache_dir / "audio_latents.pt")

    # ---- unique captions ----
    texts = build_text_cache(items, config, device, text_len)
    torch.save(texts, cache_dir / "texts.pt")
    logger.info("%d unique captions (%.1f per clip), features %s -> %s",
                len(texts["captions"]),
                float(texts["caption_counts"].float().mean())
                if "caption_counts" in texts else 1.0,
                tuple(texts["features"].shape), cache_dir / "texts.pt")


def main():
    """CLI: python -m src.data.processing.latent_caching --data-dir data/toy_video"""
    import argparse

    from src.config import DEFAULT_CONFIG, load_config

    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--config", type=str, default=DEFAULT_CONFIG)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    build_cache(args.data_dir, load_config(args.config), batch_size=args.batch_size)


if __name__ == "__main__":
    main()
