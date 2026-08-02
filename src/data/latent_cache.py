"""In-memory view of a precomputed latent cache (see processing/latent_caching.py).

Latents are stored in resolution buckets (one tensor per distinct video
shape / audio length). batch() first picks a bucket (weighted by size), then
samples items inside it, so every batch is shape-homogeneous. Legacy
single-tensor caches load as one bucket.
"""

import json
from pathlib import Path

import torch


class LatentCache:
    """Loads <data_dir>/cache/{latents,texts,audio_latents}.pt and serves
    random training batches of (video latents, text features[, audio latents])."""

    # Caches up to this size are moved to the GPU once at load time; pageable
    # host->device copies inside batch() block the stream every step otherwise.
    # A couple of thousand clips land in the single-digit GB range, small next
    # to a training step's activations, so the bar sits above that.
    MAX_DEVICE_BYTES = 24 << 30

    def __init__(self, data_dir, device=None):
        data_dir = Path(data_dir)
        cache = data_dir / "cache"
        metadata_path = data_dir / "metadata.jsonl"
        self.items = [
            json.loads(line) for line in metadata_path.read_text().splitlines() if line.strip()
        ]
        raw = torch.load(cache / "latents.pt")
        if isinstance(raw, dict):  # bucketed format
            self.buckets = raw["buckets"]          # [(Ni, C, T, H, W) fp16]
            self.bucket_index = raw["index"]       # [(Ni,) original item idx]
        else:  # legacy: one tensor, one bucket
            self.buckets = [raw]
            self.bucket_index = [torch.arange(raw.shape[0])]

        texts = torch.load(cache / "texts.pt")
        self.text_features = texts["features"]                 # (U, L_t, D)
        self.text_index = texts["index"]                       # (N,) primary caption
        self.null_feature = texts["null"]                      # (1, L_t, D)
        self.captions = texts["captions"]
        self.text_encoder_cfg = texts.get("text_encoder")
        # Optional multi-caption index: (N, K) padded ids + per-item counts.
        # Datasets that describe each clip several times (AVCaps ships ~13) can
        # then hand training a different description every time a clip is drawn.
        self.caption_ids = texts.get("caption_ids")
        self.caption_counts = texts.get("caption_counts")

        audio_path = cache / "audio_latents.pt"
        self.audio = None
        if audio_path.exists():
            self.audio = torch.load(audio_path)
            if "buckets" not in self.audio:  # legacy single-tensor format
                self.audio["buckets"] = [self.audio.pop("latents")]

        # Bucket sampling weights and the primary (largest) bucket used for
        # eval sampling and checkpoint metadata.
        sizes = torch.tensor([b.shape[0] for b in self.buckets], dtype=torch.float)
        self._bucket_weights = sizes / sizes.sum()
        self.primary = int(sizes.argmax())

        if device is not None:
            nbytes = sum(b.nbytes for b in self.buckets) + self.text_features.nbytes
            if self.audio is not None:
                nbytes += sum(b.nbytes for b in self.audio["buckets"])
            if nbytes <= self.MAX_DEVICE_BYTES:
                self.buckets = [b.to(device) for b in self.buckets]
                self.bucket_index = [i.to(device) for i in self.bucket_index]
                self.text_features = self.text_features.to(device)
                self.text_index = self.text_index.to(device)
                if self.caption_ids is not None:
                    self.caption_ids = self.caption_ids.to(device)
                    self.caption_counts = self.caption_counts.to(device)
                if self.audio is not None:
                    self.audio = {**self.audio,
                                  "buckets": [b.to(device) for b in self.audio["buckets"]]}

    def __len__(self):
        return sum(b.shape[0] for b in self.buckets)

    @property
    def num_buckets(self):
        return len(self.buckets)

    @property
    def latent_shapes(self):
        """Per-bucket (C, T, H, W)."""
        return [tuple(b.shape[1:]) for b in self.buckets]

    @property
    def latent_shape(self):
        """(C, T, H, W) of the primary (largest) bucket."""
        return tuple(self.buckets[self.primary].shape[1:])

    @property
    def text_dim(self):
        return self.text_features.shape[-1]

    @property
    def has_audio(self):
        return self.audio is not None

    def bucket_audio_dims(self, bucket_id):
        """(C_a, L_a) of one bucket, or (0, 0) for video-only caches."""
        if self.audio is None:
            return 0, 0
        return tuple(self.audio["buckets"][bucket_id].shape[1:])

    @property
    def audio_dims(self):
        return self.bucket_audio_dims(self.primary)

    def audio_meta(self):
        """Checkpoint metadata needed to decode audio at sampling time."""
        if self.audio is None:
            return None
        c_aud, l_a = self.audio_dims
        return {
            "in_channels": c_aud,
            "latent_len": l_a,
            # Positionally paired with latent_shapes: audio length scales with a
            # bucket's duration, so picking another bucket at sampling time has
            # to pick its audio length too.
            "latent_lens": [self.bucket_audio_dims(i)[1] for i in range(self.num_buckets)],
            "mean": self.audio["mean"],
            "std": self.audio["std"],
            "sample_rate": self.audio["sample_rate"],
        }

    def caption_for(self, item_index):
        """Primary caption of one original item. Captions are deduplicated, so
        item indices and caption ids are not interchangeable."""
        return self.captions[int(self.text_index[item_index])]

    @property
    def captions_per_item(self):
        """Mean number of captions each clip can be conditioned on."""
        if self.caption_counts is None:
            return 1.0
        return float(self.caption_counts.float().mean())

    def sample_text_ids(self, items, generator):
        """Caption ids for `items`, one drawn uniformly per item.

        Training draws here: with several descriptions of the same clip, the
        caption -> clip map cannot be memorized outright, and the model has to
        settle on what the descriptions share. Evaluation keeps to `text_index`
        (the primary caption) so a metric never moves because the prompt did.
        """
        if self.caption_ids is None:
            return self.text_index[items]
        counts = self.caption_counts[items]
        draw = torch.rand(counts.shape, generator=generator).to(counts.device)
        return self.caption_ids[items, (draw * counts).long()]

    def eval_bucket(self):
        """(video latents, audio latents | None) of the primary bucket, for eval."""
        audio = self.audio["buckets"][self.primary] if self.audio is not None else None
        return self.buckets[self.primary], audio

    def eval_items(self, count, generator):
        """(positions in the primary bucket, text feature ids) for eval sampling.

        Eval always generates at the primary bucket's shape, so its captions have
        to come from that bucket. Drawing a caption from the whole dataset pairs
        clips with a resolution and duration they were never trained at, which
        the model has no reason to handle.
        """
        items = self.bucket_index[self.primary]
        pos = torch.randint(0, items.shape[0], (count,), generator=generator).to(items.device)
        return pos, self.text_index[items[pos].cpu()]

    def item(self, item_index, device):
        """Return one original metadata item and its aligned cached tensors."""
        for bucket_id, original_indices in enumerate(self.bucket_index):
            positions = (original_indices == item_index).nonzero(as_tuple=False)
            if positions.numel() == 0:
                continue
            position = int(positions[0])
            text_id = int(self.text_index[item_index])
            video = self.buckets[bucket_id][position : position + 1].to(device, torch.float32)
            text = self.text_features[text_id : text_id + 1].to(device)
            audio = None
            if self.audio is not None:
                audio = self.audio["buckets"][bucket_id][position : position + 1].to(
                    device, torch.float32
                )
            return video, text, audio, bucket_id, self.items[item_index]
        raise IndexError(f"item index {item_index} is not present in cache")

    def sample_bucket(self, generator):
        """Sample one bucket proportional to its number of items."""
        if len(self.buckets) == 1:
            return 0
        return int(torch.multinomial(self._bucket_weights, 1, generator=generator))

    def pack_bin(self, x_len, max_seqs, shape_generator, sample_generator, device, layouts):
        """First-fit pack of variable-shape samples into one noised-token bin.

        `shape_generator` must be identical across DDP ranks so every rank builds
        the same layout composition (required for a shared PackPlan / compile
        graph). `sample_generator` is rank-local and only picks indices inside
        each chosen bucket.

        Returns (tokens_list, text_list, layout_ids, item_ids) whose total noised
        length is <= x_len and whose count is <= max_seqs. `item_ids` index into
        `self.captions`, so a caller can attribute loss back to single clips.
        """
        # Buckets are drawn proportional to their item count so that every clip
        # gets seen equally often. Drawing uniformly over buckets instead would
        # show the single clip of a one-item bucket as often as a nine-item
        # bucket shows all nine, and those clips then dominate the gradient.
        order = torch.multinomial(
            self._bucket_weights, max_seqs * len(self.buckets),
            replacement=True, generator=shape_generator,
        ).tolist()
        tokens_list, text_list, layout_ids, item_ids = [], [], [], []
        used = 0
        min_seq = min(lo.seq_len for lo in layouts)
        for bi in order:
            lo = layouts[bi]
            if used + lo.seq_len > x_len or len(layout_ids) >= max_seqs:
                continue
            bucket = self.buckets[bi]
            idx = int(torch.randint(0, bucket.shape[0], (1,), generator=sample_generator))
            idx_t = torch.tensor([idx], device=bucket.device if bucket.is_cuda else "cpu")
            x1 = bucket[idx_t].to(device, torch.float32)
            item_t = self.bucket_index[bi][idx_t]
            item = int(item_t.item())
            y = self.text_features[self.sample_text_ids(item_t, sample_generator)][0]
            y = y.to(device)
            a1 = None
            if self.audio is not None:
                a1 = self.audio["buckets"][bi][idx_t].to(device, torch.float32)
            tokens_list.append(lo.pack(x1, a1)[0])
            text_list.append(y)
            layout_ids.append(bi)
            item_ids.append(item)
            used += lo.seq_len
            if used + min_seq > x_len or len(layout_ids) >= max_seqs:
                break
        assert tokens_list, "pack_bin produced an empty bin"
        return tokens_list, text_list, layout_ids, item_ids

    def batch(self, batch_size, generator, device, bucket_id=None):
        """Random shape-homogeneous batch:
        (video latents fp32, text features, audio latents | None, bucket_id)."""
        bi = self.sample_bucket(generator) if bucket_id is None else bucket_id
        bucket = self.buckets[bi]
        idx = torch.randint(0, bucket.shape[0], (batch_size,), generator=generator)
        if bucket.is_cuda:
            idx = idx.to(bucket.device)
        x1 = bucket[idx].to(device, torch.float32)
        items = self.bucket_index[bi][idx]
        y = self.text_features[self.sample_text_ids(items, generator)].to(device)
        a1 = None
        if self.audio is not None:
            a1 = self.audio["buckets"][bi][idx].to(device, torch.float32)  # (B, C_a, L_a)
        return x1, y, a1, bi
