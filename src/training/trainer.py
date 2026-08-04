"""
Config-driven trainer for the text-to-video (or joint audio-video) DiT.

Handles: DDP (torchrun), EMA, LR schedule, wandb per-step logging, periodic
frame-FID and sample dumps (rank 0, raw weights by default), checkpointing.
All knobs come from the config's `train` section (see src/config.py
TRAIN_DEFAULTS).
"""

import copy
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP

from src.data.latent_cache import LatentCache
from src.inference import sample_latents
from src.log import get_logger
from src.media import save_mp4, save_wav, to_uint8_frames
from src.models import build_audio_vae, build_dit, build_video_vae
from src.training.distributed import cleanup_dist, setup_dist
from src.training.fid import AudioFAD, CLIPScoreMetric, FrameFID, VideoFVD
from src.training.objectives import OBJECTIVES
from src.training.optim import build_optimizer
from src.training.probes import LossProbe
from src.utils import PAD, TokenLayout, build_pack_plan, pack_noised, pack_text

logger = get_logger(__name__)


@torch.no_grad()
def update_ema(ema_model, model, decay=0.999):
    ema_params = dict(ema_model.named_parameters())
    names, params = zip(*model.named_parameters())
    targets = [ema_params[n] for n in names]
    torch._foreach_mul_(targets, decay)
    torch._foreach_add_(targets, list(params), alpha=1 - decay)


class Trainer:
    def __init__(self, config, data_dir, results_dir, eval_data_dir=None,
                 init_from=None):
        self.config = config
        self.tcfg = config["train"]
        self.rank, self.world_size, local_rank = setup_dist()
        self.is_main = self.rank == 0
        self.device = f"cuda:{local_rank}"
        torch.manual_seed(self.tcfg["seed"] + self.rank)

        self.results_dir = Path(results_dir)
        if self.is_main:
            self.results_dir.mkdir(parents=True, exist_ok=True)

        # ---- data ----
        self.data = LatentCache(data_dir, device=self.device)
        # The cache records which text encoder produced the features; the
        # checkpoint must carry that one so sampling rebuilds a match.
        if self.data.text_encoder_cfg is not None:
            config["text_encoder"] = self.data.text_encoder_cfg
        # One TokenLayout per resolution bucket; batches are shape-homogeneous.
        # self.layout (primary bucket) drives eval sampling and checkpoints.
        c_aud, l_a = self.data.audio_dims
        self.layouts = [
            TokenLayout(
                shape, config["model"]["patch_size"],
                patch_size_t=config["model"].get("patch_size_t", 1),
                audio_channels=self.data.bucket_audio_dims(i)[0],
                audio_len=self.data.bucket_audio_dims(i)[1],
            )
            for i, shape in enumerate(self.data.latent_shapes)
        ]
        assert len({lo.token_dim for lo in self.layouts}) == 1, \
            "buckets must share latent channels / patch size (token width)"
        self.layout = self.layouts[self.data.primary]
        self.x_ids_buckets = [lo.x_ids(self.device) for lo in self.layouts]
        self.test_data = None
        self.test_layouts = []
        if self.is_main and eval_data_dir:
            self.test_data = LatentCache(eval_data_dir, device=self.device)
            if self.test_data.text_dim != self.data.text_dim:
                raise ValueError("train/test text feature dimensions do not match")
            if self.test_data.has_audio != self.data.has_audio:
                raise ValueError("train/test audio modalities do not match")
            self.test_layouts = [
                TokenLayout(
                    shape, config["model"]["patch_size"],
                    patch_size_t=config["model"].get("patch_size_t", 1),
                    audio_channels=self.test_data.bucket_audio_dims(i)[0],
                    audio_len=self.test_data.bucket_audio_dims(i)[1],
                )
                for i, shape in enumerate(self.test_data.latent_shapes)
            ]
            if any(layout.token_dim != self.layout.token_dim for layout in self.test_layouts):
                raise ValueError("train/test video token dimensions do not match")

        # ---- model / EMA / DDP / optimizer / objective ----
        self.model = build_dit(
            config, self.device,
            video_in_channels=self.layout.token_dim,
            text_in_channels=self.data.text_dim,
            audio_in_channels=c_aud or None,
        )
        self.model.seq_l_v = self.layout.l_v  # static split; keeps compile graph-break-free
        if init_from:
            self._init_weights_from(init_from)
        self.ema = copy.deepcopy(self.model).eval().requires_grad_(False)

        objective_cls = OBJECTIVES[self.tcfg["objective"]]
        self.objective = objective_cls(
            self.tcfg, config["model"]["depth"],
            video_channels=self.layout.token_dim,
            audio_channels=c_aud or 0,
            moe_num_experts=config["model"].get("moe_num_experts", 0),
        )

        # Sequence-packing budget: static across steps so torch.compile sees one
        # graph. By default batch_size counts longest-sample equivalents, so a
        # bin holds that many long clips or proportionally more short ones.
        text_len = self.data.text_features.shape[1]
        max_x = max(lo.seq_len for lo in self.layouts)
        min_x = min(lo.seq_len for lo in self.layouts)
        self.pack_enabled = bool(self.tcfg.get("pack_enabled", True))
        self.pack_x_len = int(self.tcfg.get("pack_x_len") or max_x * self.tcfg["batch_size"])
        # Capacity has to let an all-short bin spend the whole budget, or the
        # packer runs out of slots and leaves tokens unused.
        self.pack_max_seqs = int(
            self.tcfg.get("pack_max_seqs") or -(-self.pack_x_len // min_x)
        )
        # FA4 max_seqlen is the longest possible joint segment (noised + text).
        self.pack_max_seqlen = max_x + text_len
        self.text_len = text_len
        self.probe = LossProbe(
            # The clip axis is indexed by cached item, not by caption: two items
            # can share a caption, and then the two counts differ.
            n_bins=8, n_buckets=self.data.num_buckets,
            n_clips=len(self.data), device=self.device,
        )
        if self.pack_enabled:
            logger.info(
                "packing: x_len=%d max_seqs=%d max_seqlen=%d text_len=%d",
                self.pack_x_len, self.pack_max_seqs, self.pack_max_seqlen, text_len,
            )

        if self.world_size > 1:
            self.ddp_model = DDP(
                self.model, device_ids=[local_rank],
                find_unused_parameters=(
                    self.objective.needs_unused_param_handling
                    or bool(config["model"].get("moe_num_experts", 0))
                ),
                gradient_as_bucket_view=True,
            )
            if self.tcfg["grad_compress_bf16"]:
                from torch.distributed.algorithms.ddp_comm_hooks import default_hooks

                self.ddp_model.register_comm_hook(state=None, hook=default_hooks.bf16_compress_hook)
        else:
            self.ddp_model = self.model

        # Compiled wrappers for the hot path; raw modules stay the source of
        # truth for state_dict / EMA updates / eval-time sampling.
        self.ema_fwd = self.ema
        if self.tcfg["compile"]:
            # Student and EMA each specialize per bucket. Multi-resolution
            # datasets can exceed Dynamo's default limit of eight variants.
            from torch import _dynamo

            _dynamo.config.recompile_limit = max(
                _dynamo.config.recompile_limit,
                max(32, 4 * len(self.layouts)),
            )
            mode = self.tcfg.get("compile_mode")  # e.g. "max-autotune-no-cudagraphs"
            # dynamic=False: specialize one static graph per resolution bucket
            # (dynamic-shape codegen is both slower and buggy for our backward).
            self.ddp_model = torch.compile(self.ddp_model, mode=mode, dynamic=False)
            self.ema_fwd = torch.compile(self.ema, mode=mode, dynamic=False)

        self.opt = build_optimizer(self.model, self.tcfg)

        # Cached parameter lists: rebuilding them every step (named_parameters
        # walk) costs a few ms at XL scale.
        self._params = list(self.model.parameters())
        ema_named = dict(self.ema.named_parameters())
        names, sources = zip(*self.model.named_parameters())
        self._ema_lists = ([ema_named[n] for n in names], list(sources))

        self.n_params = sum(p.numel() for p in self.model.parameters())
        self._flops_per_step = [self._estimate_step_flops(config["model"], lo)
                                for lo in self.layouts]
        self._peak_flops = self._lookup_peak_flops()

        m = config["model"]
        logger.info("latents: %d videos in %d buckets %s, %d unique captions | "
                    "world_size=%d, global batch=%d",
                    len(self.data), self.data.num_buckets, self.data.latent_shapes,
                    len(self.data.captions),
                    self.world_size, self.tcfg["batch_size"] * self.world_size)
        logger.info("%s(%dx%d): %.1fM params, %d video + %d audio + %d text tokens",
                    m["type"], m["hidden_size"], m["depth"],
                    sum(p.numel() for p in self.model.parameters()) / 1e6,
                    self.layout.l_v, l_a, self.data.text_features.shape[1])
        logger.info("objective: %s", self.objective.describe())
        logger.info("sampling: %s, %d steps (fid, cfg=%s) / %d steps (samples, cfg=%s), "
                    "shift=%s",
                    self.tcfg["sample_mode"], self.tcfg["fid"]["num_steps"],
                    self.tcfg["fid"]["cfg_scale"], self.tcfg["sample"]["num_steps"],
                    self.tcfg["sample"]["cfg_scale"], self.tcfg["sample_shift"],
                    )
        if self.test_data is not None:
            logger.info("held-out eval: %d videos in %d buckets from %s",
                        len(self.test_data), self.test_data.num_buckets, eval_data_dir)
            if self.tcfg["heldout"]["every"]:
                logger.info(
                    "held-out loss every %d steps, %d clips per bucket per split | "
                    "learned-nothing reference %s",
                    self.tcfg["heldout"]["every"],
                    self.tcfg["heldout"]["clips_per_bucket"],
                    " ".join(f"t{t:.2f} {e:.3f}"
                             for t, e in zip(self.HELDOUT_TS, self.BLIND_ERROR)),
                )

        # ---- wandb / eval components (rank 0 only) ----
        self.wb = self._init_wandb() if self.is_main else None
        self.vae = self.audio_vae = None
        self.fid_metric = self.fvd_metric = self.clip_metric = self.fad_metric = None
        self._real_fid_feats = self._real_fvd_feats = self._real_fad_feats = None
        self._gen_groups = None
        self._paired_real_stats = {}  # (bucket, position) -> (mu, sigma) of that clip
        # Metrics score the held-out split when there is one; generalization is
        # the question, and the training split can only answer how well it fit.
        if self.test_data is not None:
            self.eval_cache, self.eval_layouts = self.test_data, self.test_layouts
            self.eval_split, self.metric_prefix = "held-out", "test_"
        else:
            self.eval_cache, self.eval_layouts = self.data, self.layouts
            self.eval_split, self.metric_prefix = "train", ""
        test_enabled = self.test_data is not None and self.tcfg["test"]["every"]
        if self.is_main and (
            self.tcfg["fid"]["every"] or self.tcfg["sample"]["every"] or test_enabled
        ):
            self.vae = build_video_vae(config, self.device)
            if self.data.has_audio:
                self.audio_vae = build_audio_vae(config, self.device)
            if self.tcfg["fid"]["every"]:
                self.fid_metric = FrameFID(self.device)
                metric_classes = [("fvd_metric", VideoFVD), ("clip_metric", CLIPScoreMetric)]
                if self.data.has_audio:
                    metric_classes.append(("fad_metric", AudioFAD))
                for name, cls in metric_classes:
                    try:
                        setattr(self, name, cls(self.device))
                    except Exception as e:
                        logger.warning("%s unavailable (%s); skipping that metric", cls.__name__, e)
        self.eval_g = torch.Generator().manual_seed(self.tcfg["seed"] + 12345)

    OBJECTIVE_NAMES = {"selfflow": "self-flow", "fm": "flow-matching"}

    # Timesteps for the held-out loss readout: one deep in the noise, two
    # through the middle, one near clean data (t = 1 is clean here).
    HELDOUT_TS = (0.062, 0.312, 0.562, 0.812)
    # MSE of the best predictor of x1 - x0 that knows only that both are unit
    # Gaussians: 2 - (2t-1)^2 / (t^2 + (1-t)^2). A held-out error above this
    # means that branch is worse on unseen captions than knowing nothing.
    BLIND_ERROR = tuple(
        2.0 - (2 * t - 1) ** 2 / (t**2 + (1 - t) ** 2) for t in HELDOUT_TS
    )

    # Dense bf16 peak TFLOPS per GPU, matched by substring of the device name
    PEAK_BF16_TFLOPS = {
        "GB300": 2250, "B300": 2250, "GB200": 2250, "B200": 2250,
        "H200": 989, "H100": 989, "H800": 989, "A100": 312, "A800": 312,
    }

    def _estimate_step_flops(self, mcfg, layout):
        """Approx FLOPs per optimizer step per GPU (PaLM-style 6N + attention
        term for fwd+bwd, plus the EMA teacher forward for Self-Flow)."""
        s = layout.seq_len + self.data.text_features.shape[1]
        b = self.tcfg["batch_size"]
        lh = mcfg["depth"] * mcfg["hidden_size"]
        fwd_bwd = (6 * self.n_params + 12 * lh * s) * b * s
        teacher = (2 * self.n_params + 4 * lh * s) * b * s if self.tcfg["objective"] == "selfflow" else 0
        return fwd_bwd + teacher

    def _lookup_peak_flops(self):
        if self.tcfg["peak_tflops"]:
            return self.tcfg["peak_tflops"] * 1e12
        name = torch.cuda.get_device_name(self.device)
        for key, tflops in self.PEAK_BF16_TFLOPS.items():
            if key in name:
                return tflops * 1e12
        logger.warning("Unknown GPU %r for MFU; set train.peak_tflops to enable it", name)
        return None

    def _init_wandb(self):
        wcfg = self.tcfg["wandb"]
        if not wcfg["enabled"]:
            return None
        import wandb

        objective = self.OBJECTIVE_NAMES.get(self.tcfg["objective"], self.tcfg["objective"])
        name = wcfg["run_name"] or f"{objective}-{time.strftime('%m%d-%H%M')}"
        n_params = sum(p.numel() for p in self.model.parameters())
        kwargs = dict(project=wcfg["project"], name=name, tags=[objective],
                      config={**self.config, "world_size": self.world_size,
                              "model_params": n_params,
                              "model_params_m": round(n_params / 1e6, 1)})
        try:
            return wandb.init(**kwargs)
        except Exception as e:
            logger.warning("wandb online init failed (%s); falling back to offline mode", e)
            return wandb.init(mode="offline", **kwargs)

    def _lr_scale(self, step):
        """Relative schedule multiplier (1.0 at peak): linear warmup, then
        constant or cosine decay to lr_min/lr. Applied to every group's base_lr
        so Muon and AdamW groups keep their own peak lrs."""
        tcfg = self.tcfg
        if tcfg["warmup"] and step < tcfg["warmup"]:
            return (step + 1) / tcfg["warmup"]
        if tcfg["lr_schedule"] == "cosine":
            prog = (step - tcfg["warmup"]) / max(1, tcfg["steps"] - tcfg["warmup"])
            floor = tcfg["lr_min"] / tcfg["lr"]
            return floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * prog))
        return 1.0

    def _ema_decay(self, step):
        """EMA decay, ramped in over the first steps.

        A constant 0.999 leaves 0.999**step of the random init in the EMA -- 13%
        at step 2000 -- and Self-Flow distills the student against exactly those
        weights, with the distill term reaching full strength over the same
        span. Ramping the decay drains the init within a few dozen steps and
        still converges to the configured value.
        """
        decay = self.tcfg["ema_decay"]
        warmup = self.tcfg["ema_warmup"]
        if not warmup:
            return decay
        return min(decay, (1 + step) / (warmup + step))

    # ---- periodic eval (rank 0) ----

    def _decode_frames(self, latents):
        """Video latents -> uint8 (B, T, H, W, 3), a few clips per VAE call.

        The Wan VAE holds a per-frame feature cache across its temporal loop, so
        decoding a whole eval batch at once takes several times the memory of
        the training step it runs beside.
        """
        chunk = self.tcfg["decode_chunk"]
        return np.concatenate([
            to_uint8_frames(self.vae.decode(latents[i : i + chunk]))
            for i in range(0, latents.shape[0], chunk)
        ])

    def _decode_waves(self, cache, latents):
        """Audio latents, in `cache`'s normalization, -> waveforms (B, 2, N).

        Which cache matters: generated latents live in the *training* cache's
        normalization (that is the space the model learned), real held-out ones
        in the held-out cache's.
        """
        audio = cache.audio
        latents = latents * audio["std"].to(self.device) + audio["mean"].to(self.device)
        return self.audio_vae.decode(latents).float().clamp(-1, 1).cpu().numpy()

    def _eval_groups(self, cache, count=None, seed_offset=777):
        """Fixed eval items as [(bucket_id, positions)], covering `count` clips.

        The draw happens once and every eval reuses it: redrawing each time
        mixes item variance into the metric trend, which then moves for reasons
        that have nothing to do with the weights. Drawing over the flattened
        cache keeps buckets in proportion to their size. `count=None` takes
        every clip.
        """
        pairs = [(bi, p) for bi, b in enumerate(cache.buckets) for p in range(b.shape[0])]
        if count is None or count >= len(pairs):
            chosen = range(len(pairs))
        else:
            g = torch.Generator().manual_seed(self.tcfg["seed"] + seed_offset)
            chosen = sorted(torch.randperm(len(pairs), generator=g)[:count].tolist())
        grouped = {}
        for i in chosen:
            bucket_id, position = pairs[i]
            grouped.setdefault(bucket_id, []).append(position)
        return [(bucket_id, torch.tensor(ps)) for bucket_id, ps in sorted(grouped.items())]

    def _group_latents(self, cache, bucket_id, pos):
        """(video latents, audio latents | None) of one bucket's eval positions."""
        bucket = cache.buckets[bucket_id]
        idx = pos.to(bucket.device)
        audio = None
        if cache.has_audio:
            audio = cache.audio["buckets"][bucket_id][idx].to(self.device, torch.float32)
        return bucket[idx].to(self.device, torch.float32), audio

    def _group_text_ids(self, cache, bucket_id, pos):
        """Caption ids of one bucket's eval positions."""
        items = cache.bucket_index[bucket_id][pos.to(cache.bucket_index[bucket_id].device)]
        return cache.text_index[items.cpu()]

    def _prepare_eval_set(self):
        """Pick the eval clips and featurize the real side (once per run)."""
        fcfg = self.tcfg["fid"]
        cache = self.eval_cache
        num_real = fcfg["num_real"]
        if num_real is None and self.test_data is None:
            # Falling back to the training split: match the generated count.
            # That cache can hold thousands of clips and decoding all of them
            # for a reference Gaussian is pure overhead.
            num_real = fcfg["num_videos"]
        self._gen_groups = self._eval_groups(cache, fcfg["num_videos"])
        real_groups = self._eval_groups(cache, num_real, seed_offset=778)
        fid_feats, fvd_feats, fad_feats = [], [], []
        for bucket_id, pos in real_groups:
            video, audio = self._group_latents(cache, bucket_id, pos)
            frames = self._decode_frames(video)
            fid_feats.append(self.fid_metric.features(frames.reshape(-1, *frames.shape[2:])))
            if self.fvd_metric is not None:
                fvd_feats.append(self.fvd_metric.features(frames))
            if self.fad_metric is not None and audio is not None:
                waves = self._decode_waves(cache, audio)
                fad_feats.append(self.fad_metric.features(waves, cache.audio["sample_rate"]))
        self._real_fid_feats = np.concatenate(fid_feats)
        self._real_fvd_feats = np.concatenate(fvd_feats) if fvd_feats else None
        self._real_fad_feats = np.concatenate(fad_feats) if fad_feats else None
        logger.info(
            "%s eval set: %d generated clips vs %d real clips, %d buckets %s",
            self.eval_split, sum(len(p) for _, p in self._gen_groups),
            sum(len(p) for _, p in real_groups), len(self._gen_groups),
            [cache.latent_shapes[b] for b, _ in self._gen_groups],
        )

    @torch.no_grad()
    def _generate_group(self, cache, layout, text_ids, num_steps, cfg_scale, seed, use_ema):
        """Generate one bucket's shape -> (uint8 frames, waves | None, captions).

        Defaults to raw weights: the EMA (decay 0.999) lags by ~1k steps, so
        EMA samples/FID are misleading early in training.
        """
        net = self.ema if use_ema else self.model
        was_training = net.training
        net.eval()
        net.seq_l_v = layout.l_v  # may be stale from another bucket's step
        batch = self.tcfg["gen_batch"]
        frames, waves = [], []
        for i in range(0, len(text_ids), batch):
            ids = text_ids[i : i + batch]
            cond = cache.text_features[ids.to(cache.text_features.device)].to(self.device)
            vid_lat, aud_lat = sample_latents(
                net, cond, cache.null_feature.to(self.device), layout, self.device,
                num_steps=num_steps, cfg_scale=cfg_scale,
                # Offset per chunk: one seed for all of them would hand every
                # chunk the same noise.
                seed=None if seed is None else seed + i,
                mode=self.tcfg["sample_mode"], shift=self.tcfg["sample_shift"],
            )
            frames.append(self._decode_frames(vid_lat))
            if aud_lat is not None and self.audio_vae is not None:
                waves.append(self._decode_waves(self.data, aud_lat))
        if was_training:
            net.train()
        captions = [cache.captions[i] for i in text_ids.tolist()]
        return np.concatenate(frames), (np.concatenate(waves) if waves else None), captions

    def _paired_fid_sum(self, cache, bucket_id, pos, fake_per_video):
        """Summed Frechet distance between each generation and its own target clip.

        The two frame sets show the same content here, so the covariance shapes
        match and the distance is free to go to zero -- which is what makes this
        readable per video, where a score against the pooled real set is not.
        """
        total = 0.0
        for i, position in enumerate(pos.tolist()):
            stats = self._paired_real_stats.get((bucket_id, position))
            if stats is None:
                video, _ = self._group_latents(cache, bucket_id, pos[i : i + 1])
                frames = self._decode_frames(video)
                mu, sigma = self.fid_metric.stats(self.fid_metric.features(frames[0]))
                stats = (mu, sigma.astype("float64"))
                self._paired_real_stats[(bucket_id, position)] = stats
            total += self.fid_metric.video_fid(*stats, fake_per_video[i])
        return total

    def _eval_fid(self, step):
        """Frechet / CLIP metrics, on the held-out split when one was given.

        Every bucket of the eval split is generated at its own shape, and the
        features are pooled across buckets, so the score covers the whole split
        rather than its largest resolution.
        """
        fcfg = self.tcfg["fid"]
        cache = self.eval_cache
        if self._real_fid_feats is None:
            self._prepare_eval_set()
        pooled, per_video, fvd_fake, fad_fake = [], [], [], []
        paired_total, clip_total, count = 0.0, 0.0, 0
        for bucket_id, pos in self._gen_groups:
            frames, waves, captions = self._generate_group(
                cache, self.eval_layouts[bucket_id],
                self._group_text_ids(cache, bucket_id, pos),
                fcfg["num_steps"], fcfg["cfg_scale"],
                # Fixed noise across evals: only the weights should move.
                self.tcfg["seed"] + 777, fcfg["use_ema"],
            )
            feats = self.fid_metric.features(frames.reshape(-1, *frames.shape[2:]))
            clips = list(feats.reshape(frames.shape[0], frames.shape[1], -1))
            pooled.append(feats)
            per_video.extend(clips)
            paired_total += self._paired_fid_sum(cache, bucket_id, pos, clips)
            if self.fvd_metric is not None:
                fvd_fake.append(self.fvd_metric.features(frames))
            if self.clip_metric is not None:
                clip_total += self.clip_metric.score(frames, captions) * len(captions)
            if self.fad_metric is not None and waves is not None:
                fad_fake.append(
                    self.fad_metric.features(waves, self.data.audio["sample_rate"])
                )
            count += len(captions)

        prefix = self.metric_prefix
        metrics = {
            f"{prefix}fid": self.fid_metric.mean_video_fid(self._real_fid_feats, per_video),
            f"{prefix}fid_pooled": self.fid_metric.fid(
                self._real_fid_feats, np.concatenate(pooled)
            ),
            # Each generation against the one clip its caption asked for. Unlike
            # `fid`, which scores a video's frames against the whole real set and
            # so keeps a large covariance-mismatch floor, this one can reach zero
            # and tracks whether the requested clip came back.
            f"{prefix}fid_paired": paired_total / max(1, count),
        }
        if fvd_fake:
            metrics[f"{prefix}fvd"] = VideoFVD.fvd(
                self._real_fvd_feats, np.concatenate(fvd_fake)
            )
        if self.clip_metric is not None:
            metrics[f"{prefix}clip_score"] = clip_total / max(1, count)
        if fad_fake:
            metrics[f"{prefix}fad"] = AudioFAD.fad(
                self._real_fad_feats, np.concatenate(fad_fake)
            )
        logger.info(
            "step %6d | %s (%d %s clips)", step,
            " | ".join(f"{k} {v:.2f}" for k, v in metrics.items()), count, self.eval_split,
        )
        if self.wb is not None:
            self.wb.log(metrics, step=step)

    def _dump_samples(self, step):
        """Training-prompt samples, for eyeballing fit at the primary bucket."""
        scfg = self.tcfg["sample"]
        _, text_ids = self.data.eval_items(scfg["count"], self.eval_g)
        frames, waves, captions = self._generate_group(
            self.data, self.layout, text_ids, scfg["num_steps"], scfg["cfg_scale"],
            self.tcfg["seed"] + step, scfg["use_ema"],
        )
        sample_dir = self.results_dir / "samples"
        sample_dir.mkdir(parents=True, exist_ok=True)
        media = []
        for i in range(len(frames)):
            slug = captions[i].replace(" ", "_")[:50]
            mp4_path = sample_dir / f"step{step:07d}_{i}_{slug}.mp4"
            wav_path = None
            if waves is not None:
                wav_path = sample_dir / f"step{step:07d}_{i}_{slug}.wav"
                save_wav(waves[i], wav_path, self.data.audio["sample_rate"])
            save_mp4(frames[i], mp4_path, wav_path=wav_path)
            if self.wb is not None:
                import wandb

                media.append(wandb.Video(str(mp4_path), caption=captions[i], format="mp4"))
        logger.info("step %6d | dumped %d samples to %s", step, len(frames), sample_dir)
        if self.wb is not None:
            self.wb.log({"samples": media}, step=step)

    @torch.no_grad()
    def _dump_test_pair(self, step):
        """Dump one random held-out clip beside its generation, same prompt.

        Scores for the held-out split come from _eval_fid over many clips; a
        single item is only worth looking at.
        """
        tcfg = self.tcfg["test"]
        item_index = int(torch.randint(len(self.test_data), (1,), generator=self.eval_g))
        real_latent, cond, real_audio_latent, bucket_id, item = self.test_data.item(
            item_index, self.device,
        )
        layout = self.test_layouts[bucket_id]
        caption = item["text"]
        source = item.get("source", f"item_{item_index:05d}")

        net = self.ema if tcfg["use_ema"] else self.model
        was_training = net.training
        net.eval()
        net.seq_l_v = layout.l_v
        fake_latent, fake_audio_latent = sample_latents(
            net,
            cond,
            self.test_data.null_feature.to(self.device),
            layout,
            self.device,
            num_steps=tcfg["num_steps"],
            cfg_scale=tcfg["cfg_scale"],
            seed=self.tcfg["seed"] + step,
            mode=self.tcfg["sample_mode"], shift=self.tcfg["sample_shift"],
        )
        if was_training:
            net.train()

        real_frames = self._decode_frames(real_latent)
        fake_frames = self._decode_frames(fake_latent)
        real_wave = fake_wave = None
        if fake_audio_latent is not None and self.audio_vae is not None:
            fake_wave = self._decode_waves(self.data, fake_audio_latent)
            if real_audio_latent is not None:
                real_wave = self._decode_waves(self.test_data, real_audio_latent)

        test_dir = self.results_dir / "test_samples"
        test_dir.mkdir(parents=True, exist_ok=True)
        stem = f"step{step:07d}_item{item_index:03d}_{Path(source).stem}"
        real_wav_path = fake_wav_path = None
        if real_wave is not None:
            real_wav_path = test_dir / f"{stem}_real.wav"
            save_wav(real_wave[0], real_wav_path, self.test_data.audio["sample_rate"])
        if fake_wave is not None:
            fake_wav_path = test_dir / f"{stem}_generated.wav"
            save_wav(fake_wave[0], fake_wav_path, self.data.audio["sample_rate"])
        real_path = test_dir / f"{stem}_real.mp4"
        fake_path = test_dir / f"{stem}_generated.mp4"
        save_mp4(real_frames[0], real_path, wav_path=real_wav_path)
        save_mp4(fake_frames[0], fake_path, wav_path=fake_wav_path)

        logger.info(
            "step %6d | dumped held-out item %d (%s) to %s",
            step, item_index, source, test_dir,
        )
        if self.wb is not None:
            import wandb

            self.wb.log({
                "test_item_index": item_index,
                "test_generated": wandb.Video(
                    str(fake_path), caption=f"generated | {caption}", format="mp4"
                ),
                "test_reference": wandb.Video(
                    str(real_path), caption=f"reference | {caption}", format="mp4"
                ),
            }, step=step)

    def _heldout_groups(self, cache, per_bucket):
        """The first `per_bucket` clips of every bucket.

        Per bucket rather than a draw over the whole split, so the training and
        held-out sides carry identical resolution/duration composition and the
        two rows differ only by which clips they hold.
        """
        return [
            (bi, torch.arange(min(per_bucket, bucket.shape[0])))
            for bi, bucket in enumerate(cache.buckets)
        ]

    @torch.no_grad()
    def _heldout_errors(self, cache, layouts, groups, t):
        """(video, audio) mean per-token squared error at timestep `t`.

        The raw module, not the compiled wrapper: this path passes x_ids instead
        of a pack plan and would cost a fresh graph per bucket.
        """
        sums = torch.zeros(2, device=self.device, dtype=torch.float64)
        counts = torch.zeros(2, device=self.device, dtype=torch.float64)
        chunk = self.tcfg["decode_chunk"]
        for bucket_id, pos in groups:
            layout = layouts[bucket_id]
            self.model.seq_l_v = layout.l_v
            text_ids = self._group_text_ids(cache, bucket_id, pos)
            cond = cache.text_features[text_ids.to(cache.text_features.device)]
            cond = cond.to(self.device)
            for i in range(0, len(pos), chunk):
                video, audio = self._group_latents(cache, bucket_id, pos[i : i + chunk])
                x1 = layout.pack(video, audio)
                # Same noise for both splits at the same (bucket, offset), so the
                # comparison is not carrying a difference in draws.
                g = torch.Generator(device=self.device).manual_seed(
                    self.tcfg["seed"] + 4242 + 97 * bucket_id + i
                )
                x0 = torch.randn(x1.shape, device=self.device, generator=g, dtype=x1.dtype)
                xt = t * x1 + (1 - t) * x0
                t_tok = torch.full(xt.shape[:2], t, device=self.device)
                x_ids = layout.x_ids(self.device).unsqueeze(0).expand(x1.shape[0], -1, -1)
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    pred = self.model(
                        xt, timesteps=1 - t_tok, x_ids=x_ids, vector=cond[i : i + chunk],
                    )
                err = (-pred.float() - (x1 - x0)) ** 2
                vid = err[:, : layout.l_v, : layout.token_dim]
                sums[0] += vid.sum()
                counts[0] += vid.numel()
                if layout.has_audio:
                    aud = err[:, layout.l_v :, : layout.audio_channels]
                    sums[1] += aud.sum()
                    counts[1] += aud.numel()
        return (sums / counts.clamp(min=1.0)).tolist()

    def _eval_heldout_loss(self, step):
        """The training FM error beside the held-out one, per modality.

        A few forward passes, orders of magnitude cheaper than the Frechet
        metrics, and a far more direct overfitting readout. Video and audio
        targets differ by ~50x in dimension (a 21-frame video latent holds 774k
        numbers, its audio 14k), so the audio branch can be memorizing while the
        video branch still generalizes -- which the single logged loss, weighted
        0.15 toward audio, cannot show. Compare each test number against
        BLIND_ERROR at that t: above it, that branch is worse on unseen captions
        than a predictor that knows nothing but the unit-Gaussian marginals.
        """
        was_training = self.model.training
        self.model.eval()
        per_split = {}
        for name, cache, layouts in (
            ("train", self.data, self.layouts),
            ("test", self.test_data, self.test_layouts),
        ):
            groups = self._heldout_groups(cache, self.tcfg["heldout"]["clips_per_bucket"])
            per_split[name] = [
                self._heldout_errors(cache, layouts, groups, t) for t in self.HELDOUT_TS
            ]
        self.model.seq_l_v = self.layout.l_v
        if was_training:
            self.model.train()

        log = {}
        for m, modality in enumerate(("video", "audio")):
            if modality == "audio" and not self.data.has_audio:
                continue
            cells, gaps, x1 = [], [], [[], []]
            for i, t in enumerate(self.HELDOUT_TS):
                tr, te = per_split["train"][i][m], per_split["test"][i][m]
                cells.append(f"t{t:.2f} {tr:.3f}/{te:.3f}")
                gaps.append(te / tr if tr > 0 else float("nan"))
                log[f"heldout/{modality}_train_t{t:.2f}"] = tr
                log[f"heldout/{modality}_test_t{t:.2f}"] = te
                # Same error, expressed on the clean latent instead of the
                # velocity. v = (x1 - x_t)/(1-t), so an error d in the implied x1
                # enters the loss as d/(1-t) -- up to 28x at the low-noise end,
                # which makes the raw per-t numbers rank noise levels backwards.
                # On this scale they are comparable across t and against the
                # sampler's endpoint error (tools/probe_latent_recon.py rel_mse),
                # where ~0.01 of data variance is the threshold for sharp frames.
                for j, v in enumerate((tr, te)):
                    x1[j].append(v * (1 - t) ** 2)
                log[f"heldout/{modality}_x1err_train_t{t:.2f}"] = x1[0][-1]
                log[f"heldout/{modality}_x1err_test_t{t:.2f}"] = x1[1][-1]
            gap = sum(gaps) / len(gaps)
            log[f"heldout/{modality}_gap"] = gap
            means = [sum(v) / len(v) for v in x1]
            log[f"heldout/{modality}_x1err_train"] = means[0]
            log[f"heldout/{modality}_x1err_test"] = means[1]
            logger.info(
                "step %6d | held-out %s loss train/test %s | gap %.2fx | x1err %.3f/%.3f",
                step, modality, " ".join(cells), gap, means[0], means[1])
        if self.wb is not None:
            self.wb.log(log, step=step)

    def _init_weights_from(self, path):
        """Start from another run's weights, with a fresh optimizer and step 0.

        Duration curriculum: nothing in the weights is tied to clip length --
        RoPE coordinates come from the batch and the video/audio token split is
        set per bucket -- so a model trained on short clips is a valid starting
        point for longer ones. The optimizer state is deliberately dropped: the
        new stage gets its own warmup and cosine decay.
        """
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
        self.model.load_state_dict(ckpt["ema" if self.tcfg["init_from_ema"] else "model"])
        if self.is_main:
            logger.info("initialized from %s (step %d, %s weights, %s)",
                        path, ckpt["step"],
                        "ema" if self.tcfg["init_from_ema"] else "raw",
                        ckpt.get("latent_shapes"))

    def _save_ckpt(self, step):
        ckpt = {
            "model": self.model.state_dict(),
            "ema": self.ema.state_dict(),
            "opt": self.opt.state_dict(),
            "config": self.config,
            "latent_shape": self.data.latent_shape,      # primary bucket (inference default)
            "latent_shapes": self.data.latent_shapes,    # all training buckets
            "audio": self.data.audio_meta(),
            "step": step,
        }
        path = self.results_dir / f"ckpt_{step:07d}.pt"
        torch.save(ckpt, path)
        torch.save(ckpt, self.results_dir / "ckpt_last.pt")
        logger.info("Saved checkpoint to %s", path)

    # ---- main loop ----

    def _log_probe(self, step):
        """Report where the loss still lives: noise level, bucket, single clips."""
        stats = self.probe.flush()
        bands = " ".join(
            f"{self.probe.t_band_label(i)}:{v:.3f}"
            for i, v in enumerate(stats["t"]) if v == v  # skip unvisited (NaN)
        )
        logger.info("step %6d | video loss by t %s", step, bands)
        x1_bands = " ".join(
            f"{self.probe.t_band_label(i)}:{v:.5f}"
            for i, v in enumerate(stats["t_x1"]) if v == v
        )
        logger.info("step %6d | video x1err by t %s", step, x1_bands)
        by_bucket = " ".join(
            f"{self.data.latent_shapes[i][1]}x{self.data.latent_shapes[i][2]}"
            f"x{self.data.latent_shapes[i][3]}:{v:.3f}"
            for i, v in enumerate(stats["bucket"]) if v == v
        )
        logger.info("step %6d | video loss by bucket %s", step, by_bucket)

        clips = [(v, i) for i, v in enumerate(stats["clip"]) if v == v]
        if clips:
            clips.sort(reverse=True)
            worst = " | ".join(
                f"{v:.3f} {self.data.caption_for(i)[:44]}" for v, i in clips[:3]
            )
            logger.info("step %6d | hardest clips: %s", step, worst)

        if self.wb is not None:
            log = {f"probe/loss_t_{self.probe.t_band_label(i)}": v
                   for i, v in enumerate(stats["t"]) if v == v}
            log.update({f"probe/x1err_t_{self.probe.t_band_label(i)}": v
                        for i, v in enumerate(stats["t_x1"]) if v == v})
            log.update({f"probe/loss_bucket_{i}": v
                        for i, v in enumerate(stats["bucket"]) if v == v})
            if clips:
                values = sorted(v for v, _ in clips)
                log["probe/clip_loss_max"] = values[-1]
                log["probe/clip_loss_min"] = values[0]
                log["probe/clip_loss_median"] = values[len(values) // 2]
            self.wb.log(log, step=step)

    def _estimate_pack_flops(self, mcfg, seg_lens):
        """FLOPs for one packed bin, given its joint segment lengths.

        Attention is block-diagonal over the bin, so its quadratic term follows
        the individual segments rather than the whole bin; charging the bin
        length would inflate MFU roughly by the number of samples packed.
        """
        joint_len = self.pack_x_len + self.pack_max_seqs * self.text_len
        pad = joint_len - sum(seg_lens)  # the tail is one self-attending segment
        attn_sq = sum(s * s for s in seg_lens) + pad * pad
        lh = mcfg["depth"] * mcfg["hidden_size"]
        fwd_bwd = 6 * self.n_params * joint_len + 12 * lh * attn_sq
        if self.tcfg["objective"] != "selfflow":
            return fwd_bwd
        return fwd_bwd + 2 * self.n_params * joint_len + 4 * lh * attn_sq

    def train(self):
        tcfg = self.tcfg
        self.ddp_model.train()
        # Sample indices are rank-local; packing shape decisions are synchronized
        # so every rank builds an identical PackPlan (required for DDP + compile).
        g = torch.Generator().manual_seed(tcfg["seed"] + self.rank)
        shape_g = torch.Generator().manual_seed(tcfg["seed"])
        running_loss, running, nb = 0.0, {}, 0
        step = 0
        t_start = time.perf_counter()
        if self.is_main and tcfg["compile"]:
            logger.info(
                "compile warmup: %s",
                "1 packed-bin graph (student + EMA)" if self.pack_enabled
                else f"{self.data.num_buckets} buckets (student + EMA graphs per bucket)",
            )

        while step < tcfg["steps"]:
            scale = self._lr_scale(step)
            lr = tcfg["lr"] * scale
            for group in self.opt.param_groups:
                group["lr"] = group["base_lr"] * scale

            step_t0 = time.perf_counter()

            if self.pack_enabled:
                tokens_list, text_list, layout_ids, item_ids = self.data.pack_bin(
                    self.pack_x_len, self.pack_max_seqs, shape_g, g, self.device, self.layouts,
                )
            else:
                item_ids = None
                # Homogeneous ablation: pack batch_size samples from one bucket.
                bi = self.data.sample_bucket(shape_g)
                x1, y, a1, bi = self.data.batch(
                    tcfg["batch_size"], g, self.device, bucket_id=bi,
                )
                layout = self.layouts[bi]
                tokens_list = list(layout.pack(x1, a1).unbind(0))
                text_list = list(y.unbind(0))
                layout_ids = [bi] * len(tokens_list)
                # Temporarily override the bin budget to an exact fit.
                x_len = layout.seq_len * len(tokens_list)
                max_seqs = len(tokens_list)
                max_seqlen = layout.seq_len + self.text_len

            # CFG dropout per sample inside the bin
            null = self.data.null_feature.to(self.device).squeeze(0)
            for i, r in enumerate(torch.rand(len(text_list), generator=g).tolist()):
                if r < tcfg["cfg_dropout"]:
                    text_list[i] = null

            if self.pack_enabled:
                x_len, max_seqs, max_seqlen = self.pack_x_len, self.pack_max_seqs, self.pack_max_seqlen
            x1_tok = pack_noised(tokens_list, x_len)
            y = pack_text(text_list, max_seqs, self.text_len)
            plan = build_pack_plan(
                [self.layouts[i] for i in layout_ids],
                text_len=self.text_len,
                x_len=x_len,
                max_seqs=max_seqs,
                max_seqlen=max_seqlen,
                device=self.device,
            )
            if self.pack_enabled:
                flops = self._estimate_pack_flops(
                    self.config["model"],
                    [self.layouts[i].seq_len + self.text_len for i in layout_ids],
                )
            else:
                flops = self._flops_per_step[layout_ids[0]]
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                loss, metrics, probe = self.objective(
                    self.ddp_model, self.ema_fwd, x1_tok, y, plan, step,
                )
            self.probe.update(
                probe["video_per_sample"], probe["t"],
                torch.tensor(layout_ids, device=self.device),
                torch.tensor(item_ids, device=self.device) if item_ids else None,
                x1_per_sample=probe["video_x1_per_sample"],
            )

            self.opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self._params, max_norm=tcfg["grad_clip"])
            self.opt.step()
            targets, sources = self._ema_lists
            ema_decay = self._ema_decay(step)
            torch._foreach_mul_(targets, ema_decay)
            torch._foreach_add_(targets, sources, alpha=1 - ema_decay)
            # single sync per step: accurate step timing + free .item()s after
            torch.cuda.synchronize(self.device)
            step_dt = time.perf_counter() - step_t0
            step += 1

            metrics = {k: float(v) for k, v in metrics.items()}
            running_loss += loss.item()
            for k, v in metrics.items():
                running[k] = running.get(k, 0.0) + v
            nb += 1

            if self.wb is not None:
                log = {
                    "loss": loss.item(), **metrics, "lr": lr, "ema_decay": ema_decay,
                    "train_time_s": time.perf_counter() - t_start,
                    "tflops_per_gpu": flops / step_dt / 1e12,
                    "pack_n_samples": len(layout_ids),
                    "pack_n_tokens": int((plan.modality != PAD).sum()),
                }
                if self._peak_flops:
                    log["mfu"] = flops / step_dt / self._peak_flops
                self.wb.log(log, step=step)

            if step % tcfg["log_every"] == 0:
                parts = " | ".join(f"{k} {v / nb:.4f}" for k, v in running.items())
                logger.info("step %6d | loss %.4f | %s", step, running_loss / nb, parts)
                running_loss, running, nb = 0.0, {}, 0
                self._log_probe(step)

            if (
                self.is_main
                and self.test_data is not None
                and tcfg["heldout"]["every"]
                and step % tcfg["heldout"]["every"] == 0
            ):
                self._eval_heldout_loss(step)
            if self.is_main and tcfg["fid"]["every"] and step % tcfg["fid"]["every"] == 0:
                self._eval_fid(step)
            if self.is_main and tcfg["sample"]["every"] and step % tcfg["sample"]["every"] == 0:
                self._dump_samples(step)
            if (
                self.is_main
                and self.test_data is not None
                and tcfg["test"]["every"]
                and step % tcfg["test"]["every"] == 0
            ):
                self._dump_test_pair(step)
            if self.is_main and (step % tcfg["ckpt_every"] == 0 or step == tcfg["steps"]):
                self._save_ckpt(step)

        total_time = time.perf_counter() - t_start
        if self.wb is not None:
            self.wb.summary["total_train_time_s"] = total_time
            self.wb.finish()
        cleanup_dist()
        logger.info("Done. Total training time: %.1f min", total_time / 60)
