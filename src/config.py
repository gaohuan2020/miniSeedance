"""
Project config: model architecture, text encoder, video VAE, audio VAE and
the whole training recipe live in one JSON file (see configs/default.json).
Missing `train` keys fall back to TRAIN_DEFAULTS; any key can be overridden
from the command line with dotted paths, e.g.
    --set train.lr=3e-4 --set model.depth=24 --set train.wandb.enabled=false
Checkpoints store the resolved config so sampling/eval rebuild the exact
same components.
"""

import copy
import json
from pathlib import Path

DEFAULT_CONFIG = "configs/default.json"

TRAIN_DEFAULTS = {
    "objective": "selfflow",          # selfflow | fm
    "steps": 6000,
    "batch_size": 128,                # per GPU
    "optimizer": "adamw",             # adamw | muon (Muon on block weights + aux AdamW)
    "muon_lr": 1e-3,                  # Muon group lr (DiT-Muon default)
    "muon_momentum": 0.95,
    "lr": 1e-4,
    "lr_schedule": "constant",        # constant | cosine
    "lr_min": 0.0,                    # cosine decays to this floor
    "warmup": 0,
    "t_sampler": "lognorm",           # uniform | lognorm
    "t_logit_mean": 0.0,              # lognorm: sigmoid(randn * std + mean); t=1 is clean
    "t_logit_std": 1.0,               # data, so mean < 0 samples the noisier end more
    "t_shift": 2.95,                  # WAN2.2 video trainshift from the Self-Flow paper
    "t_low_snr": 0.05,                # paper: redraw 5% from the noisiest interval
    "t_low_snr_width": 0.05,          # that interval is t in [0, width] (t=0 is pure noise)
    # Optional clean-end mixture. Probabilities replace draws from the base
    # distribution; defaults preserve old checkpoints and the paper recipe.
    "t_clean_mid": 0.0,
    "t_clean_mid_range": [0.75, 0.90],
    "t_clean_high": 0.0,
    "t_clean_high_range": [0.90, 0.98],
    "cfg_dropout": 0.1,
    "video_weight": 1.0,
    "audio_weight": 1.0,
    "ema_decay": 0.9999,              # fixed momentum teacher used by Self-Flow
    "ema_warmup": 0,                  # paper uses the fixed decay from the first step
    "grad_clip": 1.0,
    "seed": 0,
    "peak_tflops": None,              # bf16 peak per GPU for MFU; None = look up by device name
    "compile": True,                  # torch.compile the train-step forward (student + EMA teacher)
    "grad_compress_bf16": True,       # DDP: all-reduce gradients in bf16
    # Sequence packing: several variable-shape samples share one fixed-size bin.
    # Unused tail slots are modality=PAD and are excluded from the loss.
    "pack_enabled": True,
    "pack_x_len": None,               # noised-token budget; None = batch_size * max sample
    "pack_max_seqs": None,            # max samples per bin; None = budget / min sample
    # Self-Flow options
    "mask_ratio": 0.1,                # video ratio (image: 0.25, audio: 0.5)
    "distill_weight": 0.8,            # paper's gamma
    "distill_warmup": 0,
    "student_layer": None,
    "teacher_layer": None,
    # Which weights --init-from reads. The raw student is what training
    # continues from; the EMA is smoother but lags, and a curriculum stage is
    # long enough to rebuild its own average.
    "init_from_ema": False,
    # logging / checkpoints
    "log_every": 100,
    "ckpt_every": 2000,
    "wandb": {"enabled": True, "project": "mini-agnes-video", "run_name": None},
    # periodic eval (rank 0); raw weights by default -- the EMA lags ~1k steps
    "sample_mode": "ODE",             # WAN2.2 evaluation uses 50-step ODE
    "sample_shift": 15.0,             # WAN2.2 sampleshift from the Self-Flow paper
    "gen_batch": 16,                  # clips per sampling call at eval time
    "decode_chunk": 4,                # clips per VAE decode call at eval time
    # Frechet / CLIP metrics. They score the held-out split whenever an
    # eval data-dir is given (metrics are then prefixed "test_"), else the
    # training data. num_real=None uses every clip of that split as the real
    # reference; generation stays at num_videos clips because it is the
    # expensive side.
    # Quantitative metrics use no CFG; qualitative video samples use CFG 5.
    "fid": {"every": 1000, "num_videos": 32, "num_real": None, "num_steps": 50,
            "cfg_scale": 1.0, "use_ema": True},
    "sample": {"every": 1000, "count": 4, "num_steps": 50, "cfg_scale": 5.0, "use_ema": True},
    # Optional held-out data-dir: dump one random item beside its generation.
    "test": {"every": 2000, "num_steps": 50, "cfg_scale": 5.0, "use_ema": True},
    # Held-out FM loss beside the training FM loss, per modality, at fixed
    # timesteps. A few forward passes, so it can run often; it is the signal to
    # stop on, unlike the Frechet metrics which need generation and are noisy at
    # a few dozen clips. Needs an eval data-dir.
    "heldout": {"every": 500, "clips_per_bucket": 4},
}


def _deep_merge(base, override):
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path=DEFAULT_CONFIG, overrides=None):
    """Load a config JSON, fill train defaults, apply --set overrides."""
    with open(Path(path)) as f:
        config = json.load(f)
    config["train"] = _deep_merge(TRAIN_DEFAULTS, config.get("train", {}))
    for item in overrides or []:
        apply_override(config, item)
    return config


def apply_override(config, item):
    """Apply one 'a.b.c=value' override in place; value is parsed as JSON
    when possible (numbers, booleans, null, lists), else kept as string."""
    key, _, raw = item.partition("=")
    if not _:
        raise ValueError(f"Override must look like key.path=value, got {item!r}")
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        value = raw
    node = config
    parts = key.strip().split(".")
    for part in parts[:-1]:
        node = node.setdefault(part, {})
    node[parts[-1]] = value


def model_kwargs(config):
    """DiT architecture kwargs (hidden_size / depth / num_heads / attention)."""
    m = config["model"]
    return dict(hidden_size=m["hidden_size"], depth=m["depth"], num_heads=m["num_heads"],
                attention=m.get("attention", "flash"), adaln=m.get("adaln", "per_block"),
                mhc=m.get("mhc", 0), pooled_text=m.get("pooled_text", False))


def resolve_ckpt_config(ckpt):
    """(model_kwargs, config) for a loaded checkpoint dict; the checkpoint
    carries the full config resolved at training time."""
    config = ckpt["config"]
    config["train"] = _deep_merge(TRAIN_DEFAULTS, config.get("train", {}))
    return model_kwargs(config), config
