#!/usr/bin/env python3
"""
Train the text-to-video (or joint audio-video) DiT on precomputed latents.

Everything is config-driven (see configs/default.json and src/config.py
TRAIN_DEFAULTS); any key can be overridden with --set, plus a few common
convenience flags. Launch with torchrun for DDP (see train_ddp.sh).

Usage:
    python train_video.py --data-dir data/toy_video --results-dir results/video
    python train_video.py --data-dir data/toy_video --config configs/v2.json \
        --set train.objective=fm --set model.depth=24 --set train.wandb.enabled=false
    torchrun --standalone --nproc_per_node=4 train_video.py --data-dir data/toy_video
"""

import argparse
import json

from src.config import DEFAULT_CONFIG, apply_override, load_config
from src.training.trainer import Trainer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--eval-data-dir", type=str,
                        help="held-out cached dataset for periodic random-item evaluation")
    parser.add_argument("--results-dir", type=str, default="results/video")
    parser.add_argument("--config", type=str, default=DEFAULT_CONFIG)
    parser.add_argument("--set", dest="overrides", action="append", default=[],
                        metavar="KEY.PATH=VALUE",
                        help="config override, e.g. --set train.lr=3e-4 (repeatable)")
    # Convenience shortcuts for the most common knobs
    parser.add_argument("--init-from", type=str,
                        help="checkpoint to take weights from (fresh optimizer and "
                             "schedule); for training a duration curriculum in stages")
    parser.add_argument("--objective", type=str, choices=["fm", "selfflow"])
    parser.add_argument("--steps", type=int)
    parser.add_argument("--batch-size", type=int, help="per-GPU batch size")
    parser.add_argument("--lr", type=float)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--wandb-run", type=str)
    parser.add_argument("--no-wandb", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config, overrides=args.overrides)
    shortcuts = {
        "train.objective": args.objective,
        "train.steps": args.steps,
        "train.batch_size": args.batch_size,
        "train.lr": args.lr,
        "train.seed": args.seed,
        "train.wandb.run_name": args.wandb_run,
        "train.wandb.enabled": False if args.no_wandb else None,
    }
    for key, value in shortcuts.items():
        if value is not None:
            apply_override(config, f"{key}={json.dumps(value)}")

    Trainer(config, args.data_dir, args.results_dir, eval_data_dir=args.eval_data_dir,
            init_from=args.init_from).train()


if __name__ == "__main__":
    main()
