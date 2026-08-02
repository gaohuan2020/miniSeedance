#!/usr/bin/env bash
# DDP training launcher for mini-agnes-video.
#
# Common knobs are environment variables; anything else can be overridden by
# forwarding --set key.path=value to train_video.py, e.g.:
#   bash train_ddp.sh
#   GPUS=2 STEPS=20000 OBJECTIVE=fm bash train_ddp.sh
#   WANDB_MODE=offline bash train_ddp.sh --set train.mask_ratio=0.25 --set model.depth=24
set -euo pipefail
cd "$(dirname "$0")"
# Prefer the project venv, fall back to the repo-level one
if [ -f .venv/bin/activate ]; then
    source .venv/bin/activate
elif [ -f ../.venv/bin/activate ]; then
    source ../.venv/bin/activate
fi

# ---- hardware ----
GPUS=${GPUS:-4}                              # number of GPUs
# ---- data / output / config ----
DATA_DIR=${DATA_DIR:-data/toy_video}
EVAL_DATA_DIR=${EVAL_DATA_DIR:-}
RESULTS_DIR=${RESULTS_DIR:-results/video_ddp}
CONFIG=${CONFIG:-configs/v2.json}            # v2 recipe: cosine+warmup+lognorm+distill warmup
# ---- optimization (BATCH_SIZE is per GPU; global = BATCH_SIZE * GPUS) ----
OBJECTIVE=${OBJECTIVE:-selfflow}
STEPS=${STEPS:-10000}
BATCH_SIZE=${BATCH_SIZE:-32}
LR=${LR:-1e-4}
# ---- logging / eval ----
WANDB_PROJECT=${WANDB_PROJECT:-mini-agnes-video}
# Run name defaults to "<self-flow|flow-matching>-<date>" (set by the trainer)
WANDB_RUN=${WANDB_RUN:-}
FID_EVERY=${FID_EVERY:-1000}
SAMPLE_EVERY=${SAMPLE_EVERY:-1000}

torchrun --standalone --nproc_per_node="$GPUS" train_video.py \
    --data-dir "$DATA_DIR" \
    ${EVAL_DATA_DIR:+--eval-data-dir "$EVAL_DATA_DIR"} \
    --results-dir "$RESULTS_DIR" \
    --config "$CONFIG" \
    --objective "$OBJECTIVE" \
    --steps "$STEPS" \
    --batch-size "$BATCH_SIZE" \
    --lr "$LR" \
    ${WANDB_RUN:+--wandb-run "$WANDB_RUN"} \
    --set train.wandb.project="$WANDB_PROJECT" \
    --set train.fid.every="$FID_EVERY" \
    --set train.sample.every="$SAMPLE_EVERY" \
    "$@"
