"""DDP process-group helpers (torchrun launch)."""

import os

import torch
import torch.distributed as dist


def setup_dist():
    """Init NCCL process group under torchrun; no-op for single process.

    Returns (rank, world_size, local_rank)."""
    if int(os.environ.get("WORLD_SIZE", "1")) > 1:
        dist.init_process_group("nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        return dist.get_rank(), dist.get_world_size(), local_rank
    return 0, 1, 0


def cleanup_dist():
    if dist.is_initialized():
        dist.destroy_process_group()
