"""Distributed training helpers for codec train scripts."""

from __future__ import annotations

import os
from typing import Tuple

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


def setup_distributed() -> Tuple[int, int, int, bool]:
    """Initialize NCCL process group when launched via torchrun.

    Returns (rank, local_rank, world_size, is_distributed).
    """
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        return 0, 0, 1, False

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    if not torch.cuda.is_available():
        raise RuntimeError("DDP requires CUDA")

    torch.cuda.set_device(local_rank)
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    return rank, local_rank, world_size, True


def cleanup_distributed(is_distributed: bool) -> None:
    if is_distributed and dist.is_initialized():
        dist.destroy_process_group()


def is_main_process(rank: int) -> bool:
    return rank == 0


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    if isinstance(model, DDP):
        return model.module
    return model


def wrap_ddp(
    model: torch.nn.Module,
    local_rank: int,
    *,
    find_unused_parameters: bool = False,
) -> DDP:
    return DDP(
        model,
        device_ids=[local_rank],
        output_device=local_rank,
        find_unused_parameters=find_unused_parameters,
    )


def reduce_scalar(value: float, device: torch.device, is_distributed: bool) -> float:
    if not is_distributed:
        return value
    tensor = torch.tensor(value, device=device, dtype=torch.float64)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return (tensor / dist.get_world_size()).item()


def barrier(is_distributed: bool) -> None:
    if is_distributed and dist.is_initialized():
        dist.barrier()
