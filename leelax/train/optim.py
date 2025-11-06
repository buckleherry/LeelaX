from __future__ import annotations
from typing import Optional

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler


def create_optimizer(
    model: torch.nn.Module,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    betas: tuple[float, float] = (0.9, 0.999),
) -> Optimizer:
    """Create Adam optimizer for the model."""
    return torch.optim.Adam(
        model.parameters(),
        lr=lr,
        betas=betas,
        weight_decay=weight_decay,
    )


def create_scheduler(
    optimizer: Optimizer,
    scheduler_type: str = "cosine",
    T_max: int = 1000,
    step_size: int = 200,
    gamma: float = 0.5,
) -> Optional[_LRScheduler]:
    """Factory for common schedulers."""
    if scheduler_type is None:
        return None

    scheduler_type = scheduler_type.lower()

    if scheduler_type == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max)
    elif scheduler_type == "step":
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    else:
        raise ValueError(f"Unknown scheduler_type: {scheduler_type}")

