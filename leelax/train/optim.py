from __future__ import annotations
import torch
from torch import optim


def create_optimizer(model, lr=1e-3, weight_decay=1e-4):
    """Default optimizer: AdamW."""
    return optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
        betas=(0.9, 0.999)
    )


def create_scheduler(
    optimizer,
    name: str | None = None,
    total_steps: int = 1000,
    # backwards-compatible parameters used by tests:
    scheduler_type: str | None = None,
    T_max: int | None = None,
):
    """
    Creates LR schedulers — supports:
      - cosine
      - onecycle
      - warm_restarts
      - none (constant LR)

    Backwards compatible with old API used in tests:
      create_scheduler(opt, scheduler_type="cosine", T_max=10)
    """

    # ------------------------------
    # Backwards compatibility layer
    # ------------------------------
    if scheduler_type is not None:
        name = scheduler_type.lower()

    if T_max is not None:
        total_steps = T_max

    if name is None or name == "none":
        # Constant LR fallback
        class _NoSched:
            def step(self): 
                pass
        return _NoSched()

    name = name.lower()

    # ------------------------------
    # Modern schedulers
    # ------------------------------
    if name == "cosine":
        return optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=total_steps
        )

    if name == "onecycle":
        return optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=optimizer.param_groups[0]["lr"],
            total_steps=total_steps,
            pct_start=0.15,
            anneal_strategy="cos",
            div_factor=10.0,
            final_div_factor=100.0,
        )

    if name in ("warm", "warm_restarts", "sgdr"):
        return optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=max(1, total_steps // 4),
            T_mult=2
        )

    # Fallback
    class _NoSched:
        def step(self): 
            pass

    return _NoSched()

