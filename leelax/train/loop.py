from __future__ import annotations

from typing import Dict, Any, Optional, Iterable

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from leelax.net.model import LeelaXNet
from leelax.selfplay.replay_buffer import ReplayBuffer
from leelax.train.losses import policy_loss_fn, value_loss_fn
from leelax.train.replay_dataset import make_replay_dataloader


def create_optimizer(model: torch.nn.Module, lr: float = 1e-3) -> torch.optim.Optimizer:
    return optim.Adam(model.parameters(), lr=lr)


def train_step_batch(
    model: LeelaXNet,
    batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    optimizer: torch.optim.Optimizer,
    device: str = "cpu",
    value_loss_weight: float = 1.0,
    policy_loss_weight: float = 1.0,
) -> Dict[str, Any]:
    """Train on an already prepared batch (states, policies, values)."""
    model.train()
    states, policies, values = batch
    states = states.to(device)
    policies = policies.to(device)
    values = values.to(device)

    optimizer.zero_grad()
    policy_logits, pred_values = model(states)

    p_loss = policy_loss_fn(policy_logits, policies)
    v_loss = value_loss_fn(pred_values, values)
    loss = policy_loss_weight * p_loss + value_loss_weight * v_loss
    loss.backward()
    optimizer.step()

    return {
        "loss": float(loss.item()),
        "policy_loss": float(p_loss.item()),
        "value_loss": float(v_loss.item()),
    }


def train_for_n_steps(
    model: LeelaXNet,
    replay: ReplayBuffer,
    n_steps: int = 100,
    device: str = "cpu",
    lr: float = 1e-3,
    batch_size: int = 64,
    num_workers: int = 0,
) -> None:
    """Simple training loop over the replay buffer using a DataLoader."""
    model.to(device)
    optimizer = create_optimizer(model, lr=lr)

    # build dataloader from replay buffer
    dataloader = make_replay_dataloader(
        replay,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device != "cpu"),
    )

    # we might have fewer batches than n_steps → just cycle
    data_iter: Optional[Iterable] = None

    def get_next_batch():
        nonlocal data_iter
        if data_iter is None:
            data_iter = iter(dataloader)
        try:
            return next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            return next(data_iter)

    for step in range(1, n_steps + 1):
        batch = get_next_batch()
        metrics = train_step_batch(model, batch, optimizer, device=device)

        if step % 10 == 0 or step == 1:
            print(
                f"[train] step={step} "
                f"loss={metrics['loss']:.4f} "
                f"policy={metrics['policy_loss']:.4f} "
                f"value={metrics['value_loss']:.4f}"
            )

