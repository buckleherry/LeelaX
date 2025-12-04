from __future__ import annotations

from typing import Optional

import torch
from torch.utils.data import DataLoader

from leelax.selfplay.replay_buffer import ReplayBuffer
from leelax.train.replay_dataset import make_replay_dataloader
from leelax.train.optim import create_optimizer, create_scheduler
from leelax.train.losses import policy_loss_fn, value_loss_fn


@torch.no_grad()
def _log_tb_step(tb_writer, global_step: int, loss: float, loss_p: float, loss_v: float, lr: float) -> None:
    if tb_writer is None:
        return
    tb_writer.add_scalar("train/loss", loss, global_step)
    tb_writer.add_scalar("train/policy_loss", loss_p, global_step)
    tb_writer.add_scalar("train/value_loss", loss_v, global_step)
    tb_writer.add_scalar("train/lr_group_0", lr, global_step)


def train_for_n_steps(
    model: torch.nn.Module,
    buffer: ReplayBuffer,
    steps: int,
    *,
    batch_size: int = 256,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    scheduler_name: str = "cosine",
    device: str = "cpu",
    tb_writer=None,
    global_step_start: int = 0,
    grad_clip_norm: Optional[float] = 1.0,
) -> int:
    """
    Single-phase supervised RL training over replay data.

    Args:
        model:         Policy/Value network (already moved to device).
        buffer:        Replay buffer with (state, policy, value) tuples.
        steps:         Max optimizer steps to run this phase.
        batch_size:    Minibatch size for dataloader.
        lr:            Base learning rate.
        weight_decay:  Weight decay for optimizer.
        scheduler_name:"cosine" | "onecycle" | "none" (as supported by create_scheduler).
        device:        "cpu" or "cuda".
        tb_writer:     Optional SummaryWriter for TensorBoard logging.
        global_step_start: Starting global step to continue logging curves.
        grad_clip_norm: If set, clip total grad norm to this value (recommended).

    Returns:
        global_step:   The updated global step counter after training.
    """
    assert steps > 0, "steps must be > 0"
    model.train()

    optimizer = create_optimizer(model, lr=lr, weight_decay=weight_decay)
    scheduler = create_scheduler(optimizer, name=scheduler_name, total_steps=steps)

    dl: DataLoader = make_replay_dataloader(buffer, batch_size=batch_size)

    global_step = global_step_start
    step_count = 0

    for batch in dl:
        # Safety: stop once we hit requested steps
        if step_count >= steps:
            break

        states, target_policy, target_value = batch
        states = states.to(device, non_blocking=True)
        target_policy = target_policy.to(device, non_blocking=True)
        target_value = target_value.to(device, non_blocking=True)

        policy_logits, value_pred = model(states)

        loss_p = policy_loss_fn(policy_logits, target_policy)
        loss_v = value_loss_fn(value_pred, target_value)
        loss = loss_p + loss_v

        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        if grad_clip_norm is not None and grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)

        optimizer.step()
        scheduler.step()

        step_count += 1
        global_step += 1

        # occasional console print (keeps STDOUT quiet in big runs)
        if step_count % 10 == 0 or step_count == 1 or step_count == steps:
            try:
                lr_now = optimizer.param_groups[0]["lr"]
            except Exception:
                lr_now = lr
            _log_tb_step(tb_writer, global_step, float(loss.item()), float(loss_p.item()), float(loss_v.item()), float(lr_now))

    return global_step

