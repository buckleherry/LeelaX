from __future__ import annotations

import torch
import torch.nn.functional as F


def policy_loss_fn(
    policy_logits: torch.Tensor,
    target_policy: torch.Tensor,
    label_smoothing: float = 0.0,
) -> torch.Tensor:
    """
    Cross-entropy between predicted logits and target policy distribution.

    Args:
        policy_logits: [B, 4864] raw logits
        target_policy: [B, 4864] probabilities (from MCTS visit counts)
        label_smoothing: small epsilon to smooth the target distribution.
    """
    if label_smoothing > 0:
        K = target_policy.size(-1)
        smooth = label_smoothing / K
        target_policy = (1 - label_smoothing) * target_policy + smooth

    log_probs = F.log_softmax(policy_logits, dim=-1)
    loss = -(target_policy * log_probs).sum(dim=1)
    return loss.mean()


def value_loss_fn(
    value_pred: torch.Tensor,
    target_value: torch.Tensor,
    huber_delta: float = 1.0,
) -> torch.Tensor:
    """
    Value regression loss. Huber works better than raw MSE early in training.

    Args:
        value_pred: [B,1] predicted value
        target_value: [B,1] actual game result (-1,0,1)
    """
    return F.smooth_l1_loss(value_pred, target_value, beta=huber_delta)

