from __future__ import annotations
import torch
import torch.nn.functional as F


def policy_loss_fn(policy_logits: torch.Tensor, target_policy: torch.Tensor) -> torch.Tensor:
    log_probs = F.log_softmax(policy_logits, dim=1)
    loss = -(target_policy * log_probs).sum(dim=1).mean()
    return loss


def value_loss_fn(pred_value: torch.Tensor, target_value: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(pred_value, target_value)

