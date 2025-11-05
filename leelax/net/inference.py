from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F


def masked_softmax(
    logits: torch.Tensor,
    mask: torch.Tensor,
    dim: int = -1,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Softmax over valid actions only.

    Args:
        logits: [..., N]
        mask:   same shape, 1 for legal, 0 for illegal
    Returns:
        probs:  same shape, sums to 1 over legal entries
    """
    # ensure float
    mask = mask.to(dtype=logits.dtype)
    # set illegal to very negative
    neg_inf = torch.finfo(logits.dtype).min
    masked_logits = torch.where(mask > 0, logits, torch.full_like(logits, neg_inf))
    # softmax
    probs = F.softmax(masked_logits, dim=dim)
    # in case mask was all zeros, softmax will be nan -> fix by renorm
    probs = probs * mask
    denom = probs.sum(dim=dim, keepdim=True)
    probs = probs / (denom + eps)
    return probs


def select_action(
    logits: torch.Tensor,
    mask: torch.Tensor,
    temperature: float = 1.0,
    greedy: bool = False,
) -> torch.Tensor:
    """Sample or pick an action index given logits and a legality mask.

    Args:
        logits: [B, N]
        mask:   [B, N] (0/1)
        temperature: >0, scales logits before softmax
        greedy: if True, take argmax over legal actions

    Returns:
        indices: [B] int64
    """
    if logits.dim() != 2:
        raise ValueError("logits must be [B, N]")
    if mask.dim() != 2:
        raise ValueError("mask must be [B, N]")

    if greedy:
        # set illegal to -inf
        neg_inf = torch.finfo(logits.dtype).min
        masked = torch.where(mask > 0, logits, torch.full_like(logits, neg_inf))
        return masked.argmax(dim=1)

    # temperature scaling
    if temperature != 1.0:
        logits = logits / temperature

    probs = masked_softmax(logits, mask, dim=1)  # [B, N]

    # sample per batch entry
    B, N = probs.shape
    # torch.multinomial expects probs >= 0 and rows sum to 1
    idx = torch.multinomial(probs, num_samples=1)  # [B, 1]
    return idx.squeeze(1)
