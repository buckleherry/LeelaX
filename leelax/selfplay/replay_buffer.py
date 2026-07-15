from __future__ import annotations

from typing import List, Tuple, Optional
import random
import pathlib

import numpy as np
import torch


Sample = Tuple[torch.Tensor, np.ndarray, float]
# (state: [24,8,8], policy: [4864], value: float)


class ReplayBuffer:
    """Simple ring-buffer style replay buffer for self-play samples."""

    def __init__(self, capacity: int = 50_000) -> None:
        self.capacity = int(capacity)
        self._storage: List[Sample] = []
        self._next_idx: int = 0

    def __len__(self) -> int:
        return len(self._storage)

    def add(self, sample: Sample) -> None:
        if len(self._storage) < self.capacity:
            self._storage.append(sample)
        else:
            self._storage[self._next_idx] = sample
        self._next_idx = (self._next_idx + 1) % self.capacity

    def add_many(self, samples: List[Sample]) -> None:
        for s in samples:
            self.add(s)

    def sample(self, batch_size: int) -> Sample:
        """Return a batch as stacked tensors/arrays.

        Returns:
            states: torch.Tensor [B, 24, 8, 8]
            policies: torch.Tensor [B, 4864]
            values: torch.Tensor [B, 1]
        """
        assert len(self._storage) > 0, "buffer is empty"
        batch = random.sample(self._storage, k=min(batch_size, len(self._storage)))
        states = torch.stack([s[0] for s in batch], dim=0)  # [B,24,8,8]
        policies = torch.from_numpy(np.stack([s[1] for s in batch], axis=0))  # [B,4864]
        values = torch.tensor([[s[2]] for s in batch], dtype=torch.float32)  # [B,1]
        return states, policies, values

    # ------------------------------------------------------------------
    # optional: persistence
    # ------------------------------------------------------------------
    def save_npz(self, path: str | pathlib.Path) -> None:
        """Save buffer contents to .npz (simple, not memory-optimal)."""
        path = pathlib.Path(path)
        states = np.stack([s[0].numpy() for s in self._storage], axis=0)
        policies = np.stack([s[1] for s in self._storage], axis=0)
        values = np.array([s[2] for s in self._storage], dtype=np.float32)
        np.savez_compressed(path, states=states, policies=policies, values=values)

    @classmethod
    def load_npz(cls, path: str | pathlib.Path, capacity: Optional[int] = None) -> "ReplayBuffer":
        path = pathlib.Path(path)
        data = np.load(path, allow_pickle=False)
        states = data["states"]  # [N,24,8,8]
        policies = data["policies"]  # [N,4864]
        values = data["values"]  # [N]
        n = states.shape[0]

        buf = cls(capacity=capacity or n)
        for i in range(n):
            s = torch.from_numpy(states[i])
            p = policies[i]
            v = float(values[i])
            buf.add((s, p, v))
        return buf

