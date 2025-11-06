from __future__ import annotations

from typing import Tuple

import torch
from torch.utils.data import Dataset, DataLoader

from leelax.selfplay.replay_buffer import ReplayBuffer


class ReplayDataset(Dataset):
    """Torch Dataset wrapper around the in-memory ReplayBuffer.

    This makes it possible to use standard PyTorch DataLoader features
    (shuffling, multi-process loading, pin_memory, ...).
    """

    def __init__(self, replay_buffer: ReplayBuffer) -> None:
        self.replay_buffer = replay_buffer

    def __len__(self) -> int:
        # number of stored samples, not capacity
        return len(self.replay_buffer)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # we reach into the buffer's storage; that's fine for single-process
        state, policy, value = self.replay_buffer._storage[idx]
        # ensure correct dtypes
        state = state.float()  # [24,8,8]
        policy = torch.from_numpy(policy).float()  # [4864]
        value = torch.tensor([value], dtype=torch.float32)  # [1]
        return state, policy, value


def make_replay_dataloader(
    replay_buffer: ReplayBuffer,
    batch_size: int = 64,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> DataLoader:
    """Convenience ctor for a DataLoader on top of the replay buffer."""
    dataset = ReplayDataset(replay_buffer)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
    return loader

