from __future__ import annotations
from typing import Tuple

import torch
from torch.utils.data import Dataset, DataLoader

from leelax.selfplay.replay_buffer import ReplayBuffer


class ReplayDataset(Dataset):
    def __init__(self, replay_buffer: ReplayBuffer) -> None:
        self.replay_buffer = replay_buffer

    def __len__(self) -> int:
        return len(self.replay_buffer)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        state, policy, value = self.replay_buffer._storage[idx]
        return state.float(), torch.from_numpy(policy).float(), torch.tensor([value], dtype=torch.float32)


def make_replay_dataloader(
    replay_buffer: ReplayBuffer,
    batch_size: int = 64,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> DataLoader:
    ds = ReplayDataset(replay_buffer)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

