import numpy as np
import torch

from leelax.selfplay.replay_buffer import ReplayBuffer
from leelax.train.replay_dataset import ReplayDataset, make_replay_dataloader


def test_replay_dataset_and_dataloader():
    buf = ReplayBuffer(capacity=20)

    # add some fake samples
    for _ in range(7):
        state = torch.zeros(24, 8, 8)
        policy = np.zeros(4864, dtype=np.float32)
        policy[0] = 1.0
        value = 0.5
        buf.add((state, policy, value))

    ds = ReplayDataset(buf)
    assert len(ds) == 7

    s0, p0, v0 = ds[0]
    assert s0.shape == (24, 8, 8)
    assert p0.shape == (4864,)
    assert v0.shape == (1,)

    loader = make_replay_dataloader(buf, batch_size=4, shuffle=True, num_workers=0)
    batch = next(iter(loader))
    states, policies, values = batch
    assert states.shape[0] <= 4
    assert states.shape[1:] == (24, 8, 8)
    assert policies.shape[1] == 4864
    assert values.shape[1] == 1

