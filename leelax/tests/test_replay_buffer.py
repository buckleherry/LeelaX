import numpy as np
import torch

from leelax.selfplay.replay_buffer import ReplayBuffer


def test_replay_buffer_add_and_sample(tmp_path):
    buf = ReplayBuffer(capacity=10)

    # add 5 fake samples
    for _ in range(5):
        state = torch.zeros(24, 8, 8)
        policy = np.zeros(4864, dtype=np.float32)
        policy[0] = 1.0
        value = 0.0
        buf.add((state, policy, value))

    assert len(buf) == 5

    states, policies, values = buf.sample(batch_size=3)
    assert states.shape == (3, 24, 8, 8)
    assert policies.shape == (3, 4864)
    assert values.shape == (3, 1)

    # test save/load
    out_file = tmp_path / "buffer.npz"
    buf.save_npz(out_file)

    buf2 = ReplayBuffer.load_npz(out_file)
    assert len(buf2) == 5
    s2, p2, v2 = buf2.sample(2)
    assert s2.shape == (2, 24, 8, 8)

