# leelax/tests/test_puct.py

import chess
import numpy as np
import torch

from leelax.mcts.puct import PUCT
from leelax.env.policy_index import index_to_move


def test_puct_runs_on_startpos():
    board = chess.Board()

    # simple dummy net: uniform(-ish) policy, zero value
    def dummy_net(x: torch.Tensor):
        # x: [1, 24, 8, 8]
        batch = x.size(0)
        policy_logits = torch.zeros(batch, 4672)  # all equal
        value = torch.zeros(batch, 1)             # 0.0 position
        return policy_logits, value

    mcts = PUCT(dummy_net, n_simulations=16)  # small number for test speed
    policy, action_idx = mcts.run(board, add_dirichlet=False)

    # shape checks
    assert isinstance(policy, np.ndarray)
    assert policy.shape == (4672,)
    # must sum to 1 (or very close)
    assert abs(policy.sum() - 1.0) < 1e-5

    # action must be legal
    mv = index_to_move(action_idx)
    assert mv in board.legal_moves

