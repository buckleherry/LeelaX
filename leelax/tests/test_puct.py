import chess
import numpy as np
import torch

from leelax.mcts.puct import PUCT
from leelax.env.policy_index import index_to_move


def test_puct_runs_on_startpos():
    board = chess.Board()

    def dummy_net(x: torch.Tensor):
        # x: [1, 24, 8, 8]
        b = x.size(0)
        # our action space is 4864 now
        return torch.zeros(b, 4864), torch.zeros(b, 1)

    mcts = PUCT(dummy_net, n_simulations=8, device="cpu")
    policy, action_idx = mcts.run(board, add_dirichlet=False)

    assert isinstance(policy, np.ndarray)
    assert policy.shape == (4864,)
    assert abs(policy.sum() - 1.0) < 1e-5

    mv = index_to_move(board, action_idx)
    assert mv in board.legal_moves

