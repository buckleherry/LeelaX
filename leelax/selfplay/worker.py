from __future__ import annotations

import chess
import torch
import numpy as np
from typing import Callable, Tuple

from leelax.env.encode import state_to_tensor
from leelax.env.policy_index import index_to_move
from leelax.mcts.puct import PUCT
from leelax.selfplay.game_recorder import GameRecorder


class SelfPlayWorker:
    def __init__(
        self,
        network_fn: Callable[[torch.Tensor], Tuple[torch.Tensor, torch.Tensor]],
        n_simulations: int = 64,
        temperature: float = 1.0,
        max_moves: int = 512,
        device: str = "cpu",
    ) -> None:
        self.network_fn = network_fn
        self.n_simulations = n_simulations
        self.temperature = temperature
        self.max_moves = max_moves
        self.device = device

    def play_game(self, verbose: bool = False):
        board = chess.Board()
        recorder = GameRecorder()
        move_count = 0

        while not board.is_game_over() and move_count < self.max_moves:
            state = state_to_tensor(board, canonical=True)

            mcts = PUCT(self.network_fn, n_simulations=self.n_simulations, device=self.device)
            policy, action_idx = mcts.run(board)

            # temperature sampling
            if self.temperature and self.temperature > 0:
                probs = np.power(policy, 1.0 / self.temperature)
                probs = probs / probs.sum()
                action_idx = int(np.random.choice(len(probs), p=probs))
            else:
                action_idx = int(np.argmax(policy))

            move = index_to_move(board, action_idx)
            if move not in board.legal_moves:
                move = np.random.choice(list(board.legal_moves))

            recorder.add(state, policy, board.turn, move)

            san_str = board.san(move)
            uci_str = move.uci()
            board.push(move)
            move_count += 1

            if verbose:
                print(f"[{move_count}] {uci_str} ({san_str})")

        recorder.finalize(board)
        samples = recorder.export()
        moves = recorder.moves
        return samples, moves

