from __future__ import annotations

from typing import List, Tuple
import numpy as np
import torch
import chess


class GameRecorder:
    """Stores a self-play game as (state, policy, player, move) and
    can turn it into (state, policy, z) samples for training.
    """

    def __init__(self) -> None:
        self.states: List[torch.Tensor] = []
        self.policies: List[np.ndarray] = []
        self.players: List[chess.Color] = []
        self.moves: List[chess.Move] = []
        self.result: float = 0.0
        self._finalized: bool = False

    def add(
        self,
        state: torch.Tensor,
        policy: np.ndarray,
        player: chess.Color,
        move: chess.Move,
    ) -> None:
        self.states.append(state.detach().cpu())
        self.policies.append(policy.astype(np.float32))
        self.players.append(player)
        self.moves.append(move)

    def finalize(self, board: chess.Board) -> None:
        if board.is_game_over():
            res = board.result()
            if res == "1-0":
                self.result = 1.0
            elif res == "0-1":
                self.result = -1.0
            else:
                self.result = 0.0
        else:
            self.result = 0.0
        self._finalized = True

    def export(self):
        assert self._finalized, "call finalize() before export()"
        samples: List[Tuple[torch.Tensor, np.ndarray, float]] = []
        for st, pi, pl in zip(self.states, self.policies, self.players):
            z = self.result
            if pl == chess.BLACK:
                z = -z
            samples.append((st, pi, z))
        return samples

