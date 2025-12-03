from __future__ import annotations

from typing import Callable, Tuple, List
import numpy as np
import torch
import chess

from leelax.env.encode import state_to_tensor
from leelax.env.policy_index import index_to_move
from leelax.mcts.puct import PUCT
from leelax.selfplay.game_recorder import GameRecorder


class SelfPlayWorker:
    """
    Self-play generator with temperature schedule and optional early root Dirichlet noise.

    Args
    ----
    network_fn:  (x: Tensor[N,24,8,8]) -> (policy_logits[N, A], value[N,1])
    n_simulations: MCTS simulations per move
    max_moves:    hard cap on plies to avoid endless games
    device:       "cpu" | "cuda"
    tau_moves:    number of initial plies with temperature sampling (Tau = 1.0)
    dirichlet_until: add Dirichlet noise at the root only while move_count < dirichlet_until
    """
    def __init__(
        self,
        network_fn: Callable[[torch.Tensor], Tuple[torch.Tensor, torch.Tensor]],
        n_simulations: int = 64,
        max_moves: int = 512,
        device: str = "cpu",
        tau_moves: int = 8,
        dirichlet_until: int = 16,
    ) -> None:
        self.network_fn = network_fn
        self.n_simulations = n_simulations
        self.max_moves = max_moves
        self.device = device
        self.tau_moves = tau_moves
        self.dirichlet_until = dirichlet_until

    def play_game(self, verbose: bool = False):
        board = chess.Board()
        recorder = GameRecorder()
        move_count = 0

        while not board.is_game_over() and move_count < self.max_moves:
            # MCTS for current position
            mcts = PUCT(self.network_fn, n_simulations=self.n_simulations, device=self.device)
            add_dir = move_count < self.dirichlet_until
            policy, _ = mcts.run(board, add_dirichlet=add_dir)

            # Temperature schedule: sample early, then greedy
            if move_count < self.tau_moves:
                probs = policy.astype(np.float64)
                probs = probs / probs.sum()
                action_idx = int(np.random.choice(len(probs), p=probs))
            else:
                action_idx = int(np.argmax(policy))

            move = index_to_move(board, action_idx)
            if move not in board.legal_moves:
                # fallback (rare); choose uniformly among legal moves
                move = np.random.choice(list(board.legal_moves))

            # record training tuple
            state = state_to_tensor(board, canonical=True)
            recorder.add(state, policy, board.turn, move)

            # play
            san_str = board.san(move)
            uci_str = move.uci()
            board.push(move)
            move_count += 1

            if verbose:
                print(f"[{move_count}] {uci_str} ({san_str})")

        recorder.finalize(board)
        samples = recorder.export()
        moves: List[chess.Move] = recorder.moves
        return samples, moves

