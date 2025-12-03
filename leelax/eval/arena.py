from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Tuple, Optional, List

import numpy as np
import torch
import chess
import chess.pgn

from leelax.env.encode import state_to_tensor
from leelax.env.policy_index import index_to_move
from leelax.mcts.puct import PUCT
from leelax.net.model import LeelaXNet
from leelax.eval.elo import elo_from_wdl

NetFn = Callable[[torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]

@dataclass
class ArenaConfig:
    simulations: int = 32
    max_moves: int = 200
    neutral_eval: bool = True  # no dirichlet, temp=0 for selection
    device: str = "cpu"
    save_pgn_dir: Optional[Path] = None  # if set, saves PGN per game

def load_checkpoint_as_netfn(ckpt_path: str, device: str = "cpu") -> NetFn:
    device_t = torch.device(device)
    model = LeelaXNet().to(device_t)
    ckpt = torch.load(ckpt_path, map_location=device_t)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    def net_fn(x: torch.Tensor):
        with torch.no_grad():
            x = x.to(device_t)
            return model(x)
    return net_fn

def _engine_move(board: chess.Board, net_fn: NetFn, cfg: ArenaConfig) -> Tuple[int, np.ndarray]:
    # run MCTS and select move deterministically (argmax on visit policy)
    mcts = PUCT(net_fn, n_simulations=cfg.simulations, device=cfg.device)
    policy, _ = mcts.run(board, add_dirichlet=not cfg.neutral_eval)
    # deterministic pick
    action_idx = int(np.argmax(policy))
    return action_idx, policy

def _play_single_game(white_fn: NetFn, black_fn: NetFn, cfg: ArenaConfig, game_id: int) -> Tuple[int, Optional[str]]:
    """
    Returns: result from White perspective: +1 win, 0 draw, -1 loss; and PGN string (if cfg.save_pgn_dir)
    """
    board = chess.Board()
    game = chess.pgn.Game()
    game.headers["Event"] = "LeelaX Arena"
    game.headers["Site"] = "Local"
    game.headers["Round"] = str(game_id)
    node = game

    move_count = 0
    while not board.is_game_over() and move_count < cfg.max_moves:
        side = board.turn  # True=White
        net_fn = white_fn if side == chess.WHITE else black_fn
        a_idx, _ = _engine_move(board, net_fn, cfg)
        move = index_to_move(board, a_idx)
        if move not in board.legal_moves:
            # fall back (very rare if mask/policy consistent)
            move = list(board.legal_moves)[0]
        node = node.add_variation(move)
        board.push(move)
        move_count += 1

    result = 0
    if board.is_game_over():
        outcome = board.outcome()
        if outcome.winner is None:
            result = 0
            game.headers["Result"] = "1/2-1/2"
        elif outcome.winner is True:
            result = +1
            game.headers["Result"] = "1-0"
        else:
            result = -1
            game.headers["Result"] = "0-1"
    else:
        # max-moves reached => adjudicate as draw
        result = 0
        game.headers["Result"] = "1/2-1/2"

    pgn_str = None
    if cfg.save_pgn_dir is not None:
        cfg.save_pgn_dir.mkdir(parents=True, exist_ok=True)
        pgn_str = str(game)
        out_path = cfg.save_pgn_dir / f"game_{game_id:04d}.pgn"
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(pgn_str)

    return result, pgn_str

@dataclass
class ArenaOutcome:
    wins_A: int
    draws: int
    wins_B: int
    elo_diff: float
    score: float

def run_arena(a_net: NetFn, b_net: NetFn, n_games: int, cfg: ArenaConfig) -> ArenaOutcome:
    """
    Play n_games; even-numbered games A as White, odd-numbered games B as White (alternate colors).
    Returns W/D/L for A (across both colors), plus Elo estimate (A vs B).
    """
    wins_A = draws = wins_B = 0
    for g in range(1, n_games + 1):
        a_white = (g % 2 == 1)  # game1: A white
        white_fn = a_net if a_white else b_net
        black_fn = b_net if a_white else a_net

        res_white, _ = _play_single_game(white_fn, black_fn, cfg, g)
        # convert to A's perspective
        if a_white:
            if res_white == +1: wins_A += 1
            elif res_white == 0: draws += 1
            else: wins_B += 1
        else:
            # A was black => invert
            if res_white == +1: wins_B += 1
            elif res_white == 0: draws += 1
            else: wins_A += 1

    elo_res = elo_from_wdl(wins_A, draws, wins_B)
    return ArenaOutcome(wins_A, draws, wins_B, elo_res.elo_diff, elo_res.score)

