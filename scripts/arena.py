#!/usr/bin/env python
"""
Arena matches between two checkpoints.

Example:

  python scripts/arena.py \
    --a checkpoints/exp_B_sharper_s32/model_cycle_010.pt \
    --b checkpoints/exp_B_sharper_s32/model_cycle_020.pt \
    --games 80 \
    --simulations 64 \
    --max-moves 220 \
    --device cpu \
    --neutral \
    --save-pgns \
    --out-dir arena/exp_B_c10_vs_c20
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import chess
import chess.pgn

from leelax.net.model import LeelaXNet
from leelax.mcts.puct import PUCT
from leelax.env.policy_index import legal_policy_mask, index_to_move


def load_net(checkpoint_path: str, device: str) -> LeelaXNet:
    """Load a LeelaXNet from checkpoint.

    Handles multiple formats:
      - {"model_state_dict": ..., ...}
      - {"state_dict": ..., ...}
      - raw state dict (ordered mapping of tensors)
    """
    ckpt = torch.load(checkpoint_path, map_location=device)

    if isinstance(ckpt, dict):
        if "model_state_dict" in ckpt:
            state_dict = ckpt["model_state_dict"]
        elif "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]
        else:
            # assume whole dict is already a state_dict
            state_dict = ckpt
    else:
        # very old style: checkpoint is directly a state dict
        state_dict = ckpt

    model = LeelaXNet()
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def make_net_fn(model: LeelaXNet, device: str):
    def net_fn(x: torch.Tensor):
        with torch.no_grad():
            x = x.to(device)
            policy_logits, value = model(x)
        return policy_logits, value

    return net_fn


def engine_move(
    board: chess.Board,
    net_fn,
    simulations: int,
    device: str,
    neutral: bool,
) -> Tuple[chess.Move, np.ndarray, int]:
    """
    One engine move using MCTS.

    - neutral=True: deterministic (no Dirichlet, argmax over policy)
    - neutral=False: explorative (Dirichlet + sampling over legal moves)
    """
    mcts = PUCT(net_fn, n_simulations=simulations, device=device)
    # add_dirichlet only if NOT neutral (explorative mode)
    policy, greedy_idx = mcts.run(board, add_dirichlet=not neutral)

    # policy is numpy array over all move indices
    policy = policy.astype(np.float64)

    if neutral:
        a_idx = int(greedy_idx)
    else:
        # sample only over legal moves
        mask = legal_policy_mask(board).astype(np.float64)
        probs = policy * mask
        s = probs.sum()
        if s <= 0.0:
            # fallback: greedy
            a_idx = int(greedy_idx)
        else:
            probs /= s
            a_idx = int(np.random.choice(len(probs), p=probs))

    move = index_to_move(board, a_idx)
    return move, policy, a_idx


def play_single_game(
    net_a_fn,
    net_b_fn,
    simulations: int,
    max_moves: int,
    device: str,
    neutral: bool,
    game_index: int,
) -> Tuple[str, chess.pgn.Game]:
    """
    Play one game between A and B.

    We alternate colors:
      - even game_index: A as White, B as Black
      - odd game_index : B as White, A as Black

    Returns:
      result_str (e.g. "1-0", "0-1", "1/2-1/2"),
      pgn_game (chess.pgn.Game)
    """
    board = chess.Board()

    game = chess.pgn.Game()
    game.headers["Event"] = "LeelaX Arena"
    game.headers["Site"] = "Local"
    game.headers["Round"] = str(game_index + 1)
    game.headers["Date"] = "????.??.??"

    a_is_white = (game_index % 2 == 0)
    if a_is_white:
        game.headers["White"] = "A"
        game.headers["Black"] = "B"
    else:
        game.headers["White"] = "B"
        game.headers["Black"] = "A"

    node = game

    ply = 0
    while not board.is_game_over(claim_draw=True) and ply < max_moves:
        if board.turn == chess.WHITE:
            net_fn = net_a_fn if a_is_white else net_b_fn
        else:
            net_fn = net_b_fn if a_is_white else net_a_fn

        move, _, _ = engine_move(
            board=board,
            net_fn=net_fn,
            simulations=simulations,
            device=device,
            neutral=neutral,
        )

        if move not in board.legal_moves:
            # safety fallback: pick any legal move deterministically
            move = next(iter(board.legal_moves))

        board.push(move)
        node = node.add_variation(move)
        ply += 1

    if board.is_game_over(claim_draw=True):
        result = board.result(claim_draw=True)
    else:
        # max-moves cutoff -> treat as draw
        result = "1/2-1/2"

    game.headers["Result"] = result
    return result, game


def estimate_elo_delta(score: float) -> float:
    """
    Approximate Elo difference from score (A's score vs B).
    score in [0,1]. For score ~0.5 -> delta ~0
    """
    import math

    eps = 1e-6
    score = max(min(score, 1.0 - eps), eps)
    # inverse of logistic: score = 1 / (1 + 10^(-d/400))
    d = 400.0 * math.log10(score / (1.0 - score))
    return d


def run_arena(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir)
    pgn_dir = out_dir / "pgns"
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.save_pgns:
        pgn_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading nets...")
    device = args.device
    net_a = load_net(args.a, device)
    net_b = load_net(args.b, device)
    net_a_fn = make_net_fn(net_a, device)
    net_b_fn = make_net_fn(net_b, device)

    np.random.seed(args.seed)

    results = []  # list of (game_idx, result, score_A, plies)

    for g in range(args.games):
        print(f"[game {g+1}/{args.games}] playing ...")
        result_str, game = play_single_game(
            net_a_fn=net_a_fn,
            net_b_fn=net_b_fn,
            simulations=args.simulations,
            max_moves=args.max_moves,
            device=device,
            neutral=args.neutral,
            game_index=g,
        )

        # from White's perspective:
        if result_str == "1-0":
            white_score = 1.0
        elif result_str == "0-1":
            white_score = 0.0
        else:
            white_score = 0.5

        a_is_white = (g % 2 == 0)
        if a_is_white:
            score_a = white_score
        else:
            score_a = 1.0 - white_score if result_str in ("1-0", "0-1") else 0.5

        plies = sum(1 for _ in game.mainline_moves())

        results.append(
            (
                g + 1,
                result_str,
                score_a,
                plies,
            )
        )

        if args.save_pgns:
            pgn_path = pgn_dir / f"game_{g+1:04d}.pgn"
            with open(pgn_path, "w", encoding="utf-8") as f:
                print(game, file=f)

    # aggregate
    n_games = len(results)
    w = sum(1 for _, _, s, _ in results if s == 1.0)
    l = sum(1 for _, _, s, _ in results if s == 0.0)
    d = n_games - w - l
    score = sum(s for _, _, s, _ in results) / max(1, n_games)
    delta_elo = estimate_elo_delta(score)

    print("=== Arena Result (A vs B) ===")
    print(f"W/D/L (A): {w}/{d}/{l}  (score={score:.3f})  ΔElo≈{delta_elo:+.1f}")
    print(f"Saved: {out_dir/'summary.csv'}")
    if args.save_pgns:
        print(f"PGNs: {pgn_dir}")

    # write summary.csv
    summary_path = out_dir / "summary.csv"
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["game", "result", "score_A", "plies"])
        writer.writerows(results)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Arena matches between two checkpoints")
    p.add_argument("--a", required=True, help="Path to checkpoint A (reference)")
    p.add_argument("--b", required=True, help="Path to checkpoint B (candidate)")
    p.add_argument("--games", type=int, default=80, help="Number of games")
    p.add_argument("--simulations", type=int, default=64, help="MCTS simulations per move")
    p.add_argument("--max-moves", type=int, default=220, help="Max plies per game (half-moves)")
    p.add_argument("--device", type=str, default="cpu", help="Device: cpu or cuda")
    p.add_argument(
        "--neutral",
        action="store_true",
        help="Neutral deterministic mode (no Dirichlet, argmax policy). "
        "If omitted, uses explorative sampling with Dirichlet.",
    )
    p.add_argument(
        "--save-pgns",
        action="store_true",
        help="Save PGN files for each game",
    )
    p.add_argument(
        "--out-dir",
        type=str,
        default="arena/out",
        help="Output directory for summary + PGNs",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=1234,
        help="Random seed for explorative mode",
    )
    return p.parse_args()


if __name__ == "__main__":
    run_arena(parse_args())

