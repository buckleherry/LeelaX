from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple
import io
import numpy as np
import torch
import chess
import chess.pgn

from leelax.net.model import LeelaXNet
from leelax.mcts.puct import PUCT
from leelax.env.policy_index import index_to_move

def load_checkpoint_as_netfn(ckpt_path: str, device: str = "cpu"):
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

def engine_move(board: chess.Board, net_fn, simulations: int, device: str, neutral: bool) -> Tuple[int, np.ndarray]:
    mcts = PUCT(net_fn, n_simulations=simulations, device=device)
    policy, _ = mcts.run(board, add_dirichlet=not neutral)

    if neutral:
        # deterministic engine line
        a_idx = int(np.argmax(policy))
    else:
        # explorative mode: Policy-Sampling
        probs = policy.astype(float)
        probs = probs / probs.sum()
        a_idx = int(np.random.choice(len(probs), p=probs))
    return a_idx, policy

def play_one_game(net_fn, simulations: int, max_moves: int, device: str, neutral: bool) -> Tuple[str, str, int]:
    board = chess.Board()
    game = chess.pgn.Game()
    node = game
    plies = 0

    while not board.is_game_over() and plies < max_moves:
        a_idx, _ = engine_move(board, net_fn, simulations, device, neutral)
        move = index_to_move(board, a_idx)
        if move not in board.legal_moves:
            move = list(board.legal_moves)[0]
        node = node.add_variation(move)
        board.push(move)
        plies += 1

    if board.is_game_over():
        outcome = board.outcome()
        if outcome.winner is None:
            game.headers["Result"] = "1/2-1/2"
        elif outcome.winner:
            game.headers["Result"] = "1-0"
        else:
            game.headers["Result"] = "0-1"
    else:
        game.headers["Result"] = "1/2-1/2"  # adjudicate draw

    return str(game), board.fen(), plies

def main():
    ap = argparse.ArgumentParser(description="Generate PGNs from a checkpoint for inspection")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--games", type=int, default=100)
    ap.add_argument("--simulations", type=int, default=64)
    ap.add_argument("--max-moves", type=int, default=200)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--neutral", action="store_true", help="no Dirichlet; argmax selection")
    ap.add_argument("--save-dir", type=str, default="experiments/view")
    ap.add_argument("--write-fens", action="store_true")
    ap.add_argument("--short-max-plies", type=int, default=None, help="if set, keep only games with plies <= this")
    ap.add_argument("--decisive-only", action="store_true", help="keep only non-draw games")
    args = ap.parse_args()

    out_dir = Path(args.save_dir)
    pgn_dir = out_dir / "pgns"
    fen_dir = out_dir / "fens"
    pgn_dir.mkdir(parents=True, exist_ok=True)
    if args.write_fens:
        fen_dir.mkdir(parents=True, exist_ok=True)

    net_fn = load_checkpoint_as_netfn(args.checkpoint, device=args.device)

    kept = 0
    for i in range(1, args.games + 1):
        pgn_str, fen_last, plies = play_one_game(
            net_fn,
            simulations=args.simulations,
            max_moves=args.max_moves,
            device=args.device,
            neutral=True if args.neutral else False,
        )

        # simple post-filter
        keep = True
        if args.short_max_plies is not None:
            # parse result quickly from PGN header line
            res_line = pgn_str.splitlines()[0] if pgn_str else ""
            # Count plies already known; no need to re-parse PGN
            if plies > args.short_max_plies:
                keep = False
        if keep and args.decisive_only:
            if 'Result "1/2-1/2"' in pgn_str:
                keep = False

        if not keep:
            continue

        kept += 1
        with open(pgn_dir / f"game_{i:04d}.pgn", "w", encoding="utf-8") as f:
            f.write(pgn_str)

        if args.write_fens:
            # dump full sequence of FENs (one per ply)
            # Re-play to collect FENs:
            board = chess.Board()
            fen_path = fen_dir / f"game_{i:04d}.fens"
            with open(fen_path, "w", encoding="utf-8") as ffen:
                ffen.write(board.fen() + "\n")
                game = chess.pgn.read_game(io.StringIO(pgn_str))
                board = game.board()
                for mv in game.mainline_moves():
                    board.push(mv)
                    ffen.write(board.fen() + "\n")

    print(f"Saved {kept} PGNs to {pgn_dir}")
    if args.write_fens:
        print(f"FEN files in {fen_dir}")

if __name__ == "__main__":
    main()

