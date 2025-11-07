from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import numpy as np
import torch
import chess
import chess.pgn

from leelax.net.model import LeelaXNet
from leelax.selfplay.worker import SelfPlayWorker


def game_to_pgn(moves: List[chess.Move]) -> chess.pgn.Game:
    game = chess.pgn.Game()
    node = game
    board = chess.Board()
    for mv in moves:
        node = node.add_variation(mv)
        board.push(mv)
    return game


def game_to_fens(moves: List[chess.Move]) -> List[str]:
    board = chess.Board()
    fens = [board.fen()]
    for mv in moves:
        board.push(mv)
        fens.append(board.fen())
    return fens


def main():
    parser = argparse.ArgumentParser(description="LeelaX experiment helper (short games)")
    parser.add_argument("--games", type=int, default=50)
    parser.add_argument("--simulations", type=int, default=32)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--save-dir", type=str, default="experiments/short_games")
    parser.add_argument("--max-moves", type=int, default=40, help="only store games with <= this many moves")
    args = parser.parse_args()

    device = args.device
    model = LeelaXNet().to(device)

    def net_fn(x):
        with torch.no_grad():
            x = x.to(device)
            return model(x)

    worker = SelfPlayWorker(net_fn, n_simulations=args.simulations, temperature=1.0, device=device)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    lengths: List[int] = []
    results: List[float] = []
    short_saved = 0

    for game_idx in range(1, args.games + 1):
        samples, moves = worker.play_game(verbose=False)
        lengths.append(len(samples))
        _, _, z = samples[0]
        results.append(z)

        if len(moves) <= args.max_moves:
            game = game_to_pgn(moves)
            pgn_path = save_dir / f"game_{game_idx:04d}.pgn"
            with open(pgn_path, "w") as f:
                print(game, file=f)

            fens = game_to_fens(moves)
            fen_path = save_dir / f"game_{game_idx:04d}.fens"
            with open(fen_path, "w") as f:
                for fen in fens:
                    f.write(fen + "\n")

            short_saved += 1
            print(f"[exp] saved short game {game_idx} with {len(moves)} moves")

    lengths_arr = np.array(lengths)
    print(f"[exp] total games: {len(lengths)}")
    print(f"[exp] avg length : {lengths_arr.mean():.2f}")
    print(f"[exp] short (<= {args.max_moves}) saved: {short_saved}")
    print(
        f"[exp] results: win={results.count(1.0)}, "
        f"draw={results.count(0.0)}, loss={results.count(-1.0)}"
    )


if __name__ == "__main__":
    main()

