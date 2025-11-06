from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
import chess.pgn

from leelax.selfplay.worker import SelfPlayWorker
from leelax.net.model import LeelaXNet
import torch


def play_and_collect(worker: SelfPlayWorker, n_games: int = 10):
    lengths: List[int] = []
    results: List[float] = []

    for _ in range(n_games):
        # we slightly modify worker to return also the board sequence if needed
        samples = worker.play_game(verbose=False)
        lengths.append(len(samples))
        # samples contain per-move z; we can just read z from the first
        _, _, z = samples[0]
        results.append(z)

    return lengths, results


def main():
    parser = argparse.ArgumentParser(description="LeelaX experiment helper")
    parser.add_argument("--games", type=int, default=20)
    parser.add_argument("--simulations", type=int, default=32)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    device = args.device
    model = LeelaXNet().to(device)

    def net_fn(x):
        with torch.no_grad():
            x = x.to(device)
            return model(x)

    worker = SelfPlayWorker(net_fn, n_simulations=args.simulations, temperature=1.0, device=device)

    lengths, results = play_and_collect(worker, n_games=args.games)
    print(f"[exp] games={len(lengths)}")
    print(f"[exp] avg length={np.mean(lengths):.2f} moves")
    print(f"[exp] results distribution: win={results.count(1.0)}, draw={results.count(0.0)}, loss={results.count(-1.0)}")


if __name__ == "__main__":
    main()

