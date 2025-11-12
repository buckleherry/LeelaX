from __future__ import annotations

import argparse
from pathlib import Path

import torch

from leelax.net.model import LeelaXNet
from leelax.selfplay.worker import SelfPlayWorker
from leelax.selfplay.replay_buffer import ReplayBuffer
from leelax.train.loop import train_for_n_steps
from leelax.train.logger import create_tb_writer


def main():
    parser = argparse.ArgumentParser(description="LeelaX training cycle")
    parser.add_argument("--cycles", type=int, default=3, help="number of selfplay+train cycles")
    parser.add_argument("--games-per-cycle", type=int, default=20, help="self-play games per cycle")
    parser.add_argument("--train-steps", type=int, default=200, help="training steps per cycle")
    parser.add_argument("--simulations", type=int, default=32, help="MCTS simulations per move")
    parser.add_argument("--max-moves", type=int, default=200, help="max-moves per game")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--log-dir", type=str, default="runs")
    args = parser.parse_args()

    device = args.device
    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # 1) model
    model = LeelaXNet().to(device)

    def net_fn(x):
        with torch.no_grad():
            x = x.to(device)
            return model(x)

    # 2) self-play worker + replay buffer
    worker = SelfPlayWorker(
        net_fn,
        n_simulations=args.simulations,
        temperature=1.0,
        max_moves=args.max_moves,
        device=device,
    )

    buffer = ReplayBuffer(capacity=100_000)
    writer = create_tb_writer(args.log_dir)

    for cycle in range(1, args.cycles + 1):
        print(f"[cycle {cycle}] self-play ...")
        for g in range(1, args.games_per_cycle + 1):
            samples, _moves = worker.play_game(verbose=False)
            buffer.add_many(samples)
        print(f"[cycle {cycle}] buffer size = {len(buffer)}")

        print(f"[cycle {cycle}] training ...")
        train_for_n_steps(
            model,
            buffer,
            n_steps=args.train_steps,
            device=device,
            tb_writer=writer,
        )

        ckpt_path = ckpt_dir / f"model_cycle_{cycle:03d}.pt"
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "cycle": cycle,
                "config": {
                    "games_per_cycle": args.games_per_cycle,
                    "train_steps": args.train_steps,
                    "simulations": args.simulations,
                },
            },
            ckpt_path,
        )

        print(f"[cycle {cycle}] checkpoint saved to {ckpt_path}")


if __name__ == "__main__":
    main()

