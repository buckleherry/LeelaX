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
    parser = argparse.ArgumentParser(description="LeelaX: self-play + train")
    parser.add_argument("--games", type=int, default=5, help="number of self-play games to generate")
    parser.add_argument("--train-steps", type=int, default=200, help="number of training steps")
    parser.add_argument("--simulations", type=int, default=32, help="MCTS simulations per move")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/leelax.pt")
    parser.add_argument("--log-dir", type=str, default="runs")
    args = parser.parse_args()

    device = args.device

    # model
    model = LeelaXNet().to(device)

    # network fn for MCTS
    def net_fn(x):
        with torch.no_grad():
            x = x.to(device)
            return model(x)

    # self-play
    worker = SelfPlayWorker(net_fn, n_simulations=args.simulations, temperature=1.0, device=device)
    buffer = ReplayBuffer(capacity=50_000)

    print(f"[selfplay] generating {args.games} games ...")
    for g in range(1, args.games + 1):
        samples = worker.play_game(verbose=False)
        buffer.add_many(samples)
        print(f"[selfplay] game {g}/{args.games} → {len(samples)} samples, buffer={len(buffer)}")

    # training
    writer = create_tb_writer(args.log_dir)
    print(f"[train] start training for {args.train_steps} steps ...")
    train_for_n_steps(
        model,
        buffer,
        n_steps=args.train_steps,
        device=device,
        tb_writer=writer,
    )

    # save checkpoint
    ckpt_path = Path(args.checkpoint)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
        },
        ckpt_path,
    )
    print(f"[checkpoint] saved to {ckpt_path}")


if __name__ == "__main__":
    main()

