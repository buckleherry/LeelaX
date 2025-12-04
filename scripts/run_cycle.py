#!/usr/bin/env python

from __future__ import annotations
import argparse
from pathlib import Path
import torch

from leelax.net.model import build_model
from leelax.selfplay.worker import SelfPlayWorker
from leelax.selfplay.replay_buffer import ReplayBuffer
from leelax.train.loop import train_for_n_steps


def flexible_load(sd):
    if "model_state_dict" in sd:
        return sd["model_state_dict"]
    if "state_dict" in sd:
        return sd["state_dict"]
    return sd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cycles", type=int, required=True)
    ap.add_argument("--games-per-cycle", type=int, required=True)
    ap.add_argument("--train-steps", type=int, required=True)

    ap.add_argument("--simulations", type=int, default=32)
    ap.add_argument("--max-moves", type=int, default=200)
    ap.add_argument("--device", type=str, default="cpu")

    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--scheduler", type=str, default="cosine")

    ap.add_argument("--model-size", type=str, default="128x6",
                    help="small / base / 128x6")
    ap.add_argument("--warm-start", type=str, default=None)

    ap.add_argument("--log-dir", type=str, default="runs/exp")
    ap.add_argument("--checkpoint-dir", type=str, default="checkpoints/exp")

    args = ap.parse_args()

    device = args.device

    Path(args.log_dir).mkdir(exist_ok=True, parents=True)
    Path(args.checkpoint-dir).mkdir(exist_ok=True, parents=True)

    # Build model
    model = build_model(args.model_size).to(device)
    if args.warm_start:
        raw = torch.load(args.warm_start, map_location=device)
        model.load_state_dict(flexible_load(raw), strict=False)

    # MCTS network_fn
    def net_fn(x):
        with torch.no_grad():
            x = x.to(device)
            p, v = model(x)
        return p, v

    # Replay buffer
    buffer = ReplayBuffer(capacity=100_000)

    # Self-play worker
    worker = SelfPlayWorker(
        network_fn=net_fn,
        n_simulations=args.simulations,
        temperature=1.0,
        max_moves=args.max_moves,
        device=device,
    )

    global_step = 0

    for c in range(1, args.cycles + 1):
        print(f"[cycle {c}] self-play...")

        for _ in range(args.games_per_cycle):
            samples, _ = worker.play_game(verbose=False)
            buffer.add_many(samples)

        print(f"[cycle {c}] buffer size = {len(buffer)}")
        print(f"[cycle {c}] training...")

        global_step = train_for_n_steps(
            model=model,
            buffer=buffer,
            steps=args.train_steps,
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
            scheduler_name=args.scheduler,
            device=device,
            tb_writer=None,
            global_step_start=global_step,
        )

        # checkpoint
        ckpt = {
            "model_state_dict": model.state_dict(),
            "cycle": c,
            "config": {
                "model_size": args.model_size,
                "simulations": args.simulations,
                "max_moves": args.max_moves,
            }
        }
        path = Path(args.checkpoint_dir) / f"model_cycle_{c:03d}.pt"
        torch.save(ckpt, path)
        print(f"[cycle {c}] saved {path}")


if __name__ == "__main__":
    main()

