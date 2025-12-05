#!/usr/bin/env python

from __future__ import annotations
import argparse
from pathlib import Path
import torch

from leelax.net.model import build_model
from leelax.selfplay.worker import SelfPlayWorker
from leelax.selfplay.replay_buffer import ReplayBuffer
from leelax.train.loop import train_for_n_steps
from leelax.train.logger import create_tb_writer 

def _flex_sd(sd):
    if isinstance(sd, dict):
        if "model_state_dict" in sd:
            return sd["model_state_dict"]
        if "state_dict" in sd:
            return sd["state_dict"]
    return sd

def main():
    ap = argparse.ArgumentParser(description="Self-play + train cycles")
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

    ap.add_argument("--model-size", type=str, default="128x6", help="small/base/128x6")
    ap.add_argument("--warm-start", type=str, default=None)

    ap.add_argument("--log-dir", type=str, default="runs/exp")
    ap.add_argument("--checkpoint-dir", type=str, default="checkpoints/exp")

    args = ap.parse_args()
    device = args.device

    Path(args.log_dir).mkdir(parents=True, exist_ok=True)
    Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    # Build model (+ optional warm start)
    model = build_model(args.model_size).to(device)
    if args.warm_start:
        raw = torch.load(args.warm_start, map_location=device)
        model.load_state_dict(_flex_sd(raw), strict=False)

    # network function for MCTS
    def net_fn(x: torch.Tensor):
        with torch.no_grad():
            x = x.to(device)
            return model(x)

    buffer = ReplayBuffer(capacity=100_000)
    worker = SelfPlayWorker(
        network_fn=net_fn,
        n_simulations=args.simulations,
        max_moves=args.max_moves,
        device=device,
    )

    tb = None
    try:
        tb = create_tb_writer(args.log_dir)
    except Exception:
        tb = None

    global_step = 0
    for c in range(1, args.cycles + 1):
        print(f"[cycle {c}] self-play ...")
        for _ in range(args.games_per_cycle):
            samples = worker.play_game(verbose=False)[0] 
            buffer.add_many(samples)

        print(f"[cycle {c}] buffer size = {len(buffer)}")
        print(f"[cycle {c}] training ...")

        global_step = train_for_n_steps(
            model=model,
            buffer=buffer,
            steps=args.train_steps,
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
            scheduler_name=args.scheduler,
            device=device,
            tb_writer=tb,
            global_step_start=global_step,
        )

        ckpt_path = Path(args.checkpoint_dir) / f"model_cycle_{c:03d}.pt"
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "cycle": c,
                "config": {
                    "model_size": args.model_size,
                    "simulations": args.simulations,
                    "max_moves": args.max_moves,
                    "batch_size": args.batch_size,
                    "lr": args.lr,
                    "weight_decay": args.weight_decay,
                    "scheduler": args.scheduler,
                },
            },
            ckpt_path,
        )
        print(f"[cycle {c}] checkpoint saved to {ckpt_path}")

    if tb:
        tb.close()
    print("done.")

if __name__ == "__main__":
    main()

