from __future__ import annotations

import argparse
import json
from pathlib import Path

import chess
import numpy as np
import torch

from leelax.env.encode import state_to_tensor
from leelax.env.policy_index import legal_policy_mask, move_to_index


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect chess state encodings and policy masks."
    )
    parser.add_argument(
        "--fen",
        type=str,
        default=None,
        help="FEN string to encode. If omitted, the standard start position is used.",
    )
    parser.add_argument(
        "--dump",
        type=str,
        default=None,
        help="Optional path to .npz file to dump encoding + mask.",
    )
    parser.add_argument(
        "--show-moves",
        action="store_true",
        help="Print all legal moves with their policy indices.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.fen is None or args.fen == "startpos":
        board = chess.Board()
    else:
        board = chess.Board(args.fen)

    # encode board
    state_tensor: torch.Tensor = state_to_tensor(board)  # (24, 8, 8)
    mask: np.ndarray = legal_policy_mask(board)  # (4672,)

    print("=== LeelaX Env Inspect ===")
    print(f"FEN: {board.fen()}")
    print(f"Turn: {'white' if board.turn == chess.WHITE else 'black'}")
    print(f"Encoding shape: {tuple(state_tensor.shape)} (should be (24, 8, 8))")
    print(f"Policy mask shape: {mask.shape} (should be (4672,))")
    print(f"Number of legal moves: {int(mask.sum())}")

    # optional: show moves
    if args.show_moves:
        print("\nLegal moves:")
        moves_info = []
        for mv in board.legal_moves:
            try:
                idx = move_to_index(mv)
            except Exception:
                idx = None
            moves_info.append(
                {
                    "uci": mv.uci(),
                    "san": board.san(mv),
                    "index": idx,
                }
            )
        # pretty print
        for mi in moves_info:
            print(f" - {mi['uci']:>6}  {mi['san']:<10}  idx={mi['index']}")

    # optional dump
    if args.dump is not None:
        out_path = Path(args.dump)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            out_path,
            state=state_tensor.detach().cpu().numpy(),
            mask=mask,
            fen=board.fen(),
        )
        print(f"\nDumped encoding to {out_path}")

    # also print a small JSON summary (nice for scripts)
    summary = {
        "fen": board.fen(),
        "turn": "white" if board.turn == chess.WHITE else "black",
        "state_shape": tuple(state_tensor.shape),
        "num_legal_moves": int(mask.sum()),
    }
    print("\nJSON summary:")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

