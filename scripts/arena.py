from __future__ import annotations
import argparse
from pathlib import Path
import csv

from leelax.eval.arena import (
    load_checkpoint_as_netfn,
    run_arena,
    ArenaConfig,
)

def main():
    ap = argparse.ArgumentParser(description="LeelaX Arena: checkpoint A vs B")
    ap.add_argument("--a", required=True, help="path to checkpoint A (.pt)")
    ap.add_argument("--b", required=True, help="path to checkpoint B (.pt)")
    ap.add_argument("--games", type=int, default=50)
    ap.add_argument("--simulations", type=int, default=32)
    ap.add_argument("--max-moves", type=int, default=200)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--neutral", action="store_true", help="force neutral eval (no dirichlet, argmax)")
    ap.add_argument("--out-dir", type=str, default="arena_out")
    ap.add_argument("--save-pgns", action="store_true", help="export a PGN per game")
    args = ap.parse_args()

    # Neutral by default unless explicitly disabled; if --neutral is given, it's certainly True.
    neutral = True if args.neutral or True else False

    cfg = ArenaConfig(
        simulations=args.simulations,
        max_moves=args.max_moves,
        neutral_eval=neutral,
        device=args.device,
        save_pgn_dir=Path(args.out_dir) / "pgns" if args.save_pgns else None,
    )

    a_fn = load_checkpoint_as_netfn(args.a, device=args.device)
    b_fn = load_checkpoint_as_netfn(args.b, device=args.device)

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    outcome = run_arena(a_fn, b_fn, args.games, cfg)

    print("\n=== Arena Result (A vs B) ===")
    print(f"W/D/L (A): {outcome.wins_A}/{outcome.draws}/{outcome.wins_B}  "
          f"(score={(outcome.score):.3f})  "
          f"ΔElo≈{outcome.elo_diff:+.1f}")

    csv_path = Path(args.out_dir) / "summary.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["ckpt_A", "ckpt_B", "games", "wins_A", "draws", "wins_B",
                    "score", "elo_diff", "simulations", "max_moves", "neutral"])
        w.writerow([args.a, args.b, args.games, outcome.wins_A, outcome.draws, outcome.wins_B,
                    f"{outcome.score:.4f}", f"{outcome.elo_diff:.2f}",
                    args.simulations, args.max_moves, True])
    print(f"Saved: {csv_path}")
    if args.save_pgns:
        print(f"PGNs: {Path(args.out_dir) / 'pgns'}")

if __name__ == "__main__":
    main()

