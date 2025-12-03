#!/usr/bin/env python
"""
Aggregate arena results into a small Markdown report.

Usage:

  python scripts/arena_report.py --root arena

Assumes that each arena run has a structure like:

  arena/exp_B_c10_vs_c20/summary.csv

where summary.csv has columns: game,result,score_A,plies
(as written by scripts/arena.py).
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import List, Tuple


def estimate_elo_delta(score: float) -> float:
    import math

    eps = 1e-6
    score = max(min(score, 1.0 - eps), eps)
    return 400.0 * math.log10(score / (1.0 - score))


def read_summary(path: Path) -> Tuple[int, float, int, int, int]:
    """
    Returns:
      n_games, score_A, wins, draws, losses
    """
    results: List[Tuple[int, str, float, int]] = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            game = int(row["game"])
            result = row["result"]
            score_A = float(row["score_A"])
            plies = int(row["plies"])
            results.append((game, result, score_A, plies))

    n_games = len(results)
    wins = sum(1 for _, _, s, _ in results if s == 1.0)
    losses = sum(1 for _, _, s, _ in results if s == 0.0)
    draws = n_games - wins - losses
    score = sum(s for _, _, s, _ in results) / max(1, n_games)

    return n_games, score, wins, draws, losses


def make_report(root: Path) -> str:
    rows = []
    for summary in root.rglob("summary.csv"):
        rel = summary.parent.relative_to(root)
        tag = str(rel)  # e.g. "exp_B_c10_vs_c20"

        n_games, score, w, d, l = read_summary(summary)
        delta = estimate_elo_delta(score)
        rows.append((tag, n_games, score, w, d, l, delta))

    if not rows:
        return "# Arena Report\n\nNo summary.csv files found.\n"

    rows.sort(key=lambda x: x[0])

    lines = []
    lines.append("# Arena Report\n")
    lines.append("| Match | Games | Score(A) | W | D | L | ΔElo (A-B) |")
    lines.append("|-------|-------|----------|---|---|---|-----------|")
    for tag, n_games, score, w, d, l, delta in rows:
        lines.append(
            f"| `{tag}` | {n_games} | {score:.3f} | {w} | {d} | {l} | {delta:+.1f} |"
        )
    lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Aggregate arena summary.csv files")
    p.add_argument(
        "--root",
        type=str,
        default="arena",
        help="Root directory to search for summary.csv files",
    )
    p.add_argument(
        "--output",
        type=str,
        default="-",
        help="Output file for Markdown report (default: stdout). "
        "If not '-', writes to the given file path.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    report = make_report(root)

    if args.output == "-" or args.output == "":
        print(report)
    else:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(report, encoding="utf-8")
        print(f"Wrote report to {out_path}")


if __name__ == "__main__":
    main()

