from __future__ import annotations
import chess
from typing import Dict, List

def count_checks(moves: List[chess.Move], start: chess.Board) -> int:
    board = start.copy()
    c = 0
    for mv in moves:
        board.push(mv)
        if board.is_check():
            c += 1
    return c

def advanced_pawn_score(board: chess.Board, white_perspective: bool = True) -> float:
    """Fraction of own pawns on 5th/6th/7th rank (white) or mirrored for black."""
    pawns = board.pieces(chess.PAWN, board.turn if white_perspective else not board.turn)
    if not pawns: return 0.0
    ranks = [chess.square_rank(sq) for sq in pawns]
    if white_perspective:
        adv = sum(1 for r in ranks if r >= 4)  # 5/6/7 (0-based)
    else:
        adv = sum(1 for r in ranks if r <= 3)  # mirrored
    return adv / max(len(ranks), 1)

def simple_aggression_metrics(moves: List[chess.Move], start: chess.Board) -> Dict[str, float]:
    return {
        "checks": float(count_checks(moves, start)),
        "advanced_pawns": float(advanced_pawn_score(start, white_perspective=True)),
    }

