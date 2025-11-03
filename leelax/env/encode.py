from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import chess

# Channel layout (24):
#  0..5   : White pieces  [P, N, B, R, Q, K]
#  6..11  : Black pieces  [p, n, b, r, q, k]
#  12     : Side to move (1 = white to move, else 0)  [full board]
#  13     : Castling right: White king side
#  14     : Castling right: White queen side
#  15     : Castling right: Black king side
#  16     : Castling right: Black queen side
#  17     : En passant file (column of ep square set to 1 for all ranks)
#  18     : Halfmove clock (normalized)  [full board]
#  19     : In check (side to move)
#  20     : Repetition >= 2
#  21     : Repetition >= 3
#  22     : Reserved (zeros)
#  23     : Reserved (zeros)
#
# Output shape: (24, 8, 8)  -- NCHW


PIECE_TO_INDEX = {
    chess.PAWN: 0,
    chess.KNIGHT: 1,
    chess.BISHOP: 2,
    chess.ROOK: 3,
    chess.QUEEN: 4,
    chess.KING: 5,
}

NUM_CHANNELS = 24
BOARD_SIZE = 8


def _empty_planes() -> np.ndarray:
    return np.zeros((NUM_CHANNELS, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)


def _square_to_coords(square: chess.Square) -> tuple[int, int]:
    """Convert python-chess square (0..63) to (row, col) in 8x8 tensor.

    We use row 0 = rank 8 (top), row 7 = rank 1 (bottom)
    and col 0 = file 'a'.
    """
    rank = chess.square_rank(square)  # 0 (rank1) .. 7 (rank8)
    file = chess.square_file(square)  # 0 (a) .. 7 (h)
    row = 7 - rank  # flip so that row 0 is top
    col = file
    return row, col


def _encode_pieces(board: chess.Board, planes: np.ndarray) -> None:
    """Fill planes 0..11 with piece locations."""
    for square, piece in board.piece_map().items():
        row, col = _square_to_coords(square)
        base_idx = PIECE_TO_INDEX[piece.piece_type]
        if piece.color == chess.WHITE:
            plane_idx = base_idx  # 0..5
        else:
            plane_idx = 6 + base_idx  # 6..11
        planes[plane_idx, row, col] = 1.0


def _encode_side_to_move(board: chess.Board, planes: np.ndarray) -> None:
    planes[12, :, :] = 1.0 if board.turn == chess.WHITE else 0.0


def _encode_castling(board: chess.Board, planes: np.ndarray) -> None:
    # full-board binary planes
    if board.has_kingside_castling_rights(chess.WHITE):
        planes[13, :, :] = 1.0
    if board.has_queenside_castling_rights(chess.WHITE):
        planes[14, :, :] = 1.0
    if board.has_kingside_castling_rights(chess.BLACK):
        planes[15, :, :] = 1.0
    if board.has_queenside_castling_rights(chess.BLACK):
        planes[16, :, :] = 1.0


def _encode_en_passant(board: chess.Board, planes: np.ndarray) -> None:
    if board.ep_square is None:
        return
    file_idx = chess.square_file(board.ep_square)  # 0..7
    # set the whole column to 1
    planes[17, :, file_idx] = 1.0


def _encode_halfmove_clock(board: chess.Board, planes: np.ndarray) -> None:
    # normalize to [0,1], cap at 99 to avoid extreme values
    value = min(board.halfmove_clock, 99) / 99.0
    planes[18, :, :] = value


def _encode_in_check(board: chess.Board, planes: np.ndarray) -> None:
    if board.is_check():
        planes[19, :, :] = 1.0


def _encode_repetition(board: chess.Board, planes: np.ndarray) -> None:
    # python-chess can test is_repetition(n)
    if board.is_repetition(2):
        planes[20, :, :] = 1.0
    if board.is_repetition(3):
        planes[21, :, :] = 1.0


def state_to_tensor(board: chess.Board) -> torch.Tensor:
    """Encode a python-chess Board into a (24, 8, 8) float32 tensor.

    Channels:
        0..5   : white pieces (P,N,B,R,Q,K)
        6..11  : black pieces (p,n,b,r,q,k)
        12     : side to move
        13..16 : castling rights (WK, WQ, BK, BQ)
        17     : en passant file (if any)
        18     : halfmove clock normalized
        19     : in check (side to move)
        20     : repetition >= 2
        21     : repetition >= 3
        22..23 : reserved zeros
    """
    planes = _empty_planes()
    _encode_pieces(board, planes)
    _encode_side_to_move(board, planes)
    _encode_castling(board, planes)
    _encode_en_passant(board, planes)
    _encode_halfmove_clock(board, planes)
    _encode_in_check(board, planes)
    _encode_repetition(board, planes)
    # planes[22] and planes[23] stay zero
    return torch.from_numpy(planes)

