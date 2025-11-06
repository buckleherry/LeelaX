from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import chess

# Channel layout (24):
#  0..5   : White pieces  [P, N, B, R, Q, K]
#  6..11  : Black pieces  [p, n, b, r, q, k]
#  12     : Side to move (1 = current player to move)  [full board]
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
# Output shape: (24, 8, 8)


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
    rank = chess.square_rank(square)  # 0..7 (from white's POV, 0=rank1)
    file = chess.square_file(square)  # 0..7
    row = 7 - rank  # row 0 = top
    col = file
    return row, col


def _encode_pieces(board: chess.Board, planes: np.ndarray) -> None:
    for square, piece in board.piece_map().items():
        row, col = _square_to_coords(square)
        base_idx = PIECE_TO_INDEX[piece.piece_type]
        if piece.color == chess.WHITE:
            planes[base_idx, row, col] = 1.0
        else:
            planes[6 + base_idx, row, col] = 1.0


def _encode_side_to_move(board: chess.Board, planes: np.ndarray) -> None:
    planes[12, :, :] = 1.0 if board.turn == chess.WHITE else 0.0


def _encode_castling(board: chess.Board, planes: np.ndarray) -> None:
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
    file_idx = chess.square_file(board.ep_square)
    planes[17, :, file_idx] = 1.0


def _encode_halfmove_clock(board: chess.Board, planes: np.ndarray) -> None:
    value = min(board.halfmove_clock, 99) / 99.0
    planes[18, :, :] = value


def _encode_in_check(board: chess.Board, planes: np.ndarray) -> None:
    if board.is_check():
        planes[19, :, :] = 1.0


def _encode_repetition(board: chess.Board, planes: np.ndarray) -> None:
    if board.is_repetition(2):
        planes[20, :, :] = 1.0
    if board.is_repetition(3):
        planes[21, :, :] = 1.0


def _canonicalize_planes_for_black(planes: np.ndarray) -> np.ndarray:
    """Rotate 180° and swap white/black-specific planes so that
    the side to move is always 'white' from the network's view.
    """
    # rotate all planes 180°
    planes = planes[:, ::-1, ::-1].copy()

    # swap piece planes: 0..5 <-> 6..11
    for i in range(6):
        tmp = planes[i].copy()
        planes[i] = planes[6 + i]
        planes[6 + i] = tmp

    # swap castling planes:
    # 13 (Wk) <-> 15 (Bk)
    # 14 (Wq) <-> 16 (Bq)
    tmp = planes[13].copy()
    planes[13] = planes[15]
    planes[15] = tmp

    tmp = planes[14].copy()
    planes[14] = planes[16]
    planes[16] = tmp

    # side-to-move plane: after canonicalization, it's always white to move
    planes[12, :, :] = 1.0

    return planes


def state_to_tensor(board: chess.Board, canonical: bool = True) -> torch.Tensor:
    """Encode board into (24, 8, 8). If canonical=True, always from
    side-to-move perspective (i.e. black is flipped).
    """
    planes = _empty_planes()
    _encode_pieces(board, planes)
    _encode_side_to_move(board, planes)
    _encode_castling(board, planes)
    _encode_en_passant(board, planes)
    _encode_halfmove_clock(board, planes)
    _encode_in_check(board, planes)
    _encode_repetition(board, planes)

    if canonical and board.turn == chess.BLACK:
        planes = _canonicalize_planes_for_black(planes)

    return torch.from_numpy(planes)

