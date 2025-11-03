import numpy as np
import pytest
import chess
from leelax.env.encode import state_to_tensor

# === Encoding conventions (24 channels) ===
#
#  0..5   : White pieces  [P, N, B, R, Q, K]
#  6..11  : Black pieces  [p, n, b, r, q, k]
#  12     : Side to move
#  13..16 : Castling rights (WK, WQ, BK, BQ)
#  17     : En passant file
#  18     : Halfmove clock normalized
#  19     : In check (side to move)
#  20     : Repetition >= 2
#  21     : Repetition >= 3
#  22..23 : Reserved (zeros)
#
# Expected output: (24, 8, 8)

PIECE_INDEX = {
    chess.PAWN: 0,
    chess.KNIGHT: 1,
    chess.BISHOP: 2,
    chess.ROOK: 3,
    chess.QUEEN: 4,
    chess.KING: 5,
}


def tensor_to_np(t):
    if hasattr(t, "detach"):
        return t.detach().cpu().numpy()
    return np.asarray(t)


def count_ones(plane):
    return int(np.sum(np.isclose(plane, 1.0)))


def assert_binary_plane(plane):
    assert np.all((plane == 0) | (plane == 1)), "plane must be binary {0,1}"


def make_board(fen: str) -> chess.Board:
    return chess.Board() if fen == "startpos" else chess.Board(fen)


def test_shape_and_dtype_startpos():
    board = make_board("startpos")
    x = tensor_to_np(state_to_tensor(board))
    assert x.shape == (24, 8, 8), f"Unexpected shape {x.shape}"
    assert x.dtype in (np.float32, np.float64)


def test_piece_planes_counts_startpos():
    board = make_board("startpos")
    x = tensor_to_np(state_to_tensor(board))
    # 16 white + 16 black pieces in startpos
    white_total = sum(count_ones(x[i]) for i in range(0, 6))
    black_total = sum(count_ones(x[i]) for i in range(6, 12))
    assert white_total == 16, f"White count {white_total}"
    assert black_total == 16, f"Black count {black_total}"


@pytest.mark.parametrize(
    "fen,white_to_move",
    [
        ("startpos", True),
        ("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1", False),
    ],
)
def test_side_to_move_plane(fen, white_to_move):
    board = make_board(fen)
    x = tensor_to_np(state_to_tensor(board))
    plane = x[12]
    assert_binary_plane(plane)
    if white_to_move:
        assert np.all(plane == 1)
    else:
        assert np.all(plane == 0)


@pytest.mark.parametrize(
    "fen,rights",
    [
        ("startpos", (True, True, True, True)),
        ("4k3/8/8/8/8/8/8/R3K2R w K - 0 1", (True, False, False, False)),
        ("r3k2r/8/8/8/8/8/8/4K3 w kq - 0 1", (False, False, True, True)),
    ],
)
def test_castling_right_planes(fen, rights):
    board = make_board(fen)
    x = tensor_to_np(state_to_tensor(board))
    (WK, WQ, BK, BQ) = rights
    for idx, flag in zip((13, 14, 15, 16), (WK, WQ, BK, BQ)):
        plane = x[idx]
        assert_binary_plane(plane)
        if flag:
            assert np.all(plane == 1)
        else:
            assert np.all(plane == 0)


@pytest.mark.parametrize(
    "fen,ep_file_expected",
    [
        ("startpos", None),
        ("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1", 4),
    ],
)
def test_en_passant_plane(fen, ep_file_expected):
    board = make_board(fen)
    x = tensor_to_np(state_to_tensor(board))
    plane = x[17]
    if ep_file_expected is None:
        assert np.all(plane == 0)
    else:
        ones_per_rank = [plane[r, ep_file_expected] for r in range(8)]
        assert all(v == 1 for v in ones_per_rank)
        other_cols = [c for c in range(8) if c != ep_file_expected]
        assert all(np.all(plane[:, c] == 0) for c in other_cols)


@pytest.mark.parametrize(
    "fen,halfmove",
    [
        ("startpos", 0),
        ("rnbqkbnr/pppp1ppp/8/4p3/2B5/5N2/PPPPPPPP/RNBQK2R w KQkq - 7 4", 7),
    ],
)
def test_halfmove_plane(fen, halfmove):
    board = make_board(fen)
    x = tensor_to_np(state_to_tensor(board))
    plane = x[18]
    expected = min(halfmove, 99) / 99.0
    assert np.allclose(plane, expected)


def test_in_check_plane_and_repetition_flags():
    # in-check position
    board = chess.Board("4k3/8/8/8/8/8/4R3/4K3 b - - 0 1")
    x = tensor_to_np(state_to_tensor(board))
    assert np.all(x[19] == 1), "in-check plane should be 1 when in check"

    # repetition 2x
    board = chess.Board()
    seq = [chess.Move.from_uci(m) for m in ("g1f3", "g8f6", "f3g1", "f6g8", "g1f3", "g8f6")]
    for m in seq:
        board.push(m)
    x = tensor_to_np(state_to_tensor(board))
    assert np.all(x[20] == 1)
    # repetition 3x
    board.push(chess.Move.from_uci("f3g1"))
    board.push(chess.Move.from_uci("f6g8"))
    x = tensor_to_np(state_to_tensor(board))
    assert np.all(x[21] == 1)


def test_reserved_planes_zero():
    board = chess.Board()
    x = tensor_to_np(state_to_tensor(board))
    for ch in (22, 23):
        assert np.all(x[ch] == 0), f"reserved channel {ch} must be zero"


def test_no_nans_or_infs():
    board = chess.Board()
    x = tensor_to_np(state_to_tensor(board))
    assert np.isfinite(x).all()

