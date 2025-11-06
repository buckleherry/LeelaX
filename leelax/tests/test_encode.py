import numpy as np
import chess

from leelax.env.encode import state_to_tensor


def tensor_to_np(t):
    return t.detach().cpu().numpy()


def assert_binary_plane(plane):
    assert plane.shape == (8, 8)
    assert np.all((plane == 0) | (plane == 1))


def make_board(fen: str) -> chess.Board:
    if fen == "startpos":
        return chess.Board()
    return chess.Board(fen)


def test_startpos_shape():
    board = chess.Board()
    x = tensor_to_np(state_to_tensor(board, canonical=True))
    assert x.shape == (24, 8, 8)


def test_side_to_move_plane_canonical():
    # in canonical view we ALWAYS show "current player = white"
    b_white = chess.Board()  # white to move
    xw = tensor_to_np(state_to_tensor(b_white, canonical=True))
    assert np.all(xw[12] == 1)

    b_black = chess.Board("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1")
    xb = tensor_to_np(state_to_tensor(b_black, canonical=True))
    # even though it's black to move, canonical view says "1"
    assert np.all(xb[12] == 1)


def test_en_passant_plane_canonical():
    # position with ep square on file 'e' (file index 4) for BLACK to move
    fen = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1"
    board = chess.Board(fen)
    x = tensor_to_np(state_to_tensor(board, canonical=True))
    plane = x[17]

    # in canonical view we rotated 180°, so file gets mirrored: new_file = 7 - old_file
    original_file = chess.square_file(board.ep_square)  # 4
    canonical_file = 7 - original_file                  # 3
    # whole column should be 1
    for r in range(8):
        assert plane[r, canonical_file] == 1.0

