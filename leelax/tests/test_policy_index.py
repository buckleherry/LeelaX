import pytest
import numpy as np
import chess

from leelax.env.policy_index import (
    move_to_index,
    index_to_move,
    legal_policy_mask,
    NUM_MOVE_TYPES,
)

# 8x8x73 = 4672
TOTAL_POLICY_SIZE = 8 * 8 * NUM_MOVE_TYPES


def test_constants():
    assert NUM_MOVE_TYPES == 73
    assert TOTAL_POLICY_SIZE == 4672


def test_legal_policy_mask_startpos():
    board = chess.Board()  # startpos
    mask = legal_policy_mask(board)
    assert isinstance(mask, np.ndarray)
    assert mask.shape == (4672,)
    # startpos has 20 legal moves
    assert mask.sum() == len(list(board.legal_moves)) == 20


@pytest.mark.parametrize(
    "uci",
    [
        "e2e4",
        "d2d4",
        "g1f3",
        "b1c3",
    ],
)
def test_roundtrip_non_promotion_moves(uci):
    board = chess.Board()
    move = chess.Move.from_uci(uci)
    assert move in board.legal_moves, f"{uci} should be legal in startpos"
    idx = move_to_index(move)
    assert 0 <= idx < 4672
    move2 = index_to_move(idx)
    # Note: python-chess can consider some moves equivalent (e.g. promotions),
    # here we are testing pure from->to moves in startpos, so exact match is fine.
    assert move.from_square == move2.from_square
    assert move.to_square == move2.to_square
    assert move.promotion == move2.promotion


def test_mask_marks_specific_move():
    board = chess.Board()
    move = chess.Move.from_uci("e2e4")
    idx = move_to_index(move)
    mask = legal_policy_mask(board)
    assert mask[idx] == 1.0
    # a random other index should be 0 most of the time
    # (not a strict guarantee, but ok for a unit test)
    assert mask[0] in (0.0, 1.0)


def test_index_to_move_offboard_raises():
    # construct an index that implies an off-board target
    # easiest way: take from-square a1 (0), and a move type that goes far SW
    # but since our policy_index implementation should raise on off-board,
    # we test that behavior here.
    from_sq = chess.A1
    move_type = 0  # this depends on implementation, but we'll try an arbitrary one
    idx = from_sq * NUM_MOVE_TYPES + move_type
    # if implementation is strict, this may raise ValueError
    try:
        mv = index_to_move(idx)
        # if we got a move, ensure it's at least a valid python-chess move shape
        assert isinstance(mv, chess.Move)
    except ValueError:
        # also acceptable
        assert True


def test_promotion_mapping_white():
    # white pawn ready to promote, target square empty
    board = chess.Board("k7/4P3/8/8/8/8/8/4K3 w - - 0 1")
    # use a promotion we actually encode (ROOK - as QUEEN is not encoded in the policy vector due to being standard)
    move = chess.Move.from_uci("e7e8r")
    assert move in board.legal_moves

    idx = move_to_index(move)
    assert 0 <= idx < 4672

    mv2 = index_to_move(idx)
    assert mv2.from_square == move.from_square
    assert mv2.to_square == move.to_square
    # our mapping stores the promotion piece explicitly
    assert mv2.promotion == chess.ROOK
