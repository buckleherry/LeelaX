import pytest
import numpy as np
import chess

from leelax.env.policy_index import (
    move_to_index,
    index_to_move,
    legal_policy_mask,
    NUM_MOVE_TYPES,
)

TOTAL_POLICY_SIZE = 8 * 8 * NUM_MOVE_TYPES  # should be 4864


def test_constants():
    assert NUM_MOVE_TYPES == 76
    assert TOTAL_POLICY_SIZE == 4864


def test_legal_policy_mask_startpos():
    board = chess.Board()
    mask = legal_policy_mask(board)
    assert isinstance(mask, np.ndarray)
    assert mask.shape == (4864,)
    assert mask.sum() == len(list(board.legal_moves))


@pytest.mark.parametrize(
    "uci",
    ["e2e4", "d2d4", "g1f3", "b1c3"],
)
def test_roundtrip_non_promotion_moves(uci):
    board = chess.Board()
    mv = chess.Move.from_uci(uci)
    assert mv in board.legal_moves
    idx = move_to_index(board, mv)
    mv2 = index_to_move(board, idx)
    # because we do canonical ↔ real, from/to must match
    assert mv2.from_square == mv.from_square
    assert mv2.to_square == mv.to_square


def test_promotion_white():
    # white to move, promotion possible
    board = chess.Board("k7/4P3/8/8/8/8/8/4K3 w - - 0 1")
    mv = chess.Move.from_uci("e7e8q")
    assert mv in board.legal_moves
    idx = move_to_index(board, mv)
    assert 0 <= idx < 4864
    mv2 = index_to_move(board, idx)
    assert mv2.from_square == mv.from_square
    assert mv2.to_square == mv.to_square
    # promotion piece might be preserved depending on mapping; at least it's mappable


def test_mask_marks_specific_move():
    board = chess.Board()
    mv = chess.Move.from_uci("e2e4")
    idx = move_to_index(board, mv)
    mask = legal_policy_mask(board)
    assert mask[idx] == 1.0

