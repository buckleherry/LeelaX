from __future__ import annotations
import chess
import numpy as np

# We use AlphaZero-style action space:
#  - 56 sliding moves (8 directions × 1..7)
#  - 8 knight moves
#  - 12 promotions (3 forward dirs × 4 pieces)  -- FROM PLAYER POV
# Total: 56 + 8 + 12 = 76
# Policy size: 8 * 8 * 76 = 4864

DIRECTIONS = [
    (1, 0),   # forward
    (-1, 0),  # backward
    (0, 1),   # right
    (0, -1),  # left
    (1, 1),
    (1, -1),
    (-1, 1),
    (-1, -1),
]

KNIGHT_STEPS = [
    (2, 1), (1, 2), (-1, 2), (-2, 1),
    (-2, -1), (-1, -2), (1, -2), (2, -1),
]

PROMO_DIRS = [
    (1, -1),  # forward-left
    (1, 0),   # forward
    (1, 1),   # forward-right
]

PROMO_PIECES = [
    chess.KNIGHT,
    chess.BISHOP,
    chess.ROOK,
    chess.QUEEN,
]

NUM_MOVE_TYPES = 76
TOTAL_POLICY_SIZE = 8 * 8 * NUM_MOVE_TYPES

MOVE_TYPE_TABLE: dict[tuple, int] = {}
INDEX_TO_DELTA: list[tuple] = []

idx = 0

# 56 sliding
for d_rank, d_file in DIRECTIONS:
    for dist in range(1, 8):
        MOVE_TYPE_TABLE[(d_rank * dist, d_file * dist)] = idx
        INDEX_TO_DELTA.append((d_rank * dist, d_file * dist))
        idx += 1

# 8 knights
for step in KNIGHT_STEPS:
    MOVE_TYPE_TABLE[step] = idx
    INDEX_TO_DELTA.append(step)
    idx += 1

# 12 promotions (player POV)
for d_rank, d_file in PROMO_DIRS:
    for p in PROMO_PIECES:
        MOVE_TYPE_TABLE[(d_rank, d_file, p)] = idx
        INDEX_TO_DELTA.append((d_rank, d_file, p))
        idx += 1

assert idx == NUM_MOVE_TYPES


def _move_to_index_white_pov(move: chess.Move) -> int:
    """Assumes move is already from white/player POV."""
    from_sq = move.from_square
    to_sq = move.to_square
    rf = chess.square_rank(from_sq)
    ff = chess.square_file(from_sq)
    rt = chess.square_rank(to_sq)
    ft = chess.square_file(to_sq)
    d_rank = rt - rf
    d_file = ft - ff

    if move.promotion:
        key = (d_rank, d_file, move.promotion)
        move_type = MOVE_TYPE_TABLE.get(key)
        if move_type is None:
            raise KeyError(f"unmappable promotion delta {key}")
    else:
        move_type = MOVE_TYPE_TABLE.get((d_rank, d_file))
        if move_type is None:
            raise KeyError(f"unmappable move delta {(d_rank, d_file)}")

    return from_sq * NUM_MOVE_TYPES + move_type


def _index_to_move_white_pov(index: int) -> chess.Move:
    from_sq = index // NUM_MOVE_TYPES
    move_type = index % NUM_MOVE_TYPES
    delta = INDEX_TO_DELTA[move_type]

    rf = chess.square_rank(from_sq)
    ff = chess.square_file(from_sq)

    if len(delta) == 3:
        d_rank, d_file, promo = delta
    else:
        d_rank, d_file = delta
        promo = None

    rt = rf + d_rank
    ft = ff + d_file
    if not (0 <= rt < 8 and 0 <= ft < 8):
        # off board – caller must guard
        return chess.Move(from_sq, from_sq)

    to_sq = chess.square(ft, rt)
    return chess.Move(from_sq, to_sq, promotion=promo)


def to_canonical_move(board: chess.Board, move: chess.Move) -> chess.Move:
    """Convert real-board move to canonical (white-to-move) move."""
    if board.turn == chess.WHITE:
        return move
    # mirror squares for black
    from_sq = chess.square_mirror(move.from_square)
    to_sq = chess.square_mirror(move.to_square)
    return chess.Move(from_sq, to_sq, promotion=move.promotion)


def from_canonical_move(board: chess.Board, move: chess.Move) -> chess.Move:
    """Convert canonical move back to real-board move."""
    if board.turn == chess.WHITE:
        return move
    from_sq = chess.square_mirror(move.from_square)
    to_sq = chess.square_mirror(move.to_square)
    return chess.Move(from_sq, to_sq, promotion=move.promotion)


def move_to_index(board: chess.Board, move: chess.Move) -> int:
    """Real board + real move → canonical index."""
    canon = to_canonical_move(board, move)
    return _move_to_index_white_pov(canon)


def index_to_move(board: chess.Board, index: int) -> chess.Move:
    """Canonical index → real-board move."""
    canon_move = _index_to_move_white_pov(index)
    real_move = from_canonical_move(board, canon_move)
    return real_move


def legal_policy_mask(board: chess.Board) -> np.ndarray:
    mask = np.zeros(TOTAL_POLICY_SIZE, dtype=np.float32)
    for mv in board.legal_moves:
        try:
            idx = move_to_index(board, mv)
        except KeyError:
            continue
        mask[idx] = 1.0
    return mask
