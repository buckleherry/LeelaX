from __future__ import annotations
import chess
import numpy as np

# --- Constants ---
BOARD_SIZE = 8
DIRECTIONS = [
    (1, 0),  # N
    (-1, 0), # S
    (0, 1),  # E
    (0, -1), # W
    (1, 1),  # NE
    (1, -1), # NW
    (-1, 1), # SE
    (-1, -1) # SW
]
KNIGHT_STEPS = [
    (2, 1), (1, 2), (-1, 2), (-2, 1),
    (-2, -1), (-1, -2), (1, -2), (2, -1)
]
PROMO_PIECES = [chess.KNIGHT, chess.BISHOP, chess.ROOK]
# Promotions (White moving "up")
PROMO_DIRS = [(1, -1), (1, 0), (1, 1)]

NUM_MOVE_TYPES = 73  # 8*7 + 8 + 9


def _square_to_coords(sq: int) -> tuple[int, int]:
    return chess.square_rank(sq), chess.square_file(sq)


def _coords_to_square(rank: int, file: int) -> int | None:
    if 0 <= rank < 8 and 0 <= file < 8:
        return chess.square(file, rank)
    return None


# --- Core Mapping ---

MOVE_TYPE_TABLE: dict[tuple[int, int], int] = {}
INDEX_TO_DELTA: list[tuple[int, int]] = []

# Populate sliding directions (8×7)
idx = 0
for d_rank, d_file in DIRECTIONS:
    for dist in range(1, 8):
        MOVE_TYPE_TABLE[(d_rank * dist, d_file * dist)] = idx
        INDEX_TO_DELTA.append((d_rank * dist, d_file * dist))
        idx += 1

# Knights (8)
for step in KNIGHT_STEPS:
    MOVE_TYPE_TABLE[step] = idx
    INDEX_TO_DELTA.append(step)
    idx += 1

# Promotions (3 dirs × 3 pieces)
for d_rank, d_file in PROMO_DIRS:
    for p in PROMO_PIECES:
        MOVE_TYPE_TABLE[(d_rank, d_file, p)] = idx
        INDEX_TO_DELTA.append((d_rank, d_file, p))
        idx += 1

assert idx == NUM_MOVE_TYPES, f"expected 73 move types, got {idx}"


# --- Public API ---

def move_to_index(move: chess.Move) -> int:
    """Convert a chess.Move into a policy index [0..4671]."""
    from_sq = move.from_square
    to_sq = move.to_square
    rank_from, file_from = chess.square_rank(from_sq), chess.square_file(from_sq)
    rank_to, file_to = chess.square_rank(to_sq), chess.square_file(to_sq)
    d_rank, d_file = rank_to - rank_from, file_to - file_from

    if move.promotion:
        move_type = MOVE_TYPE_TABLE[(d_rank, d_file, move.promotion)]
    else:
        move_type = MOVE_TYPE_TABLE.get((d_rank, d_file))
        if move_type is None:
            raise ValueError(f"Illegal or unmapped move delta {(d_rank, d_file)}")

    return from_sq * NUM_MOVE_TYPES + move_type


def index_to_move(index: int) -> chess.Move:
    """Inverse mapping from index -> move (approximate)."""
    from_sq = index // NUM_MOVE_TYPES
    move_type = index % NUM_MOVE_TYPES

    delta = INDEX_TO_DELTA[move_type]
    rank_from, file_from = chess.square_rank(from_sq), chess.square_file(from_sq)

    # Promotion case (tuple of 3)
    if isinstance(delta, tuple) and len(delta) == 3:
        d_rank, d_file, promo = delta
    else:
        d_rank, d_file, promo = delta[0], delta[1], None

    to_sq = _coords_to_square(rank_from + d_rank, file_from + d_file)
    if to_sq is None:
        raise ValueError(f"Index {index} leads off-board.")
    return chess.Move(from_sq, to_sq, promotion=promo)


def legal_policy_mask(board: chess.Board) -> np.ndarray:
    """Return a binary mask [4672] marking legal moves = 1."""
    mask = np.zeros(BOARD_SIZE * BOARD_SIZE * NUM_MOVE_TYPES, dtype=np.float32)
    for mv in board.legal_moves:
        try:
            mask[move_to_index(mv)] = 1.0
        except (KeyError, ValueError):
            continue  # some underpromotions may not exist
    return mask
