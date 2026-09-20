"""Human chess games as training data: what is stored, and how it becomes input.

**Store what happened, not what the network sees** (decision log). A position is
kept as its board, the side to move and its rights, the move that was played and
the game it came from - 44 bytes - and turned into the network's 19 planes only
when a batch is drawn for training. Changing the input planes or the move table
then never means reading a month of games again.

That puts two encoders in the project: :meth:`Chess.encode` for one live
position, and :func:`encode` here for a batch of stored ones. They must agree
exactly - a network trained on one and played with the other would be fed
positions it never saw - so the tests hold this one to the other, plane by
plane, and to :func:`~caissa.games.chess.action_of` action by action.
"""

from __future__ import annotations

from pathlib import Path

import chess
import numpy as np

from caissa.games.chess import ACTIONS, INDEX, PIECES, PLANES, SQUARES, ChessState

#: One position. Squares are absolute - a1 is 0, as White sees the board - and
#: the mover's mirrored view is only taken when encoding.
POSITION = np.dtype([
    # 64 squares, a nibble each: 0 empty, 1-6 White's P N B R Q K, 7-12 Black's.
    ("board", np.uint8, 32),
    ("black", np.uint8),        # 1 when Black is to move
    ("castling", np.uint8),     # bits: White O-O, White O-O-O, Black O-O, Black O-O-O
    ("en_passant", np.uint8),   # where a legal en passant capture lands, else 255
    ("halfmove", np.uint8),     # the fifty-move counter, capped at 255
    ("repetitions", np.uint8),  # times this position has occurred, counting now
    ("move", np.uint16),        # from | to << 6 | promotion piece << 12
    ("game", np.uint32),        # row in the games table
    ("ply", np.uint16),         # plies played before this position
])

#: One game.
GAME = np.dtype([
    ("result", np.int8),        # +1 White won, 0 drawn, -1 Black won
    ("white_elo", np.uint16),
    ("black_elo", np.uint16),
    ("base", np.uint16),        # time control: seconds each, 0 for correspondence
    ("increment", np.uint8),
    ("plies", np.uint16),
    ("first", np.uint64),       # row of its first position
    ("validation", np.uint8),   # 1 if held out
])

NO_SQUARE = 255
#: Nibble codes per piece plane, in :data:`PIECES` order, White's then Black's.
_CODES = np.arange(1, 13, dtype=np.uint8)[:, None]
_MIRROR = np.arange(SQUARES) ^ 56

#: (from, to, promotion column) in the mover's frame -> action; -1 where none.
#: Column 0 is no promotion or a queen's - a queen-like move either way - and
#: 1-3 the knight, bishop and rook.
_PROMOTION_COLUMN = np.zeros(7, dtype=np.int64)
_PROMOTION_COLUMN[[chess.KNIGHT, chess.BISHOP, chess.ROOK]] = [1, 2, 3]
ACTION_TABLE = np.full((SQUARES, SQUARES, 4), -1, dtype=np.int64)
for (_start, _end, _promotion), _action in INDEX.items():
    ACTION_TABLE[_start, _end, _PROMOTION_COLUMN[_promotion or 0]] = _action


def record(state: ChessState, move: chess.Move, game: int, ply: int) -> tuple:
    """A stored position: ``state`` just before ``move`` was played in it."""
    board = state.board
    masks = ([board.pieces_mask(piece, chess.WHITE) for piece in PIECES]
             + [board.pieces_mask(piece, chess.BLACK) for piece in PIECES])
    bits = np.unpackbits(np.array(masks, dtype="<u8").view(np.uint8), bitorder="little")
    codes = (bits.reshape(12, SQUARES) * _CODES).sum(axis=0).astype(np.uint8)
    castling = (int(board.has_kingside_castling_rights(chess.WHITE))
                | int(board.has_queenside_castling_rights(chess.WHITE)) << 1
                | int(board.has_kingside_castling_rights(chess.BLACK)) << 2
                | int(board.has_queenside_castling_rights(chess.BLACK)) << 3)
    en_passant = board.ep_square if board.has_legal_en_passant() else NO_SQUARE
    return (
        codes[0::2] | (codes[1::2] << 4),
        int(board.turn == chess.BLACK),
        castling,
        en_passant,
        min(board.halfmove_clock, 255),
        min(state.repetitions, 255),
        move.from_square | move.to_square << 6 | (move.promotion or 0) << 12,
        game,
        ply,
    )


def squares(positions: np.ndarray) -> np.ndarray:
    """The nibble boards unpacked: ``(n, 64)`` piece codes, absolute squares."""
    packed = positions["board"]
    codes = np.empty((len(positions), SQUARES), dtype=np.uint8)
    codes[:, 0::2] = packed & 15
    codes[:, 1::2] = packed >> 4
    return codes


def encode(positions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Network input and action indices for a batch of stored positions.

    The vectorised twin of :meth:`Chess.encode` and
    :func:`~caissa.games.chess.action_of`: planes ``(n, 19, 8, 8)`` float32 and
    actions ``(n,)``, each in the frame of the player to move.
    """
    n = len(positions)
    black = positions["black"].astype(bool)
    codes = squares(positions)
    # For Black the board is mirrored and the colours swap: codes 7-12 become
    # the mover's (planes 0-5) and 1-6 the opponent's (planes 6-11).
    codes[black] = codes[black][:, _MIRROR]
    swapped = np.where(codes == 0, 0, np.where(codes <= 6, codes + 6, codes - 6))
    relative = np.where(black[:, None], swapped, codes).astype(np.int64)

    planes = np.zeros((n, PLANES, SQUARES), dtype=np.float32)
    rows, cells = np.nonzero(relative)
    planes[rows, relative[rows, cells] - 1, cells] = 1.0

    rights = positions["castling"].astype(np.int64)
    own, theirs = np.where(black, rights >> 2, rights), np.where(black, rights, rights >> 2)
    planes[:, 12] = (own & 1)[:, None]
    planes[:, 13] = ((own >> 1) & 1)[:, None]
    planes[:, 14] = (theirs & 1)[:, None]
    planes[:, 15] = ((theirs >> 1) & 1)[:, None]

    en_passant = positions["en_passant"].astype(np.int64)
    marked = en_passant != NO_SQUARE
    frame = np.where(black, en_passant ^ 56, en_passant)
    planes[np.flatnonzero(marked), 16, frame[marked]] = 1.0
    planes[:, 17] = (np.minimum(positions["halfmove"], 100) / 100)[:, None]
    planes[:, 18] = (positions["repetitions"] > 1)[:, None]

    moves = positions["move"].astype(np.int64)
    start, end, promotion = moves & 63, (moves >> 6) & 63, moves >> 12
    start = np.where(black, start ^ 56, start)
    end = np.where(black, end ^ 56, end)
    actions = ACTION_TABLE[start, end, _PROMOTION_COLUMN[promotion]]
    return planes.reshape(n, PLANES, 8, 8), actions


def load_months(root: Path, months: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Every stored position and game of the given months, numbered as one table."""
    all_positions, all_games = [], []
    games_so_far = positions_so_far = 0
    for month in months:
        positions = np.load(root / month / "positions.npy")
        games = np.load(root / month / "games.npy")
        positions["game"] += games_so_far
        games["first"] += positions_so_far
        all_positions.append(positions)
        all_games.append(games)
        games_so_far += len(games)
        positions_so_far += len(positions)
    return np.concatenate(all_positions), np.concatenate(all_games)


def values(positions: np.ndarray, games: np.ndarray) -> np.ndarray:
    """Each position's game result for the player to move there: the value target."""
    result = games["result"][positions["game"]].astype(np.float32)
    return np.where(positions["black"].astype(bool), -result, result)


def board_of(stored: np.void) -> chess.Board:
    """A stored position back as a python-chess board, for checks and debugging."""
    codes = squares(np.asarray(stored).reshape(1))[0]
    board = chess.Board.empty()
    for square in np.flatnonzero(codes):
        code = int(codes[square])
        board.set_piece_at(int(square), chess.Piece(PIECES[(code - 1) % 6], code <= 6))
    board.turn = chess.BLACK if stored["black"] else chess.WHITE
    rights = int(stored["castling"])
    fen = "".join(letter for bit, letter in enumerate("KQkq") if rights >> bit & 1) or "-"
    board.set_castling_fen(fen)
    board.ep_square = None if stored["en_passant"] == NO_SQUARE else int(stored["en_passant"])
    board.halfmove_clock = int(stored["halfmove"])
    return board


def move_of(stored: np.void) -> chess.Move:
    """The move that was played from a stored position."""
    move = int(stored["move"])
    return chess.Move(move & 63, (move >> 6) & 63, (move >> 12) or None)


def human_samples(positions: np.ndarray, games: np.ndarray, count: int,
                  rng: np.random.Generator) -> list["Sample"]:
    """Stored human positions as training examples, for rehearsal during self-play.

    Self-play trains a network on games it played itself, and the value head is
    the part that suffers: every position of a game carries that game's single
    result, so a window of a few hundred games is a few hundred labels however
    many positions it holds. Fitted hard enough, a calibrated value head becomes
    an overconfident one - and an overconfident value head turns search from
    something that improves on the priors into something that overrules them.

    Mixing these into every batch is *rehearsal*: the old task is kept in front of
    the network while it learns the new one, so the value head keeps answering to
    39 million human outcomes rather than to a few hundred of its own games.

    Drawn from the training games only. The held-out ones are the exam that says
    whether this worked, and an exam you have revised is not a measurement.
    """
    from caissa.selfplay import Sample

    training = np.flatnonzero(~games["validation"][positions["game"]].astype(bool))
    rows = rng.choice(training, size=min(count, len(training)), replace=False)
    chosen = positions[np.sort(rows)]
    planes, actions = encode(chosen)
    labels = values(chosen, games)

    samples = []
    for index, action in enumerate(actions):
        # The policy target is the move the human played, as a distribution with
        # all of its mass on that one move - a search that visited nothing else.
        policy = np.zeros(ACTIONS, dtype=np.float32)
        policy[action] = 1.0
        samples.append(Sample(encoded=planes[index], policy=policy,
                              value=float(labels[index])))
    return samples
