"""Chess: everything the ladder has built, all at once.

The rules are not written here. Move generation, check, castling, en passant,
promotion and the draw rules come from python-chess, which is correct and tested
far beyond anything a hand-written version would be (see the decision log). What
*is* written here is the translation between chess and the framework - and that
translation is where the silent bugs would live.

**Canonical perspective, by mirroring.** Chess is not symmetric the way the
earlier boards were: White's pawns go up the board and Black's go down. So when
Black is to move, the board is mirrored - ranks reversed, colours swapped - before
the network sees it. Every position then looks as if the player to move were
White, sitting at the bottom, and one network plays both sides from one point of
view. Moves are mirrored the same way, which makes Black's kingside castling and
White's the *same action*.

**Actions: 73 move types from each of 64 squares.** A move is named by where it
starts and how it moves::

    action = move_type * 64 + from_square

56 of the types are queen-like - eight directions, one to seven squares - which
covers rooks, bishops, queens and kings, pawn pushes and captures, castling (the
king moving two squares) and promotion to a queen. Eight are knight jumps. The
last nine are underpromotions: to a knight, bishop or rook, moving straight or
capturing to either side. This is AlphaZero's encoding, and the type-major order
is what the convolutional policy head emits: one plane per move type, read off
at the square the piece starts from. Of the 4,672 indices only 1,858 name a move
that stays on the board; the rest are never legal, and the mask keeps the network
off them.

**Input: the position, not its history.** Twelve planes for the pieces (the
mover's first), four for castling rights, one for the en passant square, one for
the fifty-move counter and one marking a position that has occurred before - 19
in all. AlphaZero also showed the network the previous seven positions; that is
the first thing to try if move prediction stalls.

**No symmetries.** Mirroring the board left to right changes castling, and
flipping it top to bottom changes which way the pawns move. None of the free
augmentation the earlier games enjoyed applies here.

**Draws the framework must see as draws**: stalemate, insufficient material, the
fifty-move rule and threefold repetition - the last two applied automatically, as
engines do, rather than waiting for a player to claim them.

White is seat 0, the player who moves first.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

import chess
import numpy as np

SQUARES = 64

#: (file step, rank step) in the mover's frame: N, NE, E, SE, S, SW, W, NW. The
#: order is part of the action encoding, so it must not change once a network
#: has been trained.
QUEEN_DIRECTIONS: tuple[tuple[int, int], ...] = (
    (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1),
)
KNIGHT_JUMPS: tuple[tuple[int, int], ...] = (
    (1, 2), (2, 1), (2, -1), (1, -2), (-1, -2), (-2, -1), (-2, 1), (-1, 2),
)
#: Promotion to a queen is a queen-like move; these are the other three.
UNDERPROMOTIONS: tuple[int, ...] = (chess.KNIGHT, chess.BISHOP, chess.ROOK)
#: A promoting pawn captures towards the a-file, moves straight, or captures
#: towards the h-file.
PROMOTION_FILE_STEPS: tuple[int, ...] = (-1, 0, 1)

QUEEN_TYPES = len(QUEEN_DIRECTIONS) * 7
KNIGHT_TYPES = len(KNIGHT_JUMPS)
UNDERPROMOTION_TYPES = len(PROMOTION_FILE_STEPS) * len(UNDERPROMOTIONS)
MOVE_TYPES = QUEEN_TYPES + KNIGHT_TYPES + UNDERPROMOTION_TYPES
ACTIONS = MOVE_TYPES * SQUARES

PIECES: tuple[int, ...] = (
    chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING,
)
PLANES = 2 * len(PIECES) + 4 + 1 + 1 + 1


def _moves_by_action() -> list[tuple[int, int, int | None] | None]:
    """Every action index, as (from, to, underpromotion) in the mover's frame.

    ``None`` where the move would leave the board.
    """
    moves: list[tuple[int, int, int | None] | None] = [None] * ACTIONS

    def target(square: int, file_step: int, rank_step: int) -> int | None:
        file = chess.square_file(square) + file_step
        rank = chess.square_rank(square) + rank_step
        return chess.square(file, rank) if 0 <= file < 8 and 0 <= rank < 8 else None

    for square in range(SQUARES):
        for direction, (file_step, rank_step) in enumerate(QUEEN_DIRECTIONS):
            for distance in range(1, 8):
                end = target(square, file_step * distance, rank_step * distance)
                if end is not None:
                    move_type = direction * 7 + distance - 1
                    moves[move_type * SQUARES + square] = (square, end, None)
        for jump, (file_step, rank_step) in enumerate(KNIGHT_JUMPS):
            end = target(square, file_step, rank_step)
            if end is not None:
                moves[(QUEEN_TYPES + jump) * SQUARES + square] = (square, end, None)
        if chess.square_rank(square) == 6:  # the seventh rank: one step from promoting
            for step, file_step in enumerate(PROMOTION_FILE_STEPS):
                end = target(square, file_step, 1)
                if end is not None:
                    for index, piece in enumerate(UNDERPROMOTIONS):
                        move_type = QUEEN_TYPES + KNIGHT_TYPES + step * len(UNDERPROMOTIONS) + index
                        moves[move_type * SQUARES + square] = (square, end, piece)
    return moves


#: Action index -> (from, to, underpromotion) in the mover's frame, or None.
MOVES = _moves_by_action()
#: The inverse: (from, to, underpromotion) in the mover's frame -> action index.
INDEX = {move: action for action, move in enumerate(MOVES) if move is not None}


def action_of(board: chess.Board, move: chess.Move) -> int:
    """The action index of ``move``, in the frame of the player making it."""
    start, end = move.from_square, move.to_square
    if board.turn == chess.BLACK:
        start, end = chess.square_mirror(start), chess.square_mirror(end)
    promotion = move.promotion if move.promotion in UNDERPROMOTIONS else None
    return INDEX[(start, end, promotion)]


def move_of(board: chess.Board, action: int) -> chess.Move:
    """The move an action index names in ``board``. The inverse of :func:`action_of`."""
    entry = MOVES[action] if 0 <= action < ACTIONS else None
    if entry is None:
        raise ValueError(f"action {action} names no move that stays on the board")
    start, end, promotion = entry
    if board.turn == chess.BLACK:
        start, end = chess.square_mirror(start), chess.square_mirror(end)
    # The index cannot say "promote to a queen" - that is simply a queen-like
    # move - so a pawn arriving on the last rank that way promotes to a queen.
    if (promotion is None and board.piece_type_at(start) == chess.PAWN
            and chess.square_rank(end) in (0, 7)):
        promotion = chess.QUEEN
    return chess.Move(start, end, promotion)


@dataclass(frozen=True, eq=False)
class ChessState:
    """A position, and the part of its history the rules still need.

    ``board`` belongs to this state and is never modified after construction,
    which is what makes a mutable python-chess board safe to share across the
    search tree. It carries no move stack. ``history`` instead holds the position
    keys since the last irreversible move, current position last: all that
    threefold repetition needs, and never more than a hundred entries long,
    because the fifty-move rule ends the game first.
    """

    board: chess.Board
    history: tuple

    @cached_property
    def legal_moves(self) -> list[chess.Move]:
        # Generated once per position: the search asks whether the game is over,
        # which moves are legal, and then applies them, and each of those would
        # otherwise generate the moves again.
        return list(self.board.legal_moves)

    @property
    def repetitions(self) -> int:
        """How many times the current position has occurred, counting now."""
        return self.history.count(self.history[-1])


def position(fen: str = chess.STARTING_FEN) -> ChessState:
    """A state from a FEN string. Its repetition history starts here."""
    board = chess.Board(fen)
    # A private python-chess method, used deliberately: the hashable identity of
    # a position under the repetition rule (pieces, side to move, castling rights,
    # and en passant only when a capture is actually possible). The repetition
    # tests would catch a change to it.
    return ChessState(board, (board._transposition_key(),))


class Chess:
    """The rules of chess, behind the framework's :class:`~caissa.games.base.Game` contract."""

    name = "chess"
    action_size = ACTIONS
    board_shape = (8, 8)
    input_planes = PLANES

    def initial_state(self) -> ChessState:
        return position()

    def to_play(self, state: ChessState) -> int:
        return 0 if state.board.turn == chess.WHITE else 1

    def legal_actions(self, state: ChessState) -> np.ndarray:
        mask = np.zeros(ACTIONS, dtype=bool)
        for move in state.legal_moves:
            mask[action_of(state.board, move)] = True
        return mask

    def apply(self, state: ChessState, action: int) -> ChessState:
        board = state.board
        move = move_of(board, int(action))
        if move not in state.legal_moves:
            raise ValueError(f"{move.uci()} is not legal in {board.fen()}")
        child = board.copy(stack=False)
        child.push(move)
        key = child._transposition_key()
        # A capture, a pawn move or a lost castling right can never be undone, so
        # no earlier position can come round again: repetition counts afresh.
        history = (key,) if board.is_irreversible(move) else state.history + (key,)
        return ChessState(child, history)

    def terminal_value(self, state: ChessState) -> float | None:
        board = state.board
        # Checkmate first: a mate delivered on the hundredth half-move still wins.
        if not state.legal_moves:
            return -1.0 if board.is_check() else 0.0  # mated, or stalemated
        if (board.halfmove_clock >= 100 or state.repetitions >= 3
                or board.is_insufficient_material()):
            return 0.0
        return None

    def encode(self, state: ChessState) -> np.ndarray:
        board = state.board
        us = board.turn
        masks = ([board.pieces_mask(piece, us) for piece in PIECES]
                 + [board.pieces_mask(piece, not us) for piece in PIECES])
        if us == chess.BLACK:
            masks = [chess.flip_vertical(mask) for mask in masks]

        planes = np.zeros((PLANES, SQUARES), dtype=np.float32)
        # Bit k of a python-chess bitboard is square k = rank * 8 + file, which is
        # row-major on an 8x8 grid with the mover's back rank as row 0.
        bits = np.array(masks, dtype="<u8").view(np.uint8)
        planes[:12] = np.unpackbits(bits, bitorder="little").reshape(12, SQUARES)
        planes[12] = board.has_kingside_castling_rights(us)
        planes[13] = board.has_queenside_castling_rights(us)
        planes[14] = board.has_kingside_castling_rights(not us)
        planes[15] = board.has_queenside_castling_rights(not us)
        if board.has_legal_en_passant():
            square = board.ep_square
            planes[16, chess.square_mirror(square) if us == chess.BLACK else square] = 1.0
        planes[17] = min(board.halfmove_clock, 100) / 100
        planes[18] = state.repetitions > 1
        return planes.reshape(PLANES, 8, 8)

    def symmetries(
        self, encoded: np.ndarray, policy: np.ndarray
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        # The identity alone: a mirrored chess board is a different game (see the
        # module docstring). The first game in the ladder with nothing free here.
        return [(encoded, policy)]

    def render(self, state: ChessState) -> str:
        board = state.board
        side = "White" if board.turn == chess.WHITE else "Black"
        return f"{board}\n{side} to move - {board.fen()}"
