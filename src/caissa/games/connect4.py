"""Connect 4: the reference game.

Chosen first because it is small enough to train in an afternoon and *solved*,
so once the agent is strong we can measure it against perfect play rather than
only against itself. An agent that improves against its own past versions while
being subtly broken is a real and common outcome; an external yardstick is the
only way to catch it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

ROWS, COLS = 6, 7
CONNECT = 4

# (dr, dc) for horizontal, vertical, and the two diagonals. Each is checked in
# both directions from the last placed piece.
_DIRECTIONS = ((0, 1), (1, 0), (1, 1), (1, -1))


@dataclass(frozen=True)
class Connect4State:
    """A position, in canonical perspective.

    ``board[r][c]`` is +1 for a piece belonging to the player about to move, -1
    for the opponent, 0 for empty. Row 0 is the top of the grid, so pieces fall
    toward row 5.
    """

    board: np.ndarray
    #: Where the previous move landed, or None at the start. Kept so that
    #: terminal checks only look around the one square that changed, instead of
    #: rescanning the grid. Search calls this constantly, so it is worth it.
    last_move: tuple[int, int] | None
    #: Half-moves played. Parity identifies whose turn it is from the outside;
    #: the canonical board deliberately does not encode that.
    ply: int


class Connect4:
    name = "connect4"
    action_size = COLS
    board_shape = (ROWS, COLS)
    input_planes = 2

    def initial_state(self) -> Connect4State:
        return Connect4State(
            board=np.zeros((ROWS, COLS), dtype=np.int8), last_move=None, ply=0
        )

    def legal_actions(self, state: Connect4State) -> np.ndarray:
        # A column accepts a piece exactly when its top cell is still empty.
        return state.board[0] == 0

    def apply(self, state: Connect4State, action: int) -> Connect4State:
        column = state.board[:, action]
        empty = np.flatnonzero(column == 0)
        if empty.size == 0:
            raise ValueError(f"column {action} is full")
        row = int(empty[-1])  # lowest empty cell, since row 0 is the top

        board = state.board.copy()
        board[row, action] = 1
        # Flip so the next player also sees their own pieces as +1.
        return Connect4State(board=-board, last_move=(row, action), ply=state.ply + 1)

    def terminal_value(self, state: Connect4State) -> float | None:
        if state.last_move is None:
            return None

        # The previous mover's pieces are -1 now, because apply() flipped the
        # board on the way out.
        if self._wins_through(state.board, state.last_move, player=-1):
            return -1.0  # the player to move has already lost
        if not np.any(state.board[0] == 0):
            return 0.0  # grid full, nobody connected four
        return None

    def _wins_through(
        self, board: np.ndarray, square: tuple[int, int], player: int
    ) -> bool:
        """Whether ``player`` has four in a row passing through ``square``."""
        row, col = square
        for dr, dc in _DIRECTIONS:
            count = 1
            for sign in (1, -1):
                r, c = row + dr * sign, col + dc * sign
                while (
                    0 <= r < ROWS and 0 <= c < COLS and board[r, c] == player
                ):
                    count += 1
                    if count >= CONNECT:
                        return True
                    r, c = r + dr * sign, c + dc * sign
        return False

    def encode(self, state: Connect4State) -> np.ndarray:
        # One plane for the mover's pieces, one for the opponent's. No plane is
        # needed to say whose turn it is: the canonical perspective means the
        # answer is always "the first plane".
        return np.stack(
            [state.board == 1, state.board == -1], axis=0
        ).astype(np.float32)

    def symmetries(
        self, encoded: np.ndarray, policy: np.ndarray
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        # Connect 4's only symmetry is the left-right mirror; gravity rules out
        # the rotations and vertical flips a square board would allow. Column i
        # becomes column COLS-1-i, so the policy reverses alongside the board.
        return [
            (encoded, policy),
            (encoded[:, :, ::-1].copy(), policy[::-1].copy()),
        ]

    def render(self, state: Connect4State) -> str:
        glyphs = {1: "x", -1: "o", 0: "."}
        rows = [" ".join(glyphs[int(v)] for v in row) for row in state.board]
        return "\n".join([*rows, " ".join(str(c) for c in range(COLS))])
