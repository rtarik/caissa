"""Gomoku: five in a row on an open board.

What this one adds is **scale**. Connect 4 offers seven actions and Reversi about
ten at a time; Gomoku offers every empty intersection, which is eighty-one at the
start and still fifty by the middle of the game. Two things follow.

The policy target becomes *sparse*: search concentrates its visits on a handful
of squares, so the distribution the network is asked to reproduce is mostly
zeros. That is harder to learn than it sounds - a network can drive the loss down
a long way simply by predicting "almost nothing, almost everywhere".

And search gets thinner. Two hundred simulations spread over fifty moves is four
apiece, against nearly thirty when there are seven. The same budget buys much
less certainty, which is the real reason large action spaces are hard.

Played on 9x9 rather than the traditional 15x15. The rules are identical and the
difficulty it exercises is the same, but 225 actions and games running past two
hundred plies would cost several times the compute for no new lesson.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

SIZE = 9
SQUARES = SIZE * SIZE
CONNECT = 5

#: Horizontal, vertical, and the two diagonals.
DIRECTIONS = ((0, 1), (1, 0), (1, 1), (1, -1))


@dataclass(frozen=True)
class GomokuState:
    """A position, in canonical perspective: +1 is the player about to move."""

    board: np.ndarray
    #: Flat index of the previous move, so terminal checks look only around the
    #: square that changed instead of rescanning the board. Search calls this
    #: constantly, and on 81 squares the difference is worth having.
    last_move: int | None
    ply: int


class Gomoku:
    name = "gomoku"
    action_size = SQUARES
    board_shape = (SIZE, SIZE)
    input_planes = 2

    def initial_state(self) -> GomokuState:
        return GomokuState(
            board=np.zeros((SIZE, SIZE), dtype=np.int8), last_move=None, ply=0
        )

    def legal_actions(self, state: GomokuState) -> np.ndarray:
        # Every empty intersection, with no further restriction: this is
        # free-style Gomoku, without the opening handicaps some rulesets add to
        # curb the first player's advantage.
        return state.board.ravel() == 0

    def apply(self, state: GomokuState, action: int) -> GomokuState:
        row, col = divmod(action, SIZE)
        if state.board[row, col] != 0:
            raise ValueError(f"square {action} is already occupied")

        board = state.board.copy()
        board[row, col] = 1
        # Flip, so the next player also sees their own stones as +1.
        return GomokuState(board=-board, last_move=action, ply=state.ply + 1)

    def terminal_value(self, state: GomokuState) -> float | None:
        if state.last_move is None:
            return None

        # The previous mover's stones are -1 now, because apply() flipped the
        # board on the way out.
        if self._wins_through(state.board, state.last_move, player=-1):
            return -1.0  # the player to move has already lost
        if not (state.board == 0).any():
            return 0.0  # board full, nobody made five
        return None

    def _wins_through(self, board: np.ndarray, square: int, player: int) -> bool:
        """Whether ``player`` has five in a row passing through ``square``."""
        row, col = divmod(square, SIZE)
        for d_row, d_col in DIRECTIONS:
            count = 1
            for sign in (1, -1):
                r, c = row + d_row * sign, col + d_col * sign
                while 0 <= r < SIZE and 0 <= c < SIZE and board[r, c] == player:
                    count += 1
                    if count >= CONNECT:
                        return True
                    r += d_row * sign
                    c += d_col * sign
        return False

    def encode(self, state: GomokuState) -> np.ndarray:
        return np.stack(
            [state.board == 1, state.board == -1], axis=0
        ).astype(np.float32)

    def symmetries(
        self, encoded: np.ndarray, policy: np.ndarray
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        """The eight symmetries of a square board.

        Simpler than Reversi's, because every action *is* a square - there is no
        pass to hold out of the permutation. The multiplier matters more here
        than anywhere: games are long, positions are expensive, and eight
        variants of each is the difference between a usable amount of data and
        not enough.
        """
        squares = policy.reshape(SIZE, SIZE)
        variants: list[tuple[np.ndarray, np.ndarray]] = []
        for turns in range(4):
            for mirror in (False, True):
                board = np.rot90(encoded, turns, axes=(1, 2))
                grid = np.rot90(squares, turns)
                if mirror:
                    board = board[:, :, ::-1]
                    grid = grid[:, ::-1]
                variants.append((
                    np.ascontiguousarray(board),
                    np.ascontiguousarray(grid.ravel()),
                ))
        return variants

    def render(self, state: GomokuState) -> str:
        glyphs = {1: "x", -1: "o", 0: "."}
        rows = [
            f"{r} " + " ".join(glyphs[int(v)] for v in state.board[r])
            for r in range(SIZE)
        ]
        return "\n".join(["  " + " ".join(str(c) for c in range(SIZE)), *rows])
