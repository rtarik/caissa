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

#: Only empty points within this many squares of an existing stone are playable.
#:
#: This is domain knowledge, added deliberately, and it is worth being explicit
#: about why. AlphaZero's premise is that no such knowledge is needed - but
#: AlphaZero had roughly a million times this compute budget. Without the
#: restriction, a search over eighty-odd near-identical empty points cannot reach
#: a single terminal position: measured, twenty thousand simulations with a
#: knowledge-free evaluator return a value of exactly zero for every move. The
#: search has nothing to learn from, so the network has nothing to learn from,
#: and the loop never starts. Restricting to the neighbourhood of the stones cuts
#: the branching factor from 79 to 13 in the opening, and it converges back to
#: the full board by the midgame once the stones have spread - so it costs
#: nothing where the game is actually decided.
#:
#: It is how practical Gomoku engines have always worked. It is still a departure
#: from "zero", and it is recorded as one in PLAN.md.
NEIGHBOURHOOD = 1

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

    def to_play(self, state: GomokuState) -> int:
        # Turns strictly alternate, so the ply's parity is the seat.
        return state.ply % 2

    def legal_actions(self, state: GomokuState) -> np.ndarray:
        """Empty points near the existing stones - see :data:`NEIGHBOURHOOD`.

        Free-style otherwise: no opening handicaps of the sort some rulesets add
        to curb the first player's advantage.
        """
        empty = state.board == 0
        if not (state.board != 0).any():
            # An empty board has no neighbourhood. Opening anywhere is equivalent
            # by symmetry, so the centre point is the whole of the opening book.
            legal = np.zeros((SIZE, SIZE), dtype=bool)
            legal[SIZE // 2, SIZE // 2] = True
            return legal.ravel()

        # Dilate the occupied mask by shifting it, rather than looping over the
        # stones. Search calls this at every node, and a Python loop whose length
        # grows with the number of stones makes the whole game slower as it goes
        # on - measured at roughly twice the self-play cost by the midgame.
        occupied = state.board != 0
        near = np.zeros((SIZE, SIZE), dtype=bool)
        for d_row in range(-NEIGHBOURHOOD, NEIGHBOURHOOD + 1):
            for d_col in range(-NEIGHBOURHOOD, NEIGHBOURHOOD + 1):
                rows = slice(max(0, -d_row), min(SIZE, SIZE - d_row))
                cols = slice(max(0, -d_col), min(SIZE, SIZE - d_col))
                near[rows, cols] |= occupied[
                    slice(rows.start + d_row, rows.stop + d_row),
                    slice(cols.start + d_col, cols.stop + d_col),
                ]
        return (empty & near).ravel()

    def playable(self, state: GomokuState, action: int) -> bool:
        """Whether one action is legal, without building the whole mask.

        Search expands a node by calling :meth:`apply` once per legal move, so a
        validity check that recomputes the full board mask would do that work
        thirty times over for a single expansion. Looking at the one square and
        its neighbours is the same answer for a fraction of the cost.
        """
        row, col = divmod(action, SIZE)
        if state.board[row, col] != 0:
            return False
        if not (state.board != 0).any():
            return row == SIZE // 2 and col == SIZE // 2
        return bool(
            (
                state.board[
                    max(0, row - NEIGHBOURHOOD) : row + NEIGHBOURHOOD + 1,
                    max(0, col - NEIGHBOURHOOD) : col + NEIGHBOURHOOD + 1,
                ]
                != 0
            ).any()
        )

    def apply(self, state: GomokuState, action: int) -> GomokuState:
        row, col = divmod(action, SIZE)
        if not self.playable(state, action):
            raise ValueError(
                f"square {action} is occupied or too far from the stones"
            )

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
