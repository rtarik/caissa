"""Reversi: the game that tests whether the abstraction was real.

Three things here that Connect 4 never exercised.

**Passing.** A player with no legal move does not lose and the game does not
end - they pass, and play continues. That breaks the assumption every simple
board game encourages, that an empty move list means the game is over. It is
handled as an explicit 65th action, legal only when nothing else is, rather than
by silently skipping a turn. The implicit version would make ``apply`` sometimes
leave the same player to move, and the canonical sign flip - which every other
part of this codebase relies on happening exactly once per ``apply`` - would stop
being uniform.

**The full dihedral symmetry.** A square board is unchanged by four rotations and
their mirrors, so one position yields eight training examples rather than Connect
4's two. Gravity denied Connect 4 everything but the left-right mirror.

**Scoring.** The game ends when neither player can move, and the winner is
whoever holds more discs. There is no line to detect; the result is a count.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

SIZE = 8
SQUARES = SIZE * SIZE
#: Squares, then one more for the pass.
ACTIONS = SQUARES + 1
PASS = SQUARES

DIRECTIONS = (
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1), (0, 1),
    (1, -1), (1, 0), (1, 1),
)


@dataclass(frozen=True)
class ReversiState:
    """A position, in canonical perspective.

    ``board[r][c]`` is +1 for a disc belonging to the player about to move, -1
    for the opponent, 0 for empty.
    """

    board: np.ndarray
    #: Consecutive passes. Two in a row means neither player can move, which is
    #: the only way this game ends. A full board reaches it the same way, via two
    #: forced passes, so there is one termination rule rather than two that could
    #: disagree.
    passes: int
    ply: int


def captures(board: np.ndarray, row: int, col: int) -> list[tuple[int, int]]:
    """Discs the mover would capture by playing ``(row, col)``.

    Empty when the move is illegal, which makes this both the legality test and
    the move itself - one piece of logic rather than two that must agree.
    """
    if board[row, col] != 0:
        return []

    captured: list[tuple[int, int]] = []
    for d_row, d_col in DIRECTIONS:
        run: list[tuple[int, int]] = []
        r, c = row + d_row, col + d_col
        while 0 <= r < SIZE and 0 <= c < SIZE and board[r, c] == -1:
            run.append((r, c))
            r += d_row
            c += d_col
        # A run only counts when it is closed by one of the mover's own discs.
        if run and 0 <= r < SIZE and 0 <= c < SIZE and board[r, c] == 1:
            captured.extend(run)
    return captured


class Reversi:
    name = "reversi"
    action_size = ACTIONS
    board_shape = (SIZE, SIZE)
    input_planes = 2

    def initial_state(self) -> ReversiState:
        board = np.zeros((SIZE, SIZE), dtype=np.int8)
        # Black moves first and is the mover, so black is +1.
        board[3, 4] = board[4, 3] = 1
        board[3, 3] = board[4, 4] = -1
        return ReversiState(board=board, passes=0, ply=0)

    def legal_actions(self, state: ReversiState) -> np.ndarray:
        legal = np.zeros(ACTIONS, dtype=bool)
        for square in range(SQUARES):
            row, col = divmod(square, SIZE)
            if captures(state.board, row, col):
                legal[square] = True
        # Passing is legal only when nothing else is. Allowing it otherwise
        # would let the agent decline its turn, which is not the game.
        if not legal[:SQUARES].any():
            legal[PASS] = True
        return legal

    def apply(self, state: ReversiState, action: int) -> ReversiState:
        if action == PASS:
            if self.legal_actions(state)[:SQUARES].any():
                raise ValueError("cannot pass while a move is available")
            return ReversiState(board=-state.board, passes=state.passes + 1,
                                ply=state.ply + 1)

        row, col = divmod(action, SIZE)
        flipped = captures(state.board, row, col)
        if not flipped:
            raise ValueError(f"square {action} captures nothing, so it is illegal")

        board = state.board.copy()
        board[row, col] = 1
        for r, c in flipped:
            board[r, c] = 1
        # Flip, so the next player also sees their own discs as +1.
        return ReversiState(board=-board, passes=0, ply=state.ply + 1)

    def terminal_value(self, state: ReversiState) -> float | None:
        if state.passes < 2:
            return None
        mine = int((state.board == 1).sum())
        theirs = int((state.board == -1).sum())
        # Mover-relative, as everywhere: +1 if the player to move holds more.
        return float(np.sign(mine - theirs))

    def encode(self, state: ReversiState) -> np.ndarray:
        return np.stack(
            [state.board == 1, state.board == -1], axis=0
        ).astype(np.float32)

    def symmetries(
        self, encoded: np.ndarray, policy: np.ndarray
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        """The eight symmetries of a square board.

        The pass action is the wrinkle. Squares move under a rotation; passing
        does not - it means the same thing whichever way the board is turned - so
        it is held out of the permutation and put back afterwards. Rotating it
        along with the squares would quietly pair every augmented position with a
        policy whose last entry belongs to a different action.
        """
        squares = policy[:SQUARES].reshape(SIZE, SIZE)
        passing = policy[PASS]

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
                    np.concatenate([grid.ravel(), [passing]]).astype(policy.dtype),
                ))
        return variants

    def render(self, state: ReversiState) -> str:
        glyphs = {1: "x", -1: "o", 0: "."}
        legal = self.legal_actions(state)
        rows = []
        for r in range(SIZE):
            cells = []
            for c in range(SIZE):
                value = int(state.board[r, c])
                cells.append("*" if value == 0 and legal[r * SIZE + c]
                             else glyphs[value])
            rows.append(f"{r} " + " ".join(cells))
        mine = int((state.board == 1).sum())
        theirs = int((state.board == -1).sum())
        return "\n".join([
            "  " + " ".join(str(c) for c in range(SIZE)),
            *rows,
            f"x {mine}  o {theirs}" + ("  (must pass)" if legal[PASS] else ""),
        ])
