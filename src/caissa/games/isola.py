"""Isola: move your piece, then destroy a square.

The game that changes the shape of an action. Everything before this had a turn
that was one decision - a column, a square, a disc. Here a turn is two: where to
step, and which square to remove from the board afterwards. The player who cannot
move loses, so the whole game is a race to strand the other one.

**How a compound action is encoded** is the lesson, and it is the same answer
chess needs. The two halves are *multiplied*, not concatenated::

    action = direction * SQUARES + square_to_destroy

Eight directions times forty-nine squares is 392 actions, and the policy head
emits all of them. Chess does the same thing with 64 squares times 73 move types
for its 4672. The alternative - two separate half-moves, one for the step and one
for the demolition - would break the invariant every other part of this codebase
relies on: that ``apply`` always hands the position to the *other* player, so the
canonical sign flip happens exactly once per move.

The multiplication has a consequence that shows up in :meth:`Isola.symmetries`
and nowhere else so far: under a board rotation a compound action transforms in
*both* of its components. The destroyed square moves like any other square, and
the direction rotates with it. Getting one and not the other produces training
data that is subtly, silently wrong.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

SIZE = 7
SQUARES = SIZE * SIZE

#: The eight compass directions, in a fixed order. The order is part of the
#: action encoding, so it must not change once a network has been trained.
DIRECTIONS: tuple[tuple[int, int], ...] = (
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1), (0, 1),
    (1, -1), (1, 0), (1, 1),
)
ACTIONS = len(DIRECTIONS) * SQUARES


@dataclass(frozen=True)
class IsolaState:
    """A position, in canonical perspective.

    ``usable`` marks the squares still standing. ``mover`` and ``opponent`` are
    flat indices; ``mover`` is always the player about to move, which is what
    makes the perspective canonical without needing a colour anywhere.
    """

    usable: np.ndarray
    mover: int
    opponent: int
    ply: int


def step(square: int, direction: int) -> int | None:
    """The square one step from ``square``, or None if it leaves the board."""
    row, col = divmod(square, SIZE)
    d_row, d_col = DIRECTIONS[direction]
    row, col = row + d_row, col + d_col
    if 0 <= row < SIZE and 0 <= col < SIZE:
        return row * SIZE + col
    return None


class Isola:
    name = "isola"
    action_size = ACTIONS
    board_shape = (SIZE, SIZE)
    #: The mover's piece, the opponent's piece, and the squares still standing.
    input_planes = 3

    def initial_state(self) -> IsolaState:
        usable = np.ones(SQUARES, dtype=bool)
        # Facing each other from the middle of opposite edges.
        return IsolaState(
            usable=usable, mover=SIZE // 2, opponent=SQUARES - 1 - SIZE // 2, ply=0
        )

    def legal_actions(self, state: IsolaState) -> np.ndarray:
        legal = np.zeros(ACTIONS, dtype=bool)
        for direction in range(len(DIRECTIONS)):
            target = step(state.mover, direction)
            if target is None or not state.usable[target] or target == state.opponent:
                continue

            # Anything still standing may be destroyed except the two occupied
            # squares - including the square just vacated, which is free again.
            destroyable = state.usable.copy()
            destroyable[target] = False
            destroyable[state.opponent] = False
            legal[direction * SQUARES : (direction + 1) * SQUARES] = destroyable
        return legal

    def apply(self, state: IsolaState, action: int) -> IsolaState:
        if not self.legal_actions(state)[action]:
            direction, destroy = divmod(action, SQUARES)
            raise ValueError(
                f"cannot step {DIRECTIONS[direction]} and destroy square {destroy}"
            )

        direction, destroy = divmod(action, SQUARES)
        target = step(state.mover, direction)
        assert target is not None  # guaranteed by the legality check above

        usable = state.usable.copy()
        usable[destroy] = False

        # The mover becomes the opponent, exactly as the board flips sign in the
        # other games: one swap per apply, so perspective stays canonical.
        return IsolaState(
            usable=usable, mover=state.opponent, opponent=target, ply=state.ply + 1
        )

    def terminal_value(self, state: IsolaState) -> float | None:
        """A player with no legal action has lost. There are no draws.

        Note this is the first game here where being unable to move *is* the
        ending - Reversi taught the opposite, that an empty move list means pass.
        Which one applies is a rule of the game, and the framework has to be told.
        """
        if self.legal_actions(state).any():
            return None
        return -1.0

    def encode(self, state: IsolaState) -> np.ndarray:
        planes = np.zeros((self.input_planes, SQUARES), dtype=np.float32)
        planes[0, state.mover] = 1.0
        planes[1, state.opponent] = 1.0
        planes[2] = state.usable
        return planes.reshape(self.input_planes, SIZE, SIZE)

    def symmetries(
        self, encoded: np.ndarray, policy: np.ndarray
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        """The eight board symmetries, applied to a compound action space.

        Each variant permutes the policy twice over: the destroyed square moves
        like any square on the board, and the direction rotates with it. A
        transform that did one and not the other would pair every augmented
        position with a policy describing different moves - and with eight
        variants per position, that is eight wrong samples for every right one.
        """
        grid = policy.reshape(len(DIRECTIONS), SIZE, SIZE)

        variants: list[tuple[np.ndarray, np.ndarray]] = []
        for turns in range(4):
            for mirror in (False, True):
                board = np.rot90(encoded, turns, axes=(1, 2))
                # Rotate each direction's plane of destroy-squares...
                moved = np.rot90(grid, turns, axes=(1, 2))
                if mirror:
                    board = board[:, :, ::-1]
                    moved = moved[:, :, ::-1]
                # ...then reorder the planes, because the directions themselves
                # have rotated into one another.
                permutation = direction_permutation(turns, mirror)
                relabelled = np.empty_like(moved)
                for before, after in enumerate(permutation):
                    relabelled[after] = moved[before]

                variants.append((
                    np.ascontiguousarray(board),
                    np.ascontiguousarray(relabelled.reshape(-1)),
                ))
        return variants

    def render(self, state: IsolaState) -> str:
        rows = []
        for row in range(SIZE):
            cells = []
            for col in range(SIZE):
                square = row * SIZE + col
                if square == state.mover:
                    cells.append("X")
                elif square == state.opponent:
                    cells.append("O")
                else:
                    cells.append("." if state.usable[square] else " ")
            rows.append(f"{row} " + " ".join(cells))
        standing = int(state.usable.sum())
        return "\n".join([
            "  " + " ".join(str(c) for c in range(SIZE)),
            *rows,
            f"{standing} squares standing, {int(self.legal_actions(state).sum())} moves",
        ])


def direction_permutation(turns: int, mirror: bool) -> list[int]:
    """Where each direction index ends up under a board transform.

    ``np.rot90`` sends a cell at ``(r, c)`` to ``(N-1-c, r)``, so a displacement
    ``(dr, dc)`` becomes ``(-dc, dr)``. Mirroring the columns sends it to
    ``(dr, -dc)``. Derived rather than tabulated, so the two cannot drift apart
    if ``DIRECTIONS`` is ever reordered.
    """
    lookup = {offset: index for index, offset in enumerate(DIRECTIONS)}
    permutation = []
    for d_row, d_col in DIRECTIONS:
        for _ in range(turns % 4):
            d_row, d_col = -d_col, d_row
        if mirror:
            d_col = -d_col
        permutation.append(lookup[(d_row, d_col)])
    return permutation
