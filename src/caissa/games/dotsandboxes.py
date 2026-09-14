"""Dots and Boxes: the first game where a turn does not always pass.

Players take turns drawing a line between two adjacent dots. A line that draws
the fourth side of a box claims it - and **earns another move**. The game ends
when every line is drawn, and whoever holds more boxes wins.

That bonus move is what this rung of the ladder adds, and it reaches further
than it looks. Every earlier game kept turns strictly alternating, two of them on
purpose: Reversi makes passing an explicit action that hands the move over, and
Isolation fuses a step and a demolition into one compound action so that a turn
is always one move. Neither trick works here. A player can close box after box
in a single turn, so a "whole turn" action would be unboundedly long. The
framework has to be *told* whose turn it is instead - :meth:`DotsAndBoxes.to_play`
- and every place that flipped a value's sign once per move now flips it only
when the seat changes.

Two more things make the game distinctive.

**The outcome is a count, not a line.** The value target is still win or loss,
and that is the choice that matters for strength. A target of score *margin*
would reward running up the score over securing the win - and the famous
strategy of this game, sacrificing boxes early to take control of the long
chains at the end, is precisely a strategy that gives up margin in order to win.

**Every game lasts exactly sixty moves**, because every line gets drawn exactly
once. Nothing else on the ladder ends so predictably, or so late.

The board is the standard 5x5 boxes, which is 6x6 dots. With twenty-five boxes a
draw is impossible.

Representation
--------------
Lines are numbered horizontal first::

    horizontal(row, col) = row * 5 + col         row 0..5, col 0..4   ->  0..29
    vertical(row, col)   = 30 + row * 6 + col    row 0..4, col 0..5   -> 30..59

The *state* keeps lines and boxes as integer bitmasks. ``apply`` runs once for
every child of every node the search expands - sixty children at a time in the
opening - while the network runs once per expansion, so the rules are on the hot
path in a way they never are in a person's head. Gomoku already showed what a
slow ``apply`` costs.

The *network* sees an 11x11 lattice on which dots, lines and boxes each have a
cell: dots at (even, even), horizontal lines at (even, odd), vertical lines at
(odd, even) and box centres at (odd, odd). Every line then sits next to the boxes
it borders - the neighbourhood a convolution can actually use - and the board's
eight symmetries become ordinary rotations of an array.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

BOXES = 5
DOTS = BOXES + 1
HORIZONTAL_LINES = DOTS * BOXES
LINES = HORIZONTAL_LINES + BOXES * DOTS
SQUARES = BOXES * BOXES
LATTICE = 2 * BOXES + 1


def horizontal(row: int, col: int) -> int:
    """The line along the top of box ``(row, col)``; ``row`` runs up to BOXES."""
    return row * BOXES + col


def vertical(row: int, col: int) -> int:
    """The line down the left of box ``(row, col)``; ``col`` runs up to BOXES."""
    return HORIZONTAL_LINES + row * DOTS + col


#: The four lines around each box: top, bottom, left, right.
BOX_LINES: tuple[tuple[int, int, int, int], ...] = tuple(
    (horizontal(r, c), horizontal(r + 1, c), vertical(r, c), vertical(r, c + 1))
    for r in range(BOXES)
    for c in range(BOXES)
)


def _boxes_by_line() -> tuple[tuple[int, ...], ...]:
    adjacent: list[list[int]] = [[] for _ in range(LINES)]
    for box, lines in enumerate(BOX_LINES):
        for line in lines:
            adjacent[line].append(box)
    return tuple(tuple(boxes) for boxes in adjacent)


#: The one or two boxes each line borders.
LINE_BOXES = _boxes_by_line()
#: Each box's four lines as a bitmask, so "is it closed?" is one comparison.
BOX_MASKS = tuple(sum(1 << line for line in lines) for lines in BOX_LINES)

LINE_BITS = np.array([1 << line for line in range(LINES)], dtype=np.int64)
BOX_BITS = np.array([1 << box for box in range(SQUARES)], dtype=np.int64)


def line_cell(line: int) -> tuple[int, int]:
    """Where a line sits on the network's lattice."""
    if line < HORIZONTAL_LINES:
        row, col = divmod(line, BOXES)
        return 2 * row, 2 * col + 1
    row, col = divmod(line - HORIZONTAL_LINES, DOTS)
    return 2 * row + 1, 2 * col


def box_cell(box: int) -> tuple[int, int]:
    row, col = divmod(box, BOXES)
    return 2 * row + 1, 2 * col + 1


LINE_CELLS = np.array([r * LATTICE + c for r, c in map(line_cell, range(LINES))])
BOX_CELLS = np.array([r * LATTICE + c for r, c in map(box_cell, range(SQUARES))])

#: Where horizontal and vertical lines live on the lattice. Constant, but handed
#: to the network as two planes anyway, because telling a line's orientation from
#: the parity of its coordinates is exactly what convolutions are bad at.
HORIZONTAL_MASK = np.zeros((LATTICE, LATTICE), dtype=np.float32)
VERTICAL_MASK = np.zeros((LATTICE, LATTICE), dtype=np.float32)
for _line in range(LINES):
    (HORIZONTAL_MASK if _line < HORIZONTAL_LINES else VERTICAL_MASK)[line_cell(_line)] = 1.0

_CELL_TO_LINE = np.full(LATTICE * LATTICE, -1)
_CELL_TO_LINE[LINE_CELLS] = np.arange(LINES)

#: The eight symmetries, in the order :meth:`DotsAndBoxes.symmetries` returns them.
TRANSFORMS = tuple((turns, mirror) for turns in range(4) for mirror in (False, True))


def _symmetry_permutations() -> tuple[np.ndarray, ...]:
    """Where each line goes under each symmetry.

    Derived by transforming a lattice of *labels* with the very same ``rot90``
    and mirror calls that transform the encoded planes, so the policy and the
    board are permuted identically by construction, rather than by two pieces of
    arithmetic that have to be kept in agreement.
    """
    labels = np.arange(LATTICE * LATTICE).reshape(LATTICE, LATTICE)
    permutations = []
    for turns, mirror in TRANSFORMS:
        grid = np.rot90(labels, turns)
        if mirror:
            grid = grid[:, ::-1]
        destination = np.empty(LATTICE * LATTICE, dtype=int)
        destination[grid.ravel()] = np.arange(LATTICE * LATTICE)
        permutations.append(_CELL_TO_LINE[destination[LINE_CELLS]])
    return tuple(permutations)


SYMMETRY_PERMUTATIONS = _symmetry_permutations()


@dataclass(frozen=True)
class DotsAndBoxesState:
    """A position, in canonical perspective, as bitmasks.

    ``mine`` holds the boxes of the player to move and ``theirs`` the opponent's.
    Unlike every earlier game the two swap only when the turn passes: after a
    box-closing line the same player is still to move, so their boxes stay
    ``mine``.
    """

    #: Bit ``i`` set when line ``i`` is drawn.
    lines: int
    mine: int
    theirs: int
    #: The seat to move, 0 for whoever drew first. The ply's parity no longer says
    #: this, which is the whole reason the field exists.
    seat: int
    #: Lines drawn so far. The game ends at sixty.
    ply: int


class DotsAndBoxes:
    name = "dotsandboxes"
    action_size = LINES
    board_shape = (LATTICE, LATTICE)
    #: Lines drawn, the mover's boxes, the opponent's boxes, and the two constant
    #: orientation masks.
    input_planes = 5

    def initial_state(self) -> DotsAndBoxesState:
        return DotsAndBoxesState(lines=0, mine=0, theirs=0, seat=0, ply=0)

    def to_play(self, state: DotsAndBoxesState) -> int:
        return state.seat

    def legal_actions(self, state: DotsAndBoxesState) -> np.ndarray:
        return (state.lines & LINE_BITS) == 0

    def apply(self, state: DotsAndBoxesState, action: int) -> DotsAndBoxesState:
        action = int(action)  # a numpy integer would turn the bitmasks into numpy integers
        bit = 1 << action
        if state.lines & bit:
            raise ValueError(f"line {action} is already drawn")

        lines = state.lines | bit
        mine = state.mine
        for box in LINE_BOXES[action]:
            if (lines & BOX_MASKS[box]) == BOX_MASKS[box]:
                mine |= 1 << box

        if mine != state.mine:
            # A box closed: a bonus move. Same seat, same perspective, nothing flips.
            return DotsAndBoxesState(lines, mine, state.theirs, state.seat, state.ply + 1)
        # The turn passes, and the perspective passes with it.
        return DotsAndBoxesState(lines, state.theirs, mine, 1 - state.seat, state.ply + 1)

    def terminal_value(self, state: DotsAndBoxesState) -> float | None:
        """Win or loss for the player to move, once every line is drawn.

        Here +1 is not merely reachable but routine. The last line always closes
        a box - each box it borders already has its other three sides - so the
        player who draws it keeps the move, and the game ends on their turn.
        """
        if state.ply < LINES:
            return None
        return float(np.sign(state.mine.bit_count() - state.theirs.bit_count()))

    def score(self, state: DotsAndBoxesState) -> tuple[int, int]:
        """Boxes held, as ``(player to move, opponent)``."""
        return state.mine.bit_count(), state.theirs.bit_count()

    def drawn(self, state: DotsAndBoxesState) -> np.ndarray:
        """Which lines are drawn, as a boolean array."""
        return (state.lines & LINE_BITS) != 0

    def owners(self, state: DotsAndBoxesState) -> np.ndarray:
        """+1 for the mover's boxes, -1 for the opponent's, 0 for open ones."""
        mine = ((state.mine & BOX_BITS) != 0).astype(np.int8)
        theirs = ((state.theirs & BOX_BITS) != 0).astype(np.int8)
        return mine - theirs

    def encode(self, state: DotsAndBoxesState) -> np.ndarray:
        planes = np.zeros((self.input_planes, LATTICE, LATTICE), dtype=np.float32)
        flat = planes.reshape(self.input_planes, -1)
        flat[0, LINE_CELLS[self.drawn(state)]] = 1.0
        flat[1, BOX_CELLS[(state.mine & BOX_BITS) != 0]] = 1.0
        flat[2, BOX_CELLS[(state.theirs & BOX_BITS) != 0]] = 1.0
        planes[3] = HORIZONTAL_MASK
        planes[4] = VERTICAL_MASK
        return planes

    def symmetries(
        self, encoded: np.ndarray, policy: np.ndarray
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        """The eight symmetries of the square dot grid.

        One trap here that no earlier game had. A quarter-turn carries every
        horizontal line onto a vertical one, so rotating the whole input tensor
        leaves the two constant orientation planes *swapped* - an augmented
        position announcing horizontal lines where vertical ones stand, which
        never occurs in real play. The rotated line and box planes are already
        right; the constant planes are simply put back.
        """
        variants: list[tuple[np.ndarray, np.ndarray]] = []
        for (turns, mirror), permutation in zip(TRANSFORMS, SYMMETRY_PERMUTATIONS):
            board = np.rot90(encoded, turns, axes=(1, 2))
            if mirror:
                board = board[:, :, ::-1]
            board = board.copy()  # never write through a view of the caller's array
            board[3] = HORIZONTAL_MASK
            board[4] = VERTICAL_MASK

            moved = np.empty_like(policy)
            moved[permutation] = policy
            variants.append((board, moved))
        return variants

    def render(self, state: DotsAndBoxesState) -> str:
        drawn = self.drawn(state)
        owners = self.owners(state)
        marks = {1: "x", -1: "o", 0: " "}
        rows = []
        for row in range(DOTS):
            rows.append("•" + "•".join(
                "───" if drawn[horizontal(row, col)] else "   " for col in range(BOXES)
            ) + "•")
            if row < BOXES:
                cells = [
                    ("│" if drawn[vertical(row, col)] else " ")
                    + f" {marks[int(owners[row * BOXES + col])]} "
                    for col in range(BOXES)
                ]
                cells.append("│" if drawn[vertical(row, BOXES)] else " ")
                rows.append("".join(cells))
        mine, theirs = self.score(state)
        rows.append(f"x {mine}  o {theirs}   seat {state.seat} to move, "
                    f"{LINES - state.ply} lines left")
        return "\n".join(rows)
