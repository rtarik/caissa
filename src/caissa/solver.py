"""A perfect Connect 4 solver: the one measurement that is not self-referential.

Everything else in this project grades the agent against something the agent
produced. The loss compares it to its own search; the arena compares it to its
own past. Both can rise while the agent goes nowhere, and both already have.

Connect 4 is solved, so for this game there is an alternative: ask what the
correct move *is*. A solver answers that from first principles, owes nothing to
the network, and cannot be talked into agreeing with a mistake.

The implementation is the standard bitboard alpha-beta. Each column occupies
seven bits - six playable rows plus a sentinel that stops shifts from wrapping
into the next column - so a whole board is one integer and alignment checks are a
handful of shifts rather than a scan.

**On cost.** Solving is exponential in the empty squares. Late positions solve in
microseconds and the opening takes minutes, so :func:`accuracy` takes a
``min_ply`` and reports how much of the game it actually covered. Measuring the
endgame honestly is worth more than claiming to measure everything.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

WIDTH, HEIGHT = 7, 6
SQUARES = WIDTH * HEIGHT
#: One bit per column beyond the playable six, so `>> 1` cannot carry a stone
#: from the top of one column into the bottom of the next.
COLUMN_BITS = HEIGHT + 1

#: Centre first. Alpha-beta prunes far more when the best move is examined early,
#: and in Connect 4 the centre column is very often best.
MOVE_ORDER = (3, 2, 4, 1, 5, 0, 6)


def bottom_mask(column: int) -> int:
    return 1 << (column * COLUMN_BITS)


def top_mask(column: int) -> int:
    return 1 << (column * COLUMN_BITS + HEIGHT - 1)


def column_mask(column: int) -> int:
    return ((1 << HEIGHT) - 1) << (column * COLUMN_BITS)


def alignment(position: int) -> bool:
    """Whether ``position`` contains four in a row, in any direction."""
    for shift in (COLUMN_BITS, COLUMN_BITS - 1, COLUMN_BITS + 1, 1):
        pair = position & (position >> shift)
        if pair & (pair >> (2 * shift)):
            return True
    return False


@dataclass(frozen=True)
class Bitboard:
    """A position as two integers.

    ``position`` holds the stones of the player to move, ``mask`` holds every
    stone. The opponent's stones are ``mask ^ position``, so both players fit in
    two words and switching sides is one exclusive-or.
    """

    position: int
    mask: int
    moves: int

    @classmethod
    def from_state(cls, state) -> Bitboard:
        """Convert a :class:`~caissa.games.connect4.Connect4State`.

        The board there is row 0 at the *top* and is in canonical perspective -
        the mover's stones are +1 - which is exactly what ``position`` wants.
        """
        position = mask = 0
        for row in range(HEIGHT):
            for column in range(WIDTH):
                value = int(state.board[HEIGHT - 1 - row, column])
                if value == 0:
                    continue
                bit = 1 << (column * COLUMN_BITS + row)
                mask |= bit
                if value == 1:
                    position |= bit
        return cls(position, mask, int(np.count_nonzero(state.board)))

    def can_play(self, column: int) -> bool:
        return (self.mask & top_mask(column)) == 0

    def play(self, column: int) -> Bitboard:
        """Drop a stone and hand the position to the opponent."""
        mask = self.mask | (self.mask + bottom_mask(column))
        # Exclusive-or with the old mask leaves the *opponent's* stones as the
        # new `position`, which is what the next player should see.
        return Bitboard(self.position ^ self.mask, mask, self.moves + 1)

    def wins_with(self, column: int) -> bool:
        """Whether playing ``column`` completes four in a row immediately."""
        landed = self.position | ((self.mask + bottom_mask(column)) & column_mask(column))
        return alignment(landed)

    @property
    def key(self) -> int:
        """A unique identifier, for the transposition table."""
        return self.position + self.mask


def _negamax(board: Bitboard, alpha: int, beta: int, table: dict) -> int:
    """Score the position for the player to move.

    Positive means they win, and larger means sooner - a win with more stones
    left on the board scores higher, so the solver prefers finishing quickly and
    resisting longest rather than treating all wins as equal.
    """
    if board.moves >= SQUARES:
        return 0  # the board filled with no line of four

    for column in MOVE_ORDER:
        if board.can_play(column) and board.wins_with(column):
            return (SQUARES + 1 - board.moves) // 2

    # Nobody can win this move, so the best conceivable outcome is a win on the
    # move after. Lowering beta to that prunes lines that cannot beat it.
    ceiling = (SQUARES - 1 - board.moves) // 2
    cached = table.get(board.key)
    if cached is not None:
        ceiling = cached
    if beta > ceiling:
        beta = ceiling
        if alpha >= beta:
            return beta

    for column in MOVE_ORDER:
        if not board.can_play(column):
            continue
        score = -_negamax(board.play(column), -beta, -alpha, table)
        if score >= beta:
            return score
        if score > alpha:
            alpha = score

    table[board.key] = alpha
    return alpha


def solve(state, table: dict | None = None) -> int:
    """Exact score of ``state`` for the player to move.

    Positive is a win, zero a draw, negative a loss; the magnitude says how many
    stones remain when the game ends, so a bigger number is a faster win.
    """
    board = Bitboard.from_state(state)
    return _negamax(board, -SQUARES // 2, SQUARES // 2, table if table is not None else {})


def outcome(score: int) -> int:
    """Reduce a score to -1, 0 or +1 - the part that decides games."""
    return (score > 0) - (score < 0)


def best_moves(game, state, table: dict | None = None) -> list[int]:
    """Every move that preserves the game-theoretic outcome.

    Not every *optimal* move: winning two stones slower is still winning, and an
    agent that does it has not made a mistake worth counting. What matters is
    whether a move throws away a win, or a draw.
    """
    table = table if table is not None else {}
    best = None
    scored = []
    for action in np.flatnonzero(game.legal_actions(state)):
        action = int(action)
        child = game.apply(state, action)
        terminal = game.terminal_value(child)
        # The child's score is from the opponent's point of view, so negate it.
        score = -terminal if terminal is not None else -solve(child, table)
        scored.append((action, score))
        best = score if best is None else max(best, score)

    assert best is not None, "no legal moves in a position that is not terminal"
    return [action for action, score in scored if outcome(score) == outcome(best)]


@dataclass
class Accuracy:
    """How often a player's move preserves the game-theoretic outcome."""

    positions: int
    correct: int
    #: Positions skipped because solving them was too expensive.
    skipped: int
    #: Positions where every legal move led to the same outcome, so no mistake
    #: was available. Excluded rather than counted: a lost position in which
    #: every move loses would otherwise be a free mark, and an agent measured in
    #: enough hopeless positions would score well for doing nothing.
    trivial: int
    #: The earliest ply included, so the coverage is never implied to be total.
    min_ply: int

    @property
    def fraction(self) -> float:
        return self.correct / self.positions if self.positions else 0.0

    def summary(self) -> str:
        text = (f"{self.correct}/{self.positions} correct ({self.fraction:.1%}) "
                f"on positions from ply {self.min_ply} onward where a mistake was "
                f"possible")
        extra = []
        if self.trivial:
            extra.append(f"{self.trivial} with no wrong answer available")
        if self.skipped:
            extra.append(f"{self.skipped} too early to solve")
        return text + (f" ({', '.join(extra)})" if extra else "")


def sample_positions(game, count: int, rng, min_ply: int = 16, max_ply: int = 34):
    """Random reachable positions within a ply window.

    Drawn at random rather than from the agent's own games, deliberately. Testing
    an agent only on positions it steers itself into flatters it: it never has to
    answer the questions it is bad at. A fixed random sample is the same exam for
    every network, which is what makes the scores comparable across generations.
    """
    positions = []
    while len(positions) < count:
        state = game.initial_state()
        target = int(rng.integers(min_ply, max_ply + 1))
        for _ in range(target):
            if game.terminal_value(state) is not None:
                break
            legal = np.flatnonzero(game.legal_actions(state))
            state = game.apply(state, int(rng.choice(legal)))
        if game.terminal_value(state) is None and state.ply >= min_ply:
            positions.append(state)
    return positions


def accuracy(game, choose, positions, min_ply: int = 16) -> Accuracy:
    """Fraction of ``positions`` where ``choose`` picks an outcome-preserving move.

    ``choose(game, state) -> int`` is anything that produces a move: a network's
    policy head, a full search, or a hand-written heuristic.

    This is the only figure in the project that owes nothing to the agent. It
    cannot rise because the network learned to agree with itself, and it cannot
    be flattered by an opponent that happens to be weak in the same places.
    """
    table: dict[int, int] = {}
    correct = skipped = trivial = scored = 0
    for state in positions:
        if state.ply < min_ply:
            skipped += 1
            continue

        good = best_moves(game, state, table)
        if len(good) == int(game.legal_actions(state).sum()):
            # Every move leads to the same outcome, so nothing here can be got
            # wrong. Scoring it would measure the sample, not the player.
            trivial += 1
            continue

        if choose(game, state) in good:
            correct += 1
        scored += 1
    return Accuracy(scored, correct, skipped, trivial, min_ply)
