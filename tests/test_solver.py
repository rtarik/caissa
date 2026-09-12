"""Tests for the perfect solver.

Positions are chosen late in the game throughout: solving is exponential in the
empty squares, and a test suite that took minutes would stop being run.
"""

from __future__ import annotations

import numpy as np
import pytest

from caissa.games.connect4 import COLS, Connect4
from caissa.solver import (
    Bitboard,
    accuracy,
    alignment,
    best_moves,
    outcome,
    sample_positions,
    solve,
)


@pytest.fixture
def game() -> Connect4:
    return Connect4()


def position(game, *columns):
    state = game.initial_state()
    for column in columns:
        state = game.apply(state, column)
    return state


# ------------------------------------------------------------------- bitboard


def test_alignment_finds_each_direction():
    # Bits are (column * 7 + row), row 0 at the bottom.
    vertical = sum(1 << (0 * 7 + r) for r in range(4))
    horizontal = sum(1 << (c * 7 + 0) for c in range(4))
    rising = sum(1 << (c * 7 + c) for c in range(4))
    falling = sum(1 << (c * 7 + (3 - c)) for c in range(4))
    for board, label in [(vertical, "vertical"), (horizontal, "horizontal"),
                         (rising, "rising"), (falling, "falling")]:
        assert alignment(board), f"{label} four not detected"


def test_alignment_does_not_wrap_between_columns():
    """The sentinel bit exists for exactly this.

    Columns are seven bits wide for six playable rows. Without the spare bit, the
    top of one column would sit next to the bottom of the next, and three stones
    stacked in column 0 plus one at the foot of column 1 would read as a vertical
    four. Widening the column puts a permanent gap between them.
    """
    spanning = (1 << 3) | (1 << 4) | (1 << 5) | (1 << 7)  # col 0 rows 3-5, col 1 row 0
    assert not alignment(spanning)

    genuine = (1 << 2) | (1 << 3) | (1 << 4) | (1 << 5)  # col 0 rows 2-5
    assert alignment(genuine)


def test_bitboard_round_trips_a_state(game):
    state = position(game, 3, 3, 4, 2, 5)
    board = Bitboard.from_state(state)

    assert board.moves == 5
    assert bin(board.mask).count("1") == 5
    # `position` holds the mover's stones; canonically those are the +1 cells.
    assert bin(board.position).count("1") == int((state.board == 1).sum())
    assert board.position & ~board.mask == 0


def test_can_play_matches_the_game(game):
    state = position(game, *([0] * 6))  # column 0 full
    board = Bitboard.from_state(state)
    for column in range(COLS):
        assert board.can_play(column) == bool(game.legal_actions(state)[column])


def test_wins_with_detects_an_immediate_win(game):
    state = position(game, 0, 1, 0, 1, 0, 1)  # mover has three in column 0
    board = Bitboard.from_state(state)
    assert board.wins_with(0)
    assert not board.wins_with(4)


# --------------------------------------------------------------------- solving


# Verified late-game positions. Solving is exponential in the empty squares, so
# anything before about ply 16 is impractical in Python and these all sit well
# past it - a test suite that took minutes would stop being run.
UNIQUE_ANSWER = (1, 5, 3, 6, 6, 5, 0, 0, 4, 2, 5, 2, 1, 0, 2, 4, 0, 5, 2, 2, 4, 6)
WINNING = (4, 4, 6, 4, 5, 5, 1, 0, 2, 1, 6, 6, 0, 3, 5, 0, 5, 0, 3, 5, 2, 2, 1, 5,
           1, 6, 2, 2, 3)
LOSING = (3, 3, 6, 5, 5, 4, 4, 2, 6, 3, 1, 5, 1, 6, 4, 0, 0, 3, 0, 0, 3, 6, 3, 5,
          6, 5)


def test_a_win_in_one_scores_the_maximum_available(game):
    """Scores count stones left on the board, so an immediate win is the best.

    Cheap despite the early ply: the solver checks for an immediate win before
    searching anything.
    """
    state = position(game, 0, 1, 0, 1, 0, 1)
    assert solve(state) == (42 + 1 - state.ply) // 2 > 0


def test_a_winning_position_scores_positive(game):
    state = position(game, *WINNING)
    assert solve(state) > 0
    assert set(best_moves(game, state)) == {3, 4}


def test_a_losing_position_scores_negative(game):
    state = position(game, *LOSING)
    assert solve(state) < 0


def test_every_move_is_listed_when_every_move_loses(game):
    """In a lost position nothing preserves a better outcome, so nothing is wrong.

    Worth pinning down, because it is why :func:`accuracy` excludes such
    positions instead of scoring them - they would be free marks.
    """
    state = position(game, *LOSING)
    legal = [a for a in range(COLS) if game.legal_actions(state)[a]]
    assert best_moves(game, state) == legal


def test_only_one_move_can_preserve_the_outcome(game):
    state = position(game, *UNIQUE_ANSWER)
    assert best_moves(game, state) == [3]
    assert int(game.legal_actions(state).sum()) == COLS, "six wrong answers available"


def test_scores_are_symmetric_across_a_move(game):
    """After a best move the opponent must see the mirrored outcome.

    A win for one side is a loss for the other; a solver that disagreed with
    itself across a single ply would be producing meaningless numbers.
    """
    state = position(game, *UNIQUE_ANSWER)
    before = outcome(solve(state))
    after = outcome(-solve(game.apply(state, best_moves(game, state)[0])))
    assert before == after == 1


def test_a_blunder_throws_the_outcome_away(game):
    """The complement: a move outside the list must change the result."""
    state = position(game, *UNIQUE_ANSWER)
    blunder = next(a for a in range(COLS)
                   if game.legal_actions(state)[a] and a not in best_moves(game, state))
    assert outcome(-solve(game.apply(state, blunder))) < outcome(solve(state))


def test_a_drawn_full_board_scores_zero(game):
    order = [0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0,
             2, 3, 2, 3, 2, 3, 3, 2, 3, 2, 3, 2,
             4, 5, 4, 5, 4, 5, 5, 4, 5, 4, 5, 4,
             6, 6, 6, 6, 6]
    state = position(game, *order)  # one square left, and it draws
    assert solve(state) == 0


def test_outcome_reduces_to_three_values():
    assert outcome(17) == 1 and outcome(-3) == -1 and outcome(0) == 0


# -------------------------------------------------------------------- accuracy


def test_sampled_positions_respect_the_ply_window(game):
    rng = np.random.default_rng(0)
    for state in sample_positions(game, 20, rng, min_ply=18, max_ply=26):
        assert state.ply >= 18
        assert game.terminal_value(state) is None


def test_perfect_play_scores_one(game):
    """The solver graded against itself must be flawless, or the grader is wrong."""
    rng = np.random.default_rng(1)
    states = sample_positions(game, 12, rng, min_ply=24, max_ply=30)

    def perfect(g, state):
        return best_moves(g, state)[0]

    result = accuracy(game, perfect, states, min_ply=24)
    assert result.positions + result.trivial == 12
    assert result.positions > 0, "no scorable position in the sample"
    assert result.fraction == 1.0


def test_a_blundering_player_scores_below_perfect(game):
    """Picking the *worst* move must be detectably worse than picking the best."""
    rng = np.random.default_rng(2)
    states = sample_positions(game, 12, rng, min_ply=24, max_ply=30)

    def worst(g, state):
        good = set(best_moves(g, state))
        legal = [a for a in range(g.action_size) if g.legal_actions(state)[a]]
        bad = [a for a in legal if a not in good]
        return bad[0] if bad else legal[0]

    result = accuracy(game, worst, states, min_ply=24)
    assert result.positions > 0
    assert result.fraction < 1.0


def test_accuracy_reports_what_it_skipped(game):
    rng = np.random.default_rng(3)
    states = sample_positions(game, 4, rng, min_ply=24, max_ply=30)
    states.append(position(game, 3, 3))  # ply 2, far too early to solve

    result = accuracy(game, lambda g, s: best_moves(g, s)[0], states, min_ply=24)
    assert result.skipped == 1
    assert "too early to solve" in result.summary()


def test_positions_with_no_wrong_answer_are_excluded(game):
    """A lost position where every move loses is not a question, so it is not asked.

    Counting them would let an agent score well by being measured in hopeless
    positions - the benchmark would be reporting the sample, not the player.
    """
    states = [position(game, *LOSING)]
    result = accuracy(game, lambda g, s: 0, states, min_ply=20)

    assert result.trivial == 1
    assert result.positions == 0
    assert "no wrong answer available" in result.summary()
