"""Tests for the arena and the Elo arithmetic."""

from __future__ import annotations

import numpy as np
import pytest

from caissa.arena import (
    MatchResult,
    Player,
    elo_difference,
    expected_score,
    games_needed,
    play_game,
    play_match,
    random_opening,
    resolvable_elo,
)
from caissa.games.connect4 import COLS, Connect4


@pytest.fixture
def game() -> Connect4:
    return Connect4()


def position(game, *columns):
    state = game.initial_state()
    for column in columns:
        state = game.apply(state, column)
    return state


class AlwaysColumn:
    """Drops into one column whenever it can, else the lowest legal one."""

    def __init__(self, column):
        self.column = column

    def evaluate(self, game, state):
        priors = np.zeros(game.action_size, dtype=np.float32)
        legal = game.legal_actions(state)
        priors[self.column if legal[self.column] else np.argmax(legal)] = 1.0
        return priors, 0.0


def column_player(name, column):
    return Player(name, AlwaysColumn(column), simulations=0)


# ------------------------------------------------------------------- scoring


def test_first_player_win_scores_one(game):
    """Both sides stack their own column; whoever starts gets four first."""
    result = play_game(game, column_player("a", 0), column_player("b", 1),
                       game.initial_state(), np.random.default_rng(0))
    assert result == 1.0


def test_second_player_win_scores_zero(game):
    """The harder direction, and the one a sign error gets wrong.

    After 1,0,1,0,1 the side that moves on even plies holds three in column 1.
    The arena's *first* player inherits the other side, so it plays column 0
    while its opponent completes column 1 and wins.
    """
    opening = position(game, 1, 0, 1, 0, 1)
    result = play_game(game, column_player("a", 0), column_player("b", 1),
                       opening, np.random.default_rng(0))
    assert result == 0.0


def test_draw_scores_a_half(game):
    """A full board with no line of four is worth half a point to each side."""
    class FillLeftmost:
        def evaluate(self, game, state):
            priors = np.zeros(game.action_size, dtype=np.float32)
            priors[int(np.argmax(game.legal_actions(state)))] = 1.0
            return priors, 0.0

    # Reach a drawn full board by handing both players the same filling rule
    # from a position engineered to end level.
    order = [0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0,
             2, 3, 2, 3, 2, 3, 3, 2, 3, 2, 3, 2,
             4, 5, 4, 5, 4, 5, 5, 4, 5, 4, 5, 4,
             6, 6, 6, 6, 6]
    opening = position(game, *order)
    result = play_game(game, Player("a", FillLeftmost(), 0),
                       Player("b", FillLeftmost(), 0), opening,
                       np.random.default_rng(0))
    assert result == 0.5


# ------------------------------------------------------------------- pairing


def test_colour_reversed_pairs_cancel_the_first_move_advantage(game):
    """Two players who each win when they start must finish a match level.

    This is the whole point of pairing. Without it, a match measures which side
    got to move first more often - and in Connect 4, which is a first-player win
    with perfect play, that swamps any real difference between the players.
    """
    result = play_match(game, column_player("a", 0), column_player("b", 1),
                        games=10, rng=np.random.default_rng(0), opening_plies=0)
    assert result.games == 10
    assert result.score == 0.5
    assert result.wins == result.losses == 5


def test_both_games_in_a_pair_share_an_opening(game, monkeypatch):
    """One opening per pair, not one per game.

    If each game drew its own opening, the pairing would compare the two players
    on two different positions and the variance reduction it exists for would be
    lost. Counting the draws is the direct way to check: four games played as two
    pairs must consume exactly two openings.

    Watching what the players observe does not work, incidentally - in the
    reversed game a player moves *second*, so the first position it sees is one
    ply past the opening.
    """
    import caissa.arena as arena

    calls = []
    original = arena.random_opening

    def spy(g, plies, rng):
        state = original(g, plies, rng)
        calls.append(state.board.tobytes())
        return state

    monkeypatch.setattr(arena, "random_opening", spy)
    play_match(game, column_player("a", 0), column_player("b", 1), games=4,
               rng=np.random.default_rng(1), opening_plies=4)

    assert len(calls) == 2, "expected one opening per colour-reversed pair"


def test_a_match_needs_at_least_one_pair(game):
    with pytest.raises(ValueError):
        play_match(game, column_player("a", 0), column_player("b", 1), games=1,
                   rng=np.random.default_rng(0))


def test_openings_are_never_finished_games(game):
    """A match started from a decided position would score without a move."""
    rng = np.random.default_rng(0)
    for plies in (0, 2, 6, 12, 20):
        for _ in range(30):
            state = random_opening(game, plies, rng)
            assert game.terminal_value(state) is None


def test_openings_vary(game):
    rng = np.random.default_rng(0)
    boards = {random_opening(game, 4, rng).board.tobytes() for _ in range(30)}
    assert len(boards) > 5


# ----------------------------------------------------------------------- elo


def test_even_score_is_zero_elo():
    assert elo_difference(0.5) == pytest.approx(0.0)


def test_elo_is_antisymmetric():
    for score in (0.6, 0.75, 0.9, 0.99):
        assert elo_difference(score) == pytest.approx(-elo_difference(1 - score))


def test_expected_score_inverts_elo():
    for score in (0.05, 0.3, 0.5, 0.7, 0.95):
        assert expected_score(elo_difference(score)) == pytest.approx(score)


def test_certain_results_do_not_blow_up():
    """A clean sweep is unbounded in principle; it must still return a number."""
    assert np.isfinite(elo_difference(1.0))
    assert np.isfinite(elo_difference(0.0))
    assert elo_difference(1.0) > 0 > elo_difference(0.0)


def test_precision_costs_quadratically():
    """Halving the difference you want to detect quadruples the games needed.

    The reason 20-game matches settle nothing: near even, one Elo point is worth
    about 0.0014 of a point per game while a single game's score has a standard
    deviation of 0.5.

    The relationship only holds near even, where the logistic is close to linear.
    By 200 Elo the expected score is 0.76 and the curve has flattened, so the
    rule breaks down - which is why the small differences are what is checked.
    """
    for elo in (5, 10, 20, 40):
        assert games_needed(elo) == pytest.approx(games_needed(elo * 2) * 4, rel=0.05)

    assert games_needed(10) > 4000, "detecting 10 Elo is a four-figure undertaking"
    assert games_needed(200) < 20, "a crushing difference is obvious quickly"
    with pytest.raises(ValueError):
        games_needed(0)


# -------------------------------------------------------------- match results


def test_match_result_arithmetic():
    result = MatchResult("a", "b", wins=6, draws=2, losses=2)
    assert result.games == 10
    assert result.score == 0.7
    assert result.elo == pytest.approx(elo_difference(0.7))


def test_confidence_interval_brackets_the_estimate():
    result = MatchResult("a", "b", wins=60, draws=0, losses=40)
    low, high = result.interval
    assert low < result.elo < high


def test_more_games_narrow_the_interval():
    """The point estimate is the same; only the certainty differs.

    Reported together so that a 60% score over 20 games cannot be mistaken for
    the same claim as a 60% score over 2,000.
    """
    small = MatchResult("a", "b", wins=12, draws=0, losses=8)
    large = MatchResult("a", "b", wins=1200, draws=0, losses=800)
    assert small.score == large.score

    small_width = small.interval[1] - small.interval[0]
    large_width = large.interval[1] - large.interval[0]
    assert large_width < small_width / 5


def test_summary_mentions_both_players():
    text = MatchResult("new", "old", wins=6, draws=2, losses=2).summary()
    assert "new" in text and "old" in text and "70.0%" in text


def test_resolvable_elo_inverts_games_needed():
    for elo in (20, 50, 100, 200):
        assert resolvable_elo(games_needed(elo)) == pytest.approx(elo, rel=0.05)


def test_alphago_zero_gate_was_calibrated_to_its_sample():
    """400 games at 55% is 35 Elo, and 400 games resolves 34.

    Recorded as a test because it is the clearest illustration of the rule: a
    promotion threshold has to be chosen to match the number of games behind it.
    """
    assert elo_difference(0.55) == pytest.approx(35, abs=1)
    assert resolvable_elo(400) == pytest.approx(34, abs=2)


def test_a_cheap_gate_cannot_resolve_a_cheap_threshold():
    """40 games at a 55% threshold promotes on noise.

    40 games resolves only ~111 Elo, so a 55% (35 Elo) threshold is well inside
    the margin of error and will fire on results that mean nothing.
    """
    assert resolvable_elo(40) > elo_difference(0.55) * 2
    assert not MatchResult("a", "b", wins=22, draws=0, losses=18).significant


def test_significance_tracks_the_interval():
    lopsided = MatchResult("a", "b", wins=90, draws=0, losses=10)
    assert lopsided.significant
    assert "(not significant)" not in lopsided.summary()

    marginal = MatchResult("a", "b", wins=11, draws=0, losses=9)
    assert not marginal.significant
    assert "(not significant)" in marginal.summary()
