"""Tests for fitting a rating to games against rated opponents.

The fit is the one piece of arithmetic every rating on the site depends on, and
it fails in the quiet way arithmetic does: a flipped sign or a wrong constant
still produces a plausible-looking number. So it is checked against results
whose answer is known exactly from the Elo formula itself.
"""

from __future__ import annotations

import numpy as np
import pytest

from caissa.rating import expected, fit


def test_equal_ratings_expect_an_even_score():
    assert expected(1800, 1800) == pytest.approx(0.5)


def test_four_hundred_points_is_ten_to_one():
    """The defining constant of the scale: 400 points means odds of 10 to 1."""
    assert expected(1900, 1500) == pytest.approx(10 / 11)
    assert expected(1500, 1900) == pytest.approx(1 / 11)


def test_an_even_score_against_one_opponent_is_that_opponent_s_rating():
    results = [(1700, 1.0), (1700, 0.0)] * 20
    rating, low, high = fit(results)
    assert rating == pytest.approx(1700, abs=1)
    assert low < 1700 < high


def test_the_fit_recovers_the_rating_the_scores_were_made_from():
    """Scores generated from a known rating must fit back to it.

    Three opponents, each played the number of games that makes the expected
    score an exact fraction; the fit has to land on the rating that produced them.
    """
    true = 1850
    results = []
    for opponent in (1500, 1850, 2200):
        wins = round(expected(true, opponent) * 1000)
        results += [(opponent, 1.0)] * wins + [(opponent, 0.0)] * (1000 - wins)
    rating, _, _ = fit(results)
    assert rating == pytest.approx(true, abs=3)


def test_draws_count_as_half():
    """Forty draws against a 2000 opponent is an even score against a 2000 opponent."""
    rating, _, _ = fit([(2000, 0.5)] * 40)
    assert rating == pytest.approx(2000, abs=1)


def test_more_games_narrow_the_interval():
    few = fit([(1600, 1.0), (1600, 0.0)] * 5)
    many = fit([(1600, 1.0), (1600, 0.0)] * 200)
    assert (many[2] - many[1]) < (few[2] - few[1]) / 4


def test_a_clean_sweep_is_a_bound_not_a_number():
    """Winning every game says "at least this strong", and the fit must say so too."""
    rating, low, high = fit([(1320, 1.0)] * 24, highest=4000)
    assert high == 4000
    assert low > 1320


def test_nothing_to_fit_is_an_error():
    with pytest.raises(ValueError):
        fit([])


# ------------------------------------------------------------- a whole pool


from caissa.rating import fit_pool


def pool_games(truth: dict[str, float], pairs: list[tuple[str, str]], per_pair: int = 2000):
    """Games whose scores are exactly what the true ratings predict."""
    games = []
    for a, b in pairs:
        wins = round(expected(truth[a], truth[b]) * per_pair)
        games += [(a, b, 1.0)] * wins + [(a, b, 0.0)] * (per_pair - wins)
    return games


def test_a_pool_fit_recovers_the_ratings_that_made_the_scores():
    truth = {"L0": 0.0, "L1": 150.0, "L2": 420.0, "R0": 80.0, "R1": 300.0}
    # Only cross-family games: no L ever plays another L, as in the real check.
    pairs = [(l, r) for l in ("L0", "L1", "L2") for r in ("R0", "R1")]
    fitted = fit_pool(pool_games(truth, pairs), anchor="L0", prior=0.0)
    for player, rating in truth.items():
        assert fitted[player] == pytest.approx(rating, abs=4), player


def test_the_anchor_is_zero_and_only_gaps_mean_anything():
    truth = {"a": 1000.0, "b": 1200.0}
    fitted = fit_pool(pool_games(truth, [("a", "b")]), anchor="a", prior=0.0)
    assert fitted["a"] == pytest.approx(0.0)
    assert fitted["b"] == pytest.approx(200.0, abs=3)


def test_a_clean_sweep_stays_finite_with_the_prior():
    """Twenty wins and no losses has no finite maximum; the virtual draw gives one."""
    fitted = fit_pool([("a", "b", 1.0)] * 20, anchor="b")
    assert 0 < fitted["a"] < 2000


def test_the_prior_barely_moves_a_well_measured_gap():
    truth = {"a": 0.0, "b": 250.0}
    games = pool_games(truth, [("a", "b")], per_pair=4000)
    assert fit_pool(games, anchor="a", prior=0.5)["b"] == pytest.approx(250.0, abs=5)


def test_the_prior_is_exactly_half_a_virtual_draw():
    """Its size is a choice, so it is pinned to the number the choice implies.

    Twenty wins plus half a virtual draw is 20.25 points from 20.5 games, against
    0.25; the two-player fit then puts the winner 400 * log10(20.25 / 0.25) above
    the loser - about 763. A prior applied to the score but not to the games
    played, or the other way round, lands somewhere else.
    """
    fitted = fit_pool([("a", "b", 1.0)] * 20, anchor="b", prior=0.5)
    assert fitted["a"] == pytest.approx(400 * np.log10(20.25 / 0.25), abs=1)


def test_the_prior_is_exactly_virtual_draws_in_a_real_pool():
    """With two players the prior's effect on games played cancels out, so this
    is checked in a pool of three, against its definition: one virtual draw per
    pairing must fit exactly like one real draw added to every pairing."""
    games = ([("a", "b", 1.0)] * 9 + [("a", "b", 0.0)] * 3
             + [("a", "c", 1.0)] * 2 + [("a", "c", 0.0)] * 6
             + [("b", "c", 1.0)] * 5 + [("b", "c", 0.5)] * 4)
    with_prior = fit_pool(games, anchor="a", prior=1.0)
    real_draws = games + [("a", "b", 0.5), ("a", "c", 0.5), ("b", "c", 0.5)]
    explicit = fit_pool(real_draws, anchor="a", prior=0.0)
    for player in "abc":
        assert with_prior[player] == pytest.approx(explicit[player], abs=0.01), player
