"""Tests for fitting a rating to games against rated opponents.

The fit is the one piece of arithmetic every rating on the site depends on, and
it fails in the quiet way arithmetic does: a flipped sign or a wrong constant
still produces a plausible-looking number. So it is checked against results
whose answer is known exactly from the Elo formula itself.
"""

from __future__ import annotations

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
