"""Tests for the scripts that measure the levels from outside their own family.

Both decide what the page may claim. `swept` decides whether a step between two
levels was measured at all, and `bounds` whether a chess rating is a rating or
only a ceiling. Get either wrong and the page prints a number the games never
gave - which is how "off the scale" and "about 0" came about.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def script(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses look their module up here
    spec.loader.exec_module(module)
    return module


crossfamily = script("crossfamily")
stockfish = script("stockfish")


def test_a_level_that_won_every_game_is_swept():
    games = [("now Master", "old Beginner", 1.0), ("now Master", "old Master", 1.0)]
    assert crossfamily.swept(games, "now Master")


def test_a_level_that_lost_every_game_is_swept_too():
    games = [("now Beginner", "old Strong", 0.0), ("now Beginner", "old Master", 0.0)]
    assert crossfamily.swept(games, "now Beginner")


def test_a_single_draw_is_enough_to_be_measured():
    games = [("now Master", "old Master", 1.0), ("now Master", "old Master", 0.5)]
    assert not crossfamily.swept(games, "now Master")


def test_a_mixed_record_is_measured():
    games = [("now Strong", "old Master", 1.0), ("now Strong", "old Master", 0.0)]
    assert not crossfamily.swept(games, "now Strong")


def test_only_the_players_own_games_count():
    # A sweeper's opponents losing says nothing about the sweep itself.
    games = [("now Master", "old Master", 1.0), ("now Strong", "old Master", 0.0)]
    assert crossfamily.swept(games, "now Master")


def test_gaps_are_each_current_level_over_the_one_below():
    ratings = {"now Beginner": 0, "now Casual": 353, "now Strong": 673, "now Master": 854,
               "old Beginner": -160, "old Master": 494}
    assert crossfamily.gaps(ratings) == [353, 320, 181]


def test_a_rating_inside_the_range_is_not_a_bound():
    assert stockfish.bounds(2213, 2374) == {}


def test_an_interval_reaching_the_floor_leaves_only_a_ceiling():
    assert stockfish.bounds(stockfish.FLOOR, 958.4) == {"below": 958}


def test_an_interval_reaching_the_top_leaves_only_a_floor():
    assert stockfish.bounds(2640.2, stockfish.CEILING) == {"above": 2640}
