"""Tests for building the opening book and the opening names.

Both builders live in scripts, which are not a package, so they are loaded from
their files here. The things worth pinning are the ones that fail quietly: a
threshold applied to the wrong count, a key that does not match the one the
browser computes, and - for names - which name wins when two lines reach the
same position.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import chess
import numpy as np
from caissa.data.chess import POSITION, record
from caissa.games.chess import Chess, action_of

ROOT = Path(__file__).resolve().parent.parent


def script(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


book = script("book")
openings = script("openings")


def games(lines: list[list[str]]) -> np.ndarray:
    """Stored positions for games played along the given lines of UCI moves."""
    game = Chess()
    rows = []
    for number, moves in enumerate(lines):
        state = game.initial_state()
        for ply, uci in enumerate(moves):
            move = chess.Move.from_uci(uci)
            rows.append(record(state, move, number, ply))
            state = game.apply(state, action_of(state.board, move))
    return np.array(rows, dtype=POSITION)


START = " ".join(chess.Board().fen().split(" ")[:4])


def test_the_book_keeps_what_strong_players_chose_often():
    """Six games open e4, three d4, one a3: a3 is below both thresholds."""
    positions = games([["e2e4"]] * 6 + [["d2d4"]] * 3 + [["a2a3"]])
    built = book.build(positions, plies=4, min_games=5, min_share=0.15, min_count=2)
    assert built[START] == [["e2e4", 6], ["d2d4", 3]]


def test_a_move_needs_both_the_share_and_the_count():
    """d4 is a fifth of the games but only twice: kept for share, dropped for count."""
    positions = games([["e2e4"]] * 8 + [["d2d4"]] * 2)
    built = book.build(positions, plies=4, min_games=5, min_share=0.1, min_count=3)
    assert built[START] == [["e2e4", 8]]


def test_rarely_reached_positions_stay_out():
    positions = games([["e2e4", "c7c5"]] * 3 + [["d2d4", "d7d5"]] * 7)
    built = book.build(positions, plies=4, min_games=5, min_share=0.1, min_count=1)
    after_d4 = " ".join(chess.Board("rnbqkbnr/pppppppp/8/8/3P4/8/PPP1PPPP/RNBQKBNR b KQkq - 0 1")
                        .fen().split(" ")[:4])
    after_e4 = " ".join(chess.Board("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1")
                        .fen().split(" ")[:4])
    assert after_d4 in built
    assert after_e4 not in built, "reached three times, below the five-game floor"


def test_the_book_stops_at_its_depth():
    positions = games([["e2e4", "e7e5", "g1f3"]] * 10)
    shallow = book.build(positions, plies=1, min_games=5, min_share=0.1, min_count=1)
    assert list(shallow) == [START]


def test_keys_are_the_fen_s_first_four_fields_with_only_legal_en_passant():
    """The browser keys positions this way; a different string would empty the book.

    After 1. e4 no black pawn can take on e3, so no en passant square is written;
    after 1. e4 d5 2. e5 f5 White can take on f6, so f6 is.
    """
    after_e4 = games([["e2e4", "c7c5"]])[1]
    assert book.fen_key(after_e4) == "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq -"
    capturable = games([["e2e4", "d7d5", "e4e5", "f7f5", "e5f6"]])[4]
    assert book.fen_key(capturable) == "rnbqkbnr/ppp1p1pp/8/3pPp2/8/8/PPPP1PPP/RNBQKBNR w KQkq f6"


def test_a_longer_line_names_a_shared_position():
    """Two lines reaching one position: the longer, more specific, name is kept."""
    lines = [
        ("C20", "King's Pawn Game", "1. e4 e5"),
        ("C99", "A tempo lost and regained", "1. e4 e5 2. Nf3 Nf6 3. Ng1 Ng8"),
    ]
    names, clashes = openings.build(lines)
    key = " ".join(chess.Board("rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2")
                   .fen().split(" ")[:4])
    assert clashes == 1
    assert names[key] == ["C99", "A tempo lost and regained"]


def test_between_equals_the_first_name_stands():
    lines = [
        ("D30", "Queen's Gambit Declined", "1. d4 d5 2. c4 e6"),
        ("A13", "English Opening", "1. c4 e6 2. d4 d5"),
    ]
    names, clashes = openings.build(lines)
    assert clashes == 1
    assert list(names.values()) == [["D30", "Queen's Gambit Declined"]]

