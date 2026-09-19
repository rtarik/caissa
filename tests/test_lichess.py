"""Tests for reading Lichess dumps: which games are kept, and what is stored from them.

The games are built here, formatted the way Lichess's monthly files are - headers,
then every move on one line with a clock comment - from random legal moves, so
the tests exercise the real format without shipping any real data.
"""

from __future__ import annotations

import io
import zlib

import chess
import numpy as np
import pytest

from caissa.data.chess import board_of, move_of
from caissa.data.lichess import Filter, chunks, convert_text, games_in, moves_of, rejection

KEPT = {
    "Event": "Rated Blitz game", "Site": "https://lichess.org/abcdefgh",
    "White": "alice", "Black": "bob", "Result": "1-0",
    "WhiteElo": "2310", "BlackElo": "2250", "TimeControl": "180+2", "Termination": "Normal",
}


def random_line(plies: int, seed: int) -> list[str]:
    rng = np.random.default_rng(seed)
    board = chess.Board()
    sans = []
    for _ in range(plies):
        moves = list(board.legal_moves)
        move = moves[int(rng.integers(len(moves)))]
        sans.append(board.san(move))
        board.push(move)
    return sans


def lichess(sans: list[str], **headers: str) -> bytes:
    tags = KEPT | headers
    head = "\n".join(f'[{name} "{value}"]' for name, value in tags.items())
    moves = " ".join(
        f"{ply // 2 + 1}{'.' if ply % 2 == 0 else '...'} {san} {{ [%clk 0:02:5{ply % 10}] }}"
        for ply, san in enumerate(sans))
    return f"{head}\n\n{moves} {tags['Result']}\n\n".encode()


@pytest.mark.parametrize(("headers", "reason"), [
    ({}, None),
    ({"Event": "Rated Rapid game"}, None),
    ({"Event": "Rated Classical game"}, None),
    ({"Event": "Rated Correspondence game", "TimeControl": "-"}, None),
    ({"Event": "Rated Bullet game"}, "bullet"),
    ({"Event": "Rated UltraBullet game"}, "bullet"),
    ({"Termination": "Time forfeit"}, "not finished on the board"),
    ({"Termination": "Abandoned"}, "not finished on the board"),
    ({"WhiteTitle": "BOT"}, "bot"),
    ({"BlackElo": "2199"}, "rating"),
    ({"WhiteElo": "?"}, "unrated"),
    ({"Result": "*"}, "no result"),
])
def test_the_headers_decide_what_is_kept(headers, reason):
    assert rejection(lichess(random_line(30, 0), **headers), Filter()) == reason


def test_a_kept_game_is_stored_move_by_move():
    sans = random_line(40, seed=1)
    positions, games, tally = convert_text(lichess(sans, Result="0-1"), Filter())

    assert tally["kept"] == 1 and len(games) == 1
    assert games[0]["result"] == -1
    assert (games[0]["white_elo"], games[0]["black_elo"]) == (2310, 2250)
    assert (games[0]["base"], games[0]["increment"]) == (180, 2)
    assert games[0]["plies"] == len(positions) == 40

    board = chess.Board()
    for ply, (stored, san) in enumerate(zip(positions, sans)):
        assert stored["ply"] == ply and stored["game"] == 0
        assert board_of(stored).board_fen() == board.board_fen()
        assert move_of(stored) == board.parse_san(san)
        board.push_san(san)


def test_comments_evals_and_annotations_are_ignored():
    text = (b'[Event "Rated Blitz game"]\n\n1. e4?! { [%eval 0.2] [%clk 0:03:00] } '
            b'1... e5!! { [%clk 0:03:00] } $1 2. Nf3 { [%clk 0:02:58] } 1-0\n\n')
    assert moves_of(text) == ["e4", "e5", "Nf3"]


def test_short_and_unreadable_games_are_skipped():
    assert convert_text(lichess(random_line(19, 2)), Filter())[2]["short"] == 1
    assert convert_text(lichess(random_line(20, 2)), Filter())[2]["kept"] == 1
    broken = random_line(30, 3)
    broken[10] = "Qh5xh7"  # not a move in that position
    assert convert_text(lichess(broken), Filter())[2]["unreadable"] == 1


def test_games_are_numbered_in_order_around_the_rejected_ones():
    text = b"".join([
        lichess(random_line(30, 4)),
        lichess(random_line(30, 5), Event="Rated Bullet game"),
        lichess(random_line(24, 6)),
        lichess(random_line(30, 7), BlackElo="1500"),
        lichess(random_line(36, 8)),
    ])
    positions, games, tally = convert_text(text, Filter())
    assert tally["games"] == 5 and tally["kept"] == 3
    assert tally["bullet"] == 1 and tally["rating"] == 1
    assert list(games["plies"]) == [30, 24, 36]
    assert list(games["first"]) == [0, 30, 54]
    assert list(np.bincount(positions["game"])) == [30, 24, 36]


def test_validation_games_are_chosen_by_their_id():
    """The same game is held out on every run, and about one in a hundred is."""
    sites = [f"https://lichess.org/{index:08d}" for index in range(400)]
    text = b"".join(lichess(random_line(20, 9), Site=site) for site in sites)
    _, games, _ = convert_text(text, Filter())
    expected = [zlib.crc32(site.encode()) % 100 == 0 for site in sites]
    assert list(games["validation"].astype(bool)) == expected
    assert 0 < sum(expected) < 12


def test_chunks_are_cut_between_games():
    text = b"".join(lichess(random_line(20 + index, index)) for index in range(6))
    pieces = list(chunks(io.BytesIO(text), size=700))
    assert b"".join(pieces) == text
    assert sum(len(games_in(piece)) for piece in pieces) == 6
    assert all(piece.startswith(b"[Event ") for piece in pieces)
