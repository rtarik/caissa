"""Reading Lichess's monthly game dumps into stored positions.

A month is one zstd-compressed PGN file of every rated game played - 47 million
of them for January 2020. Nearly all are thrown away, so the order of work is
what makes this fast: games are split out of the decompressed stream and judged
on their headers alone, and only the few that survive have their moves parsed,
replayed and stored. The headers decide:

* **Both players 2200+** - see the decision log for why the main network learns
  from strong games rather than from the owner's level.
* **Blitz or slower.** A two-second bullet move is not a decision worth copying.
* **Finished on the board.** Time forfeits are dropped: their result was decided
  by a clock, and the value head would learn that the side ahead lost.
* **No bots.** Lichess lets engine accounts play rated games; this is about how
  people play.
* **At least 20 plies**, so abandoned games do not teach a short game's result.

Every kept game is replayed through :class:`~caissa.games.chess.Chess` itself,
so the repetition counts, en passant rights and legality of what is stored are
the framework's own rather than a second implementation's.
"""

from __future__ import annotations

import re
import zlib
from collections import Counter
from dataclasses import dataclass

import chess
import numpy as np

from caissa.data.chess import GAME, POSITION, record
from caissa.games.chess import Chess, action_of, position

#: Events kept: blitz or slower. Bullet and UltraBullet are the ones left out.
SLOW_ENOUGH = (b"Blitz", b"Rapid", b"Classical", b"Correspondence")
RESULTS = {b"1-0": 1, b"0-1": -1, b"1/2-1/2": 0}
SEPARATOR = b"\n\n[Event "

_COMMENT = re.compile(rb"\{[^}]*\}")
_MOVE_NUMBER = re.compile(rb"\d+\.(?:\.\.)?")


@dataclass(frozen=True)
class Filter:
    min_rating: int = 2200
    min_plies: int = 20
    #: One game in this many is held out for validation, chosen by its ID.
    validation_every: int = 100


def games_in(text: bytes) -> list[bytes]:
    """Split PGN text into games, each starting at its ``[Event`` header."""
    parts = text.split(SEPARATOR)
    games = [parts[0]] + [b"[Event " + part for part in parts[1:]]
    return [game for game in games if game.strip()]


def header(game: bytes, name: bytes) -> bytes | None:
    """The value of a PGN header, or None if the game does not have it."""
    tag = b"[" + name + b' "'
    start = game.find(tag)
    if start < 0:
        return None
    start += len(tag)
    return game[start:game.find(b'"', start)]


def rejection(game: bytes, rules: Filter) -> str | None:
    """Why the headers rule a game out, or None if they do not."""
    headers = game[:game.find(b"\n\n")]
    event = header(headers, b"Event") or b""
    if not any(kind in event for kind in SLOW_ENOUGH):
        return "bullet"
    if header(headers, b"Termination") != b"Normal":
        return "not finished on the board"
    if b'Title "BOT"' in headers:
        return "bot"
    try:
        ratings = int(header(headers, b"WhiteElo")), int(header(headers, b"BlackElo"))
    except (TypeError, ValueError):
        return "unrated"
    if min(ratings) < rules.min_rating:
        return "rating"
    if header(headers, b"Result") not in RESULTS:
        return "no result"
    return None


def moves_of(game: bytes) -> list[str]:
    """The moves of a game in SAN, with Lichess's clock and eval comments removed."""
    text = _COMMENT.sub(b" ", game[game.find(b"\n\n") + 2:])
    moves = []
    for token in text.split():
        if _MOVE_NUMBER.fullmatch(token) or token in RESULTS or token == b"*" or token[:1] == b"$":
            continue
        moves.append(token.rstrip(b"?!").decode())
    return moves


def time_control(game: bytes) -> tuple[int, int]:
    """(base seconds, increment seconds); correspondence has neither."""
    try:
        base, increment = (header(game, b"TimeControl") or b"").split(b"+")
        return min(int(base), 65535), min(int(increment), 255)
    except ValueError:
        return 0, 0


def convert(game: bytes, rules: Filter) -> tuple[np.ndarray, tuple] | str:
    """A kept game's positions and its row in the games table, or why it was skipped.

    Positions are numbered from zero within the game; the caller places them.
    """
    reason = rejection(game, rules)
    if reason:
        return reason
    sans = moves_of(game)
    if len(sans) < rules.min_plies:
        return "short"

    rules_of_chess = Chess()
    state = position()
    positions = np.empty(len(sans), dtype=POSITION)
    try:
        for ply, san in enumerate(sans):
            move = state.board.parse_san(san)
            positions[ply] = record(state, move, 0, ply)
            state = rules_of_chess.apply(state, action_of(state.board, move))
    except ValueError:
        return "unreadable"

    site = header(game, b"Site") or b""
    base, increment = time_control(game)
    info = (
        RESULTS[header(game, b"Result")],
        int(header(game, b"WhiteElo")),
        int(header(game, b"BlackElo")),
        base,
        increment,
        len(sans),
        0,
        int(zlib.crc32(site) % rules.validation_every == 0),
    )
    return positions, info


def convert_text(text: bytes, rules: Filter) -> tuple[np.ndarray, np.ndarray, Counter]:
    """Convert every game in a stretch of PGN text.

    Returns positions and games numbered from zero within this text, and a tally
    of what happened to each game.
    """
    tally: Counter = Counter()
    kept_positions: list[np.ndarray] = []
    kept_games: list[tuple] = []
    first = 0
    for game in games_in(text):
        tally["games"] += 1
        outcome = convert(game, rules)
        if isinstance(outcome, str):
            tally[outcome] += 1
            continue
        positions, info = outcome
        positions["game"] = len(kept_games)
        kept_positions.append(positions)
        kept_games.append(info[:6] + (first,) + info[7:])
        first += len(positions)
        tally["kept"] += 1

    positions = (np.concatenate(kept_positions) if kept_positions
                 else np.empty(0, dtype=POSITION))
    return positions, np.array(kept_games, dtype=GAME), tally


def chunks(stream, size: int = 1 << 25):
    """Read PGN from ``stream`` in pieces of roughly ``size`` bytes, cut between games."""
    carry = b""
    while True:
        block = stream.read(size)
        if not block:
            break
        text = carry + block
        cut = text.rfind(SEPARATOR)
        if cut <= 0:
            carry = text
            continue
        carry = text[cut + 2:]  # the next game starts with its [Event header
        yield text[:cut + 2]
    if carry.strip():
        yield carry
