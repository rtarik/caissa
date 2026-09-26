"""An opening book from our own strong games, for the site's chess engine.

    python scripts/book.py --months 2020-01

The imitation network learned from people, so its first choice in any position
is the *most popular* human move there - and it plays that choice every time.
Against 1. e4 that is 1... c5, always, and every game begins as the last one did.

A book fixes that without making the engine any weaker. For each position that
strong players reached often in the opening, it records which moves they chose
and how often. The engine samples from those, weighted by popularity: the main
line most often, the respectable alternatives some of the time, and never a move
that strong players do not play. Randomness in the network's own choices would
also vary the games, but it would do it by sometimes playing worse moves.

Positions are keyed by the first four fields of their FEN - pieces, side to
move, castling rights and a *legal* en passant square - which is exactly what
the browser's chess state uses as its repetition key, so both sides compute the
same string. Keyed by position rather than by move order, the book also covers
transpositions: however a known position is reached, it is recognised.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from caissa.data.chess import board_of, load_months, move_of


def position_keys(positions: np.ndarray) -> np.ndarray:
    """One row per position, as numbers that are equal exactly when positions are.

    The board's 32 bytes, the side to move, the castling rights and the en
    passant square are everything that distinguishes two positions for the
    purposes of the next move - the same four things a FEN's first fields hold.
    """
    board = np.ascontiguousarray(positions["board"]).view(np.uint64).reshape(len(positions), 4)
    extra = (positions["black"].astype(np.uint64)
             | positions["castling"].astype(np.uint64) << 8
             | positions["en_passant"].astype(np.uint64) << 16)
    return np.concatenate([board, extra[:, None]], axis=1)


def fen_key(stored: np.void) -> str:
    """The position's first four FEN fields, as the browser's chess state writes them."""
    return " ".join(board_of(stored).fen().split(" ")[:4])


def build(positions: np.ndarray, plies: int, min_games: int,
          min_share: float, min_count: int) -> dict[str, list[list]]:
    """Positions within ``plies`` of the start seen at least ``min_games`` times,
    each with the moves played there often enough to be worth repeating."""
    early = positions[positions["ply"] < plies]
    _, first_row, inverse, counts = np.unique(
        position_keys(early), axis=0, return_index=True, return_inverse=True, return_counts=True)

    # Every (position, move) pair counted in one sort, then walked position by
    # position: one pass over the data, however many positions the book keeps.
    pairs, played = np.unique(
        np.stack([inverse.ravel().astype(np.int64), early["move"].astype(np.int64)], axis=1),
        axis=0, return_counts=True)
    starts = np.concatenate([[0], np.flatnonzero(np.diff(pairs[:, 0])) + 1, [len(pairs)]])

    book: dict[str, list[list]] = {}
    for begin, end in zip(starts[:-1], starts[1:]):
        index = int(pairs[begin, 0])
        total = int(counts[index])
        if total < min_games:
            continue
        kept = [(int(move), int(times)) for (_, move), times in zip(pairs[begin:end], played[begin:end])
                if times >= min_count and times / total >= min_share]
        if not kept:
            continue
        stored = early[first_row[index]]
        entries = []
        for move, times in sorted(kept, key=lambda pair: -pair[1]):
            probe = stored.copy()
            probe["move"] = move
            entries.append([move_of(probe).uci(), times])
        book[fen_key(stored)] = entries
    return book


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/chess"))
    parser.add_argument("--months", nargs="+", default=["2020-01"])
    parser.add_argument("--plies", type=int, default=16,
                        help="how deep into the opening the book reaches")
    parser.add_argument("--min-games", type=int, default=200,
                        help="how often a position must have been reached to be in the book")
    parser.add_argument("--min-share", type=float, default=0.03,
                        help="the smallest share of a position's games a move needs to be kept")
    parser.add_argument("--min-count", type=int, default=20,
                        help="and the fewest times it must have been played")
    parser.add_argument("--out", type=Path, default=Path("web/public/book.json"))
    args = parser.parse_args()

    positions, _games = load_months(args.data, args.months)
    book = build(positions, args.plies, args.min_games, args.min_share, args.min_count)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "source": f"Lichess games from {', '.join(args.months)} between players rated 2200 and above",
        "plies": args.plies,
        "minGames": args.min_games,
        "minShare": args.min_share,
        "positions": book,
    }, separators=(",", ":")) + "\n")

    moves = sum(len(entries) for entries in book.values())
    print(f"{len(book):,} positions, {moves:,} moves -> {args.out} "
          f"({args.out.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
