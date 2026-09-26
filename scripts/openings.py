"""Opening names, keyed by position, for the move list and the PGN headers.

    python scripts/openings.py

Lichess publishes a list of named openings - ECO code, name, and the moves that
reach it - as a public-domain collection (lichess-org/chess-openings, CC0).
Each line is played out here and filed under the position it reaches, keyed by
the first four fields of the FEN, the same key the opening book and the
browser's chess state use.

Keyed by position rather than by move order on purpose. The same position is
often reached by different routes, and naming the moves would call a Queen's
Gambit reached via 1. c4 an English Opening forever. Naming the position gets
it right however the players got there, which is how Lichess itself does it.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
from pathlib import Path

import chess
import chess.pgn


def position_key(board: chess.Board) -> str:
    """The first four FEN fields: what makes two positions the same one."""
    return " ".join(board.fen().split(" ")[:4])


def read_lines(folder: Path) -> list[tuple[str, str, str]]:
    """(eco, name, pgn) from every TSV in the folder, in file order."""
    lines = []
    for path in sorted(folder.glob("*.tsv")):
        with path.open(newline="") as handle:
            for row in csv.DictReader(handle, delimiter="\t"):
                lines.append((row["eco"], row["name"], row["pgn"]))
    return lines


def build(lines: list[tuple[str, str, str]]) -> tuple[dict[str, list[str]], int]:
    """Position key -> [eco, name], and how many positions had more than one name.

    Where two named lines reach the same position, the longer line's name is
    kept: it is the more specific description of how the game is going.
    """
    best: dict[str, tuple[int, str, str]] = {}
    clashes = 0
    for eco, name, pgn in lines:
        game = chess.pgn.read_game(io.StringIO(pgn))
        board = game.board()
        plies = 0
        for move in game.mainline_moves():
            board.push(move)
            plies += 1
        key = position_key(board)
        if key in best:
            clashes += 1
            if plies <= best[key][0]:
                continue
        best[key] = (plies, eco, name)
    return {key: [eco, name] for key, (_, eco, name) in best.items()}, clashes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("data/openings"),
                        help="the lichess-org/chess-openings TSV files")
    parser.add_argument("--out", type=Path, default=Path("web/public/openings.json"))
    args = parser.parse_args()

    lines = read_lines(args.source)
    names, clashes = build(lines)
    args.out.write_text(json.dumps({
        "source": "lichess-org/chess-openings, public domain (CC0)",
        "positions": names,
    }, separators=(",", ":")) + "\n")
    print(f"{len(lines):,} named lines -> {len(names):,} positions "
          f"({clashes} reached by more than one line) -> {args.out} "
          f"({args.out.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
