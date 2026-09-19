"""Turn a month of Lichess games into stored training positions.

    python scripts/lichess.py 2020-01
    python scripts/lichess.py 2020-01 --limit-gb 2      # a sample, to measure first

Downloads the month into data/raw if it is not already there - resumably, and
checked against Lichess's published checksum - then streams it through zstd and
converts it across worker processes. Writes data/chess/<month>/:

    positions.npy   one row per position (caissa.data.chess.POSITION)
    games.npy       one row per game (caissa.data.chess.GAME)
    stats.json      what was read, what was kept, and why the rest was not

The raw file is kept: another rating band - the owner-level network of Phase
9.4b - can be filtered from it without downloading it again.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
import urllib.request
from collections import Counter, deque
from multiprocessing import get_context
from pathlib import Path

import numpy as np

from caissa.data.chess import GAME, POSITION, board_of, move_of
from caissa.data.lichess import Filter, chunks, convert_text

SOURCE = "https://database.lichess.org/standard/"


def download(month: str, raw: Path) -> Path:
    name = f"lichess_db_standard_rated_{month}.pgn.zst"
    path = raw / name
    raw.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(SOURCE + "sha256sums.txt") as sums:
        expected = next(line.split()[0].decode() for line in sums if name.encode() in line)
    print(f"downloading {name}", flush=True)
    subprocess.run(["curl", "-fsSL", "-C", "-", "--retry", "10", "--retry-all-errors",
                    "-o", str(path), SOURCE + name], check=True)
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1 << 24), b""):
            digest.update(block)
    if digest.hexdigest() != expected:
        raise SystemExit(f"{name}: checksum mismatch; delete it and run again")
    return path


def spot_check(positions: np.ndarray, rng: np.random.Generator, count: int = 2000) -> int:
    """How many of a random sample of stored moves are illegal in their stored board."""
    rows = rng.choice(len(positions), size=min(count, len(positions)), replace=False)
    return sum(move_of(positions[row]) not in board_of(positions[row]).legal_moves for row in rows)


def summary(games: np.ndarray, positions: np.ndarray, tally: Counter, read: int,
            seconds: float, month: str, rules: Filter) -> dict:
    elos = np.concatenate([games["white_elo"], games["black_elo"]]) if len(games) else np.zeros(1)
    held_out = games["validation"].astype(bool)
    base = games["base"].astype(int) + 40 * games["increment"].astype(int)
    return {
        "month": month,
        "filter": {"min_rating": rules.min_rating, "min_plies": rules.min_plies},
        "decompressed_gb": round(read / 1e9, 2),
        "seconds": round(seconds),
        "games_read": tally["games"],
        "games_kept": int(len(games)),
        "rejected": {k: v for k, v in tally.most_common() if k not in ("games", "kept")},
        "positions": int(len(positions)),
        "validation": {"games": int(held_out.sum()),
                       "positions": int(games["plies"][held_out].sum())},
        "results": {"white": int((games["result"] == 1).sum()),
                    "draw": int((games["result"] == 0).sum()),
                    "black": int((games["result"] == -1).sum())},
        "mean_plies": round(float(games["plies"].mean()), 1) if len(games) else 0,
        "elo": {q: int(np.percentile(elos, p)) for q, p in
                (("min", 0), ("median", 50), ("p90", 90), ("max", 100))},
        # Lichess's own categories, by estimated game length: base + 40 x increment.
        "time_controls": {"correspondence": int((games["base"] == 0).sum()),
                          "blitz": int(((base > 0) & (base < 480)).sum()),
                          "rapid": int(((base >= 480) & (base < 1500)).sum()),
                          "classical": int((base >= 1500).sum())},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("month", help="YYYY-MM")
    parser.add_argument("--min-rating", type=int, default=2200)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit-gb", type=float, default=None,
                        help="stop after this much decompressed text: a sample, to measure")
    parser.add_argument("--raw", type=Path, default=Path("data/raw"))
    parser.add_argument("--out", type=Path, default=Path("data/chess"))
    args = parser.parse_args()

    rules = Filter(min_rating=args.min_rating)
    path = args.raw / f"lichess_db_standard_rated_{args.month}.pgn.zst"
    if not path.exists():
        path = download(args.month, args.raw)
    out = args.out / (args.month + ("-sample" if args.limit_gb else ""))
    out.mkdir(parents=True, exist_ok=True)
    limit = args.limit_gb * 1e9 if args.limit_gb else None

    started = time.perf_counter()
    # --long=31 admits the largest window zstd can write, which big dumps may use.
    zstd = subprocess.Popen(["zstd", "-dc", "--long=31", str(path)], stdout=subprocess.PIPE)
    tally: Counter = Counter()
    games: list[np.ndarray] = []
    kept_games = kept_positions = read = 0
    reported = 0.0
    scratch = out / "positions.bin"

    with scratch.open("wb") as positions_file, get_context("spawn").Pool(args.workers) as pool:
        def collect(result) -> None:
            nonlocal kept_games, kept_positions
            positions, chunk_games, chunk_tally = result
            positions["game"] += kept_games
            chunk_games["first"] += kept_positions
            positions.tofile(positions_file)
            games.append(chunk_games)
            kept_games += len(chunk_games)
            kept_positions += len(positions)
            tally.update(chunk_tally)

        pending: deque = deque()
        for text in chunks(zstd.stdout):
            read += len(text)
            pending.append(pool.apply_async(convert_text, (text, rules)))
            # Bounded, so a fast reader cannot pile the whole month up in memory.
            while len(pending) >= 2 * args.workers:
                collect(pending.popleft().get())
            if read / 1e9 >= reported + 5:
                reported = read / 1e9
                print(f"{reported:6.1f} GB  {time.perf_counter() - started:5.0f} s  "
                      f"{tally['games']:>10,} games read  {kept_games:>8,} kept  "
                      f"{kept_positions:>11,} positions", flush=True)
            if limit and read >= limit:
                break
        while pending:
            collect(pending.popleft().get())
    zstd.kill()
    zstd.wait()

    positions = np.fromfile(scratch, dtype=POSITION)
    all_games = np.concatenate(games) if games else np.empty(0, dtype=GAME)
    np.save(out / "positions.npy", positions)
    np.save(out / "games.npy", all_games)
    scratch.unlink()

    stats = summary(all_games, positions, tally, read, time.perf_counter() - started,
                    args.month, rules)
    stats["illegal_in_spot_check"] = spot_check(positions, np.random.default_rng(0))
    (out / "stats.json").write_text(json.dumps(stats, indent=2) + "\n")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
