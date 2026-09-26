"""A rough rating for each difficulty level, measured against Stockfish.

    python scripts/stockfish.py --games 24

Every rating is a statement about a pool of players. Our levels have only ever
played each other, so their Elo differences are real but float free: nothing
says where the ladder sits. Anchoring it needs opponents whose ratings are known,
and Stockfish can be told to play at one - `UCI_LimitStrength` with a `UCI_Elo`
between 1320 and 3190.

Each level plays a grid of those strengths, both colours, and one rating per
level is fitted to all of its results at once by maximum likelihood: the rating
that makes the observed scores most probable under the Elo model. Mismatched
games are not wasted - a sweep against a weak opponent says "at least this
strong" - but the games near an even score carry most of the information.

Two honest caveats, both repeated on the site. Stockfish's `UCI_Elo` is
calibrated against computer rating lists, not against people, so this is a
rating on that scale and a FIDE figure only roughly. And the numbers are for the
settings measured here - the network, the simulation budgets, the openings - so
a change to any of them means measuring again.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
from dataclasses import dataclass
from pathlib import Path

import chess
import chess.engine
import numpy as np
import torch

from caissa.arena import Player
from caissa.games.chess import Chess, action_of
from caissa.network import NetworkConfig, NetworkEvaluator, PolicyValueNet
from caissa.rating import fit

#: The site's levels, as (name, simulations a move). Kept in step with
#: web/src/levels.ts by hand; the test for that file checks the order.
LEVELS = (("Beginner", 0), ("Casual", 40), ("Strong", 200), ("Master", 600))

#: Stockfish strengths each level plays. Wide on purpose: the fit needs games on
#: both sides of a level's rating, and where that is was not known in advance.
STRENGTHS = (1320, 1500, 1700, 1900, 2100, 2300, 2500)

#: Stockfish's thinking time per move. Its strength limiter is calibrated with
#: time to think; starved of it, the named rating stops meaning much.
MOVE_SECONDS = 0.1

#: Games longer than this are called drawn, as self-play does.
MAX_PLIES = 300


@dataclass
class Task:
    level: str
    simulations: int
    strength: int
    seconds: float
    #: True when our engine has the white pieces.
    white: bool
    seed: int


_worker: dict = {}


def _setup(checkpoint: str, stockfish: str) -> None:
    """One network and one Stockfish per worker process, reused for every game."""
    torch.set_num_threads(1)
    game = Chess()
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    net = PolicyValueNet.for_game(game, NetworkConfig(**saved["config"]["network"]))
    net.load_state_dict(saved["network"])
    engine = chess.engine.SimpleEngine.popen_uci(stockfish)
    engine.configure({"Threads": 1, "Hash": 16, "UCI_LimitStrength": True})
    _worker.update(game=game, evaluator=NetworkEvaluator(net), engine=engine)


def _play(task: Task) -> tuple[str, int, float]:
    """One game; returns (level, Stockfish strength, our score: 1, 0.5 or 0)."""
    game, engine = _worker["game"], _worker["engine"]
    engine.configure({"UCI_Elo": task.strength})
    player = Player(task.level, _worker["evaluator"], task.simulations)
    rng = np.random.default_rng(task.seed)

    state = game.initial_state()
    ours = 0 if task.white else 1
    plies = 0
    while (result := game.terminal_value(state)) is None and plies < MAX_PLIES:
        if game.to_play(state) == ours:
            action = player.choose(game, state, rng)
        else:
            move = engine.play(state.board, chess.engine.Limit(time=task.seconds)).move
            action = action_of(state.board, move)
        state = game.apply(state, action)
        plies += 1

    if result is None:
        return task.level, task.strength, 0.5
    # `result` is for the player to move in the final position.
    ours_to_move = game.to_play(state) == ours
    mover_score = (result + 1) / 2
    return task.level, task.strength, mover_score if ours_to_move else 1 - mover_score


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("models/chess-imitation1.pt"))
    parser.add_argument("--stockfish", default="stockfish")
    parser.add_argument("--games", type=int, default=24,
                        help="per level and Stockfish strength, split between colours")
    parser.add_argument("--levels", nargs="+", default=None,
                        help="only these levels, by name - for a targeted check")
    parser.add_argument("--strengths", type=int, nargs="+", default=None,
                        help="only these Stockfish strengths")
    parser.add_argument("--move-seconds", type=float, default=MOVE_SECONDS,
                        help="Stockfish's thinking time; its limiter was calibrated with more")
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("web/public/ratings.json"))
    args = parser.parse_args()

    levels = [(n, s) for n, s in LEVELS if not args.levels or n in args.levels]
    strengths = tuple(args.strengths or STRENGTHS)
    grid = [(level, simulations, strength, index)
            for level, simulations in levels
            for strength in strengths
            for index in range(args.games)]
    tasks = [Task(level, simulations, strength, args.move_seconds,
                  white=index % 2 == 0, seed=args.seed + number)
             for number, (level, simulations, strength, index) in enumerate(grid)]
    print(f"{len(tasks)} games: {len(levels)} levels x {len(strengths)} Stockfish strengths "
          f"x {args.games}, {args.move_seconds * 1000:.0f} ms a Stockfish move", flush=True)

    outcomes: dict[str, list[tuple[int, float]]] = {level: [] for level, _ in levels}
    with mp.get_context("spawn").Pool(
        args.workers, initializer=_setup,
        initargs=(str(args.checkpoint), args.stockfish),
    ) as pool:
        for done, (level, strength, score) in enumerate(
            pool.imap_unordered(_play, tasks), start=1
        ):
            outcomes[level].append((strength, score))
            if done % 24 == 0 or done == len(tasks):
                print(f"  {done}/{len(tasks)} games played", flush=True)

    rows = []
    print(f"\n{'level':<10}{'rating':>8}{'95% interval':>18}   score by Stockfish strength")
    for level, simulations in levels:
        rating, low, high = fit(outcomes[level])
        by_strength = {
            strength: np.mean([s for opp, s in outcomes[level] if opp == strength])
            for strength in strengths
        }
        shown = "  ".join(f"{strength}:{score:.0%}" for strength, score in by_strength.items())
        print(f"{level:<10}{rating:>8.0f}{f'{low:.0f}-{high:.0f}':>18}   {shown}")
        rows.append({
            "label": level, "simulations": simulations,
            "rating": round(rating), "low": round(low), "high": round(high),
            "games": len(outcomes[level]),
            "scores": {str(k): round(float(v), 3) for k, v in by_strength.items()},
        })

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "reference": "Stockfish 19, UCI_LimitStrength",
        "moveSeconds": args.move_seconds,
        "checkpoint": args.checkpoint.name,
        "levels": rows,
    }, indent=1) + "\n")
    print(f"\nwritten to {args.out}")


if __name__ == "__main__":
    main()
