"""The difficulty ladder measured against a different network, not against itself.

    python scripts/crossfamily.py --games 40

`levels.py` measured each level by playing it against the level below: the same
network, searching less. Chess showed what that does. Against Stockfish, the
step from Strong to Master is worth about 250 Elo; against its own shallower
self it looked like 374. The two sides of such a game share every blind spot,
so the deeper search knows exactly where its twin will go wrong and aims for it.
An opponent with different blind spots does not make those tailored mistakes.

This measures the ladder without that effect, for the games that kept older
networks from training. The older network plays at the same four budgets, so it
has a ladder of its own; the current network's levels play *only* those older
opponents, never each other; and every rating is fitted from all the results at
once. The gaps between the current levels then come entirely from games against
a different network. If they shrink the way chess's did, the old table was
inflated for every game, not just the one that could be checked.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch

from caissa.games import GAMES
from caissa.mcts import MCTSConfig
from caissa.network import NetworkConfig, PolicyValueNet
from caissa.parallel import ParallelArena
from caissa.rating import fit_pool
from caissa.selfplay import SelfPlayConfig

LEVELS = (("Beginner", 0), ("Casual", 40), ("Strong", 200), ("Master", 600))

#: The network the site plays, and an older one from the same training run.
#:
#: The older one has to be close. Its ladder must overlap the current one all
#: the way up, or the current Master wins every game, and a level that wins
#: every game has no rating to take a gap from - only the fit's prior. The first
#: attempt used generations from a third of the way through training, and in
#: Reversi the current Master duly won all 160 of its games against gen10. These
#: were chosen as about 200 Elo behind at equal budgets, going by the training
#: measurements in PLAN.md (Isolation's were not recorded); measured here they
#: came out between level and about 350 behind, depending on game and level.
REFERENCES = {
    "reversi": ("models/reversi-latest.pt", "models/reversi-gen0020.pt"),
    "gomoku": ("models/gomoku-latest.pt", "models/gomoku-gen0030.pt"),
    "isola": ("models/isola-latest.pt", "models/isola-gen0020.pt"),
    "dotsandboxes": ("models/dotsandboxes-latest.pt", "models/dotsandboxes-gen0040.pt"),
}


def load(path: str, game) -> tuple[NetworkConfig, dict]:
    saved = torch.load(path, map_location="cpu", weights_only=False)
    config = NetworkConfig(**saved["config"]["network"])
    net = PolicyValueNet.for_game(game, config)
    net.load_state_dict(saved["network"])
    return config, {key: value.cpu() for key, value in net.state_dict().items()}


def gaps(ratings: dict[str, float]) -> list[float]:
    """Each current level's rating over the one below it."""
    names = [f"now {name}" for name, _ in LEVELS]
    return [ratings[b] - ratings[a] for a, b in zip(names, names[1:])]


def swept(games: list[tuple[str, str, float]], player: str) -> bool:
    """Whether a player won every game, or lost every game.

    Such a player's rating is set by the fit's prior, not by anything it did, so
    a gap to or from it is not a measurement.
    """
    scores = [score for a, _, score in games if a == player]
    return all(score == 1.0 for score in scores) or all(score == 0.0 for score in scores)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", type=int, default=40,
                        help="per pairing; 40 is 2 pairs per worker on ten, as slow as 24")
    parser.add_argument("--only", nargs="+", default=None, choices=sorted(REFERENCES))
    parser.add_argument("--older", nargs="+", default=[], metavar="GAME=CHECKPOINT",
                        help="play against another generation, e.g. isola=models/isola-gen0025.pt")
    parser.add_argument("--resamples", type=int, default=200,
                        help="bootstrap resamples for the intervals")
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--levels", type=Path, default=Path("web/public/levels.json"),
                        help="the self-ladder measured by levels.py, to compare against")
    parser.add_argument("--out", type=Path, default=Path("web/public/crossfamily.json"))
    args = parser.parse_args()

    self_ladder = json.loads(args.levels.read_text())["games"] if args.levels.exists() else {}
    older_paths = dict(item.split("=", 1) for item in args.older)
    # Merged into what is already there, so one game can be measured again alone.
    report: dict[str, dict] = json.loads(args.out.read_text()) if args.out.exists() else {}
    for name in args.only or REFERENCES:
        current_path, older_path = REFERENCES[name]
        older_path = older_paths.get(name, older_path)
        game = GAMES[name]()
        current, older = load(current_path, game), load(older_path, game)
        print(f"\n{name}: current {Path(current_path).stem} against {Path(older_path).stem}, "
              f"{args.games} games a pairing", flush=True)

        games: list[tuple[str, str, float]] = []
        with ParallelArena(name, current[0], MCTSConfig(), SelfPlayConfig(),
                           workers=args.workers) as arena:
            for level, sims in LEVELS:
                for ref_level, ref_sims in LEVELS:
                    result = arena.match(current, older, games=args.games,
                                         simulations=(sims, ref_sims), seed=args.seed)
                    player, opponent = f"now {level}", f"old {ref_level}"
                    games += ([(player, opponent, 1.0)] * result.wins
                              + [(player, opponent, 0.5)] * result.draws
                              + [(player, opponent, 0.0)] * result.losses)
                    print(f"  {player:<13} vs {opponent:<13} {result.score:5.0%}", flush=True)

        fitted = fit_pool(games, anchor="now Beginner")
        # Bootstrap: refit on games resampled within each pairing, for intervals.
        rng = random.Random(args.seed)
        by_pair: dict[tuple[str, str], list] = {}
        for g in games:
            by_pair.setdefault(g[:2], []).append(g)
        samples = []
        for _ in range(args.resamples):
            resampled = [rng.choice(pair) for pair in by_pair.values() for _ in pair]
            samples.append(gaps(fit_pool(resampled, anchor="now Beginner")))
        low, high = np.percentile(np.array(samples), [2.5, 97.5], axis=0)

        measured = gaps(fitted)
        old = [rung["elo"] if rung["score"] < 1 else None for rung in self_ladder.get(name, [])]
        steps = []
        print(f"\n  {'step':<22}{'against itself':>16}{'against older':>16}{'95% range':>18}")
        for (lower, _), (upper, _), gap, lo, hi, own in zip(
            LEVELS, LEVELS[1:], measured, low, high, old or [None] * 3
        ):
            # A gap touching a level that won or lost everything is the prior talking.
            unmeasured = swept(games, f"now {lower}") or swept(games, f"now {upper}")
            own_text = f"{own:+.0f}" if own is not None else "sweep"
            ours = "swept" if unmeasured else f"{gap:+.0f}"
            spread = "" if unmeasured else f"{lo:+.0f} to {hi:+.0f}"
            print(f"  {upper + ' over ' + lower:<22}{own_text:>16}{ours:>16}{spread:>18}")
            steps.append({"level": upper, "gap": round(gap), "low": round(lo), "high": round(hi),
                          "measured": not unmeasured})
        report[name] = {
            "current": Path(current_path).stem,
            "older": Path(older_path).stem,
            "gamesPerPairing": args.games,
            "ratings": {player: round(rating) for player, rating in fitted.items()},
            "steps": steps,
        }

    args.out.write_text(json.dumps(report, indent=1) + "\n")
    print(f"\nwritten to {args.out}")


if __name__ == "__main__":
    main()
