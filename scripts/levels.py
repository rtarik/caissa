"""What each difficulty level on the site is actually worth, per game.

    python scripts/levels.py --games 60

The site offers one opponent per game at four thinking budgets. The budgets are
the same everywhere - 0, 40, 200 and 600 simulations a move - but what they
*buy* is not: a game with seven moves to consider gets far more out of forty
simulations than one with eighty, and a network that already plays well needs
less help than one that does not.

So each level plays the level below it, same network on both sides, and the gaps
chain into a ladder per game. Read them as within-game distances only. Elo does
not transfer between games, and the numbers here are not comparable to a human
rating on any scale.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from caissa.arena import resolvable_elo
from caissa.games import GAMES
from caissa.mcts import MCTSConfig
from caissa.network import NetworkConfig, PolicyValueNet
from caissa.parallel import ParallelArena
from caissa.selfplay import SelfPlayConfig

#: The site's four levels, weakest first, as (name, simulations a move).
LEVELS = (("Beginner", 0), ("Casual", 40), ("Strong", 200), ("Master", 600))

#: Which checkpoint each game is played from, matching what the site exports.
CHECKPOINTS = {
    "connect4": "models/connect4-latest.pt",
    "reversi": "models/reversi-latest.pt",
    "gomoku": "models/gomoku-latest.pt",
    "isola": "models/isola-latest.pt",
    "dotsandboxes": "models/dotsandboxes-latest.pt",
    "chess": "models/chess-imitation1.pt",
}


def load(path: Path, game) -> tuple[NetworkConfig, dict]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    config = NetworkConfig(**checkpoint["config"]["network"])
    net = PolicyValueNet.for_game(game, config)
    net.load_state_dict(checkpoint["network"])
    return config, {key: value.cpu() for key, value in net.state_dict().items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", type=int, default=60,
                        help="per rung; 60 resolves about 90 Elo")
    parser.add_argument("--chess-games", type=int, default=24,
                        help="chess at 600 simulations is slow enough to want its own count")
    parser.add_argument("--only", nargs="+", default=None, choices=sorted(CHECKPOINTS))
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("web/public/levels.json"))
    args = parser.parse_args()

    results: dict[str, list[dict]] = {}
    for name in args.only or CHECKPOINTS:
        path = Path(CHECKPOINTS[name])
        if not path.exists():
            print(f"{name}: no checkpoint at {path}, skipped", flush=True)
            continue

        game = GAMES[name]()
        config, state = load(path, game)
        count = args.chess_games if name == "chess" else args.games
        print(f"\n{name}: {count} games a rung "
              f"(resolves ~{resolvable_elo(count):.0f} Elo)", flush=True)

        rungs = []
        with ParallelArena(name, config, MCTSConfig(), SelfPlayConfig(),
                           workers=args.workers) as arena:
            for (lower, weak), (upper, strong) in zip(LEVELS, LEVELS[1:]):
                result = arena.match(
                    (config, state), (config, state), games=count,
                    simulations=(strong, weak), seed=args.seed,
                    names=(upper, lower),
                )
                low, high = result.interval
                rungs.append({
                    "level": upper, "below": lower,
                    "simulations": strong, "belowSimulations": weak,
                    "score": result.score, "elo": result.elo,
                    "low": None if low == float("-inf") else low,
                    "high": None if high == float("inf") else high,
                })
                print(f"  {upper:<9} ({strong:>3} sims) vs {lower:<9} "
                      f"({weak:>3}): {result.summary()}", flush=True)
        results[name] = rungs

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"levels": [
        {"name": name, "simulations": sims} for name, sims in LEVELS
    ], "games": results}, indent=1) + "\n")
    print(f"\nwritten to {args.out}")


if __name__ == "__main__":
    main()
