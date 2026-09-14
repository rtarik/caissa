"""Play two checkpoints against each other.

    python scripts/evaluate.py models/connect4-gen0010.pt models/connect4-latest.pt

Games are played in colour-reversed pairs from shared random openings, so the
result measures the difference between the networks rather than the advantage of
moving first. Pass ``--simulations 0`` to compare the raw policy heads with no
search at all, which answers what the *networks* learned rather than what search
can recover on their behalf.

Pass ``--workers 8`` to split the match across processes, whole pairs to each, the
way training's own evaluations run. A match big enough to resolve small gains is
otherwise the better part of an hour.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from caissa.arena import Player, play_match, resolvable_elo
from caissa.games import GAMES
from caissa.mcts import MCTSConfig
from caissa.network import NetworkConfig, NetworkEvaluator, PolicyValueNet
from caissa.parallel import ParallelArena
from caissa.selfplay import SelfPlayConfig


def load(path: Path, game):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint["game"] != game.name:
        raise SystemExit(f"{path} is for {checkpoint['game']}, not {game.name}")
    config = NetworkConfig(**checkpoint["config"]["network"])
    net = PolicyValueNet.for_game(game, config)
    net.load_state_dict(checkpoint["network"])
    return net, config, checkpoint["iteration"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("first", type=Path)
    parser.add_argument("second", type=Path)
    parser.add_argument("--game", default="connect4", choices=sorted(GAMES))
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--simulations", type=int, default=50,
                        help="0 plays straight from the policy head")
    parser.add_argument("--opening-plies", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=1,
                        help="processes to split the match across; 1 plays it here")
    args = parser.parse_args()

    game = GAMES[args.game]()
    first, first_config, first_iteration = load(args.first, game)
    second, second_config, second_iteration = load(args.second, game)

    print(f"{args.first.name} (iteration {first_iteration}) vs "
          f"{args.second.name} (iteration {second_iteration})")
    print(f"{args.games} games at {args.simulations} simulations; "
          f"this sample resolves ~{resolvable_elo(args.games):.0f} Elo\n", flush=True)

    if args.workers > 1:
        # The pool is also built for self-play, hence the configs; a match uses neither.
        with ParallelArena(game.name, first_config, MCTSConfig(), SelfPlayConfig(),
                           args.workers) as arena:
            result = arena.match(
                (first_config, first.state_dict()),
                (second_config, second.state_dict()),
                games=args.games, simulations=args.simulations, seed=args.seed,
                opening_plies=args.opening_plies, names=(args.first.stem, args.second.stem),
            )
    else:
        result = play_match(
            game,
            Player(args.first.stem, NetworkEvaluator(first), args.simulations),
            Player(args.second.stem, NetworkEvaluator(second), args.simulations),
            args.games,
            np.random.default_rng(args.seed),
            args.opening_plies,
        )
    print(result.summary())
    if not result.significant:
        print("\nThis result does not distinguish the two networks. Either they are "
              f"closer than ~{resolvable_elo(args.games):.0f} Elo apart, or more "
              "games are needed.")


if __name__ == "__main__":
    main()
