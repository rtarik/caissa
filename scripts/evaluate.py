"""Play two checkpoints against each other.

    python scripts/evaluate.py models/connect4-gen0010.pt models/connect4-latest.pt

Games are played in colour-reversed pairs from shared random openings, so the
result measures the difference between the networks rather than the advantage of
moving first. Pass ``--simulations 0`` to compare the raw policy heads with no
search at all, which answers what the *networks* learned rather than what search
can recover on their behalf.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from caissa.arena import Player, play_match, resolvable_elo
from caissa.games import GAMES
from caissa.network import NetworkConfig, NetworkEvaluator, PolicyValueNet


def load(path: Path, game):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint["game"] != game.name:
        raise SystemExit(f"{path} is for {checkpoint['game']}, not {game.name}")
    settings = checkpoint["config"]["network"]
    net = PolicyValueNet.for_game(game, NetworkConfig(**settings))
    net.load_state_dict(checkpoint["network"])
    return net, checkpoint["iteration"]


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
    args = parser.parse_args()

    game = GAMES[args.game]()
    first, first_iteration = load(args.first, game)
    second, second_iteration = load(args.second, game)

    print(f"{args.first.name} (iteration {first_iteration}) vs "
          f"{args.second.name} (iteration {second_iteration})")
    print(f"{args.games} games at {args.simulations} simulations; "
          f"this sample resolves ~{resolvable_elo(args.games):.0f} Elo\n", flush=True)

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
