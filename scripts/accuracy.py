"""Grade a network against perfect play.

    python scripts/accuracy.py models/connect4-latest.pt --positions 200

The only measurement in this project that owes nothing to the agent. The loss
compares the network to its own search; the arena compares it to its own past.
Both can rise while nothing improves. A solver says what the correct move *is*.

Positions are drawn at random within a ply window rather than from the agent's
own games - testing an agent only where it chooses to go flatters it, and a fixed
random sample is the same exam for every network. Solving is exponential in the
empty squares, so the window has a floor; the report says what it covered.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from caissa.arena import Player
from caissa.evaluator import UniformEvaluator
from caissa.games import GAMES
from caissa.network import NetworkConfig, NetworkEvaluator, PolicyValueNet
from caissa.solver import accuracy, sample_positions


def load(path: Path, game) -> PolicyValueNet:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint["game"] != game.name:
        raise SystemExit(f"{path} is for {checkpoint['game']}, not {game.name}")
    net = PolicyValueNet.for_game(game, NetworkConfig(**checkpoint["config"]["network"]))
    net.load_state_dict(checkpoint["network"])
    return net


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--game", default="connect4", choices=sorted(GAMES))
    parser.add_argument("--positions", type=int, default=200)
    parser.add_argument("--min-ply", type=int, default=18)
    parser.add_argument("--max-ply", type=int, default=22)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    game = GAMES[args.game]()
    trained = load(args.checkpoint, game)
    torch.manual_seed(123)
    untrained = PolicyValueNet.for_game(game, trained.config)

    states = sample_positions(game, args.positions, np.random.default_rng(args.seed),
                              args.min_ply, args.max_ply)
    print(f"{args.positions} random positions, ply {args.min_ply}-{args.max_ply}\n",
          flush=True)

    rng = np.random.default_rng(args.seed)
    contenders = [
        ("random move", None),
        ("untrained net, no search", Player("u", NetworkEvaluator(untrained), 0)),
        ("trained net, no search", Player("t0", NetworkEvaluator(trained), 0)),
        ("search only (no network)", Player("s", UniformEvaluator(), 50)),
        ("trained net, 50 sims", Player("t50", NetworkEvaluator(trained), 50)),
        ("trained net, 200 sims", Player("t200", NetworkEvaluator(trained), 200)),
    ]

    for label, player in contenders:
        if player is None:
            def choose(g, state):
                return int(rng.choice(np.flatnonzero(g.legal_actions(state))))
        else:
            def choose(g, state, p=player):
                return p.choose(g, state, rng)
        result = accuracy(game, choose, states, args.min_ply)
        print(f"  {label:<26} {result.fraction:6.1%}   "
              f"({result.correct}/{result.positions})", flush=True)

    print(f"\n  scored on {result.positions} positions where a mistake was possible; "
          f"{result.trivial} were already decided either way")


if __name__ == "__main__":
    main()
