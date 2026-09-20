"""Grade checkpoints on held-out human moves.

    python scripts/humanmoves.py models/chess-imitation1.pt models/chess-gen0020.pt

The forgetting check. Self-play trains a network on its own games, which start
out worse than the human games it was cloned from, and every number inside that
loop falls together - the policy loss drops as the network agrees with a search
that is itself guided by the network. Nothing in it can tell improvement from
drift.

This exam is fixed and outside the loop: the same held-out positions, the same
seed, for every checkpoint, so the columns are comparable. A few points of
accuracy lost while the agent gets stronger in the arena is the network trading
human style for its own; a collapse is catastrophic forgetting, and means the
self-play stage needs human positions mixed back into its buffer.

Accuracy is not skill. Human moves are not best moves, so 100% is neither
reachable nor desirable - what matters is the shape of the change.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from caissa.data.heldout import PHASES, Validation, load_months
from caissa.games.chess import Chess
from caissa.network import NetworkConfig, PolicyValueNet, best_device


def load(path: Path, game: Chess) -> PolicyValueNet:
    """Rebuild a checkpoint's network, whatever stage wrote it."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint["game"] != game.name:
        raise ValueError(f"{path} is a {checkpoint['game']} checkpoint")
    net = PolicyValueNet.for_game(game, NetworkConfig(**checkpoint["config"]["network"]))
    net.load_state_dict(checkpoint["network"])
    return net


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoints", type=Path, nargs="+")
    parser.add_argument("--data", type=Path, default=Path("data/chess"))
    parser.add_argument("--months", nargs="+", default=["2020-01"],
                        help="must include the months the exam was taken from, so the "
                             "held-out games are the same ones")
    parser.add_argument("--positions", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    positions, games = load_months(args.data, args.months)
    # The same seed as the imitation run, so this is that run's exam continued.
    exam = Validation(positions, games, args.positions, np.random.default_rng(args.seed))
    device = best_device()
    game = Chess()

    print(f"{len(exam.actions):,} held-out positions from {', '.join(args.months)}\n")
    header = ("checkpoint", "accuracy", *(name for name, _, _ in PHASES),
              "value loss", "value sign")
    print(f"{header[0]:<28}" + "".join(f"{column:>12}" for column in header[1:]))

    for path in args.checkpoints:
        result = exam.measure(load(path, game).to(device), device)
        row = (result["accuracy"], *(result[f"accuracy_{name}"] for name, _, _ in PHASES))
        print(f"{path.stem:<28}"
              + "".join(f"{value:>11.1%}" + " " for value in row)
              + f"{result['value_loss']:>11.3f} {result['value_sign']:>11.1%}")


if __name__ == "__main__":
    main()
