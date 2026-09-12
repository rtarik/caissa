"""Run the self-improvement loop.

    python scripts/train.py --iterations 10 --games 30

Reports, alongside the losses, how the *raw network* scores three tactical
positions with known answers - no search involved. That distinction matters:
search can find a win in one with an untrained network, so any measurement that
includes search tells you very little about whether the network is learning. The
probe asks what the network believes on its own.

This is a stopgap. Phase 3 replaces it with a proper arena and Elo tracking.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from caissa.games import GAMES
from caissa.learn import LearnConfig, Learner
from caissa.mcts import MCTSConfig
from caissa.network import NetworkConfig, NetworkEvaluator
from caissa.selfplay import SelfPlayConfig
from caissa.train import TrainConfig

# Connect 4 positions as move sequences, each with exactly one correct answer.
# Uniqueness is not obvious by eye - an opponent three-in-a-row with both ends
# open is a double threat with *no* saving move - so it is checked at startup by
# validate_probes() rather than trusted.
PROBES = [
    ((0, 1, 0, 1, 0, 1), 0, "win in one"),
    ((0, 1, 0, 1, 0), 0, "block vertical"),
    ((2, 6, 1, 1, 4, 4, 0), 3, "block horizontal"),
]


def position(game, columns):
    state = game.initial_state()
    for column in columns:
        state = game.apply(state, column)
    return state


def validate_probes(game) -> None:
    """Refuse to score against a position whose answer is not unique."""
    for columns, answer, label in PROBES:
        state = position(game, columns)
        assert game.terminal_value(state) is None, f"{label}: already over"

        legal = [a for a in range(game.action_size) if game.legal_actions(state)[a]]
        wins = [a for a in legal if game.terminal_value(game.apply(state, a)) == -1.0]
        if wins:
            correct = wins
        else:
            correct = []
            for action in legal:
                after = game.apply(state, action)
                if game.terminal_value(after) is not None:
                    continue
                if not any(
                    game.legal_actions(after)[b]
                    and game.terminal_value(game.apply(after, b)) == -1.0
                    for b in range(game.action_size)
                ):
                    correct.append(action)
        assert correct == [answer], (
            f"{label}: expected unique answer {answer}, found {correct}"
        )


def probe(game, net) -> str:
    """What the network alone thinks, with no search to rescue it."""
    evaluator = NetworkEvaluator(net)
    parts = []
    for columns, answer, label in PROBES:
        priors, _ = evaluator.evaluate(game, position(game, columns))
        mark = "y" if int(priors.argmax()) == answer else "n"
        parts.append(f"{label}: {priors[answer]:.2f} {mark}")
    return "  |  ".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--game", default="connect4", choices=sorted(GAMES))
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--games", type=int, default=30)
    parser.add_argument("--simulations", type=int, default=50)
    parser.add_argument("--train-steps", type=int, default=150)
    parser.add_argument("--blocks", type=int, default=4)
    parser.add_argument("--channels", type=int, default=64)
    parser.add_argument("--buffer", type=int, default=60_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("models"))
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    game = GAMES[args.game]()
    validate_probes(game)
    config = LearnConfig(
        games_per_iteration=args.games,
        train_steps_per_iteration=args.train_steps,
        buffer_capacity=args.buffer,
        network=NetworkConfig(blocks=args.blocks, channels=args.channels),
        mcts=MCTSConfig(simulations=args.simulations),
        selfplay=SelfPlayConfig(),
        train=TrainConfig(),
    )
    learner = Learner(game, config, seed=args.seed)

    print(f"{args.game}: {learner.net.parameter_count():,} parameters, "
          f"{args.simulations} simulations per move")
    print(f"before       {probe(game, learner.net)}\n", flush=True)

    for _ in range(args.iterations):
        stats = learner.run_iteration()
        print(stats.summary(), flush=True)
        print(f"             {probe(game, learner.net)}", flush=True)
        learner.save(args.out / f"{args.game}-latest.pt")

    print(f"\nsaved to {args.out / f'{args.game}-latest.pt'}")


if __name__ == "__main__":
    main()
