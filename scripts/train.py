"""Run the self-improvement loop.

    python scripts/train.py --iterations 40 --games 250 --workers 10

Progress is reported by playing the current network against the one from
``--eval-every`` iterations ago, with a confidence interval. That replaces the
tactical probes this script used to print, which misled twice: first by posing a
position with no correct answer, then by scoring the network on positions far
outside the distribution it actually plays.

Read the Elo column with the caveat it prints. Chaining differences between
consecutive generations assumes transitivity, and self-play agents break that
routinely - a network can beat its predecessor while losing to something older.
The cumulative figure is a progress indicator, not a rating. The absolute measure
is the solver comparison in Phase 3c.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from caissa.arena import Player, play_match, resolvable_elo
from caissa.games import GAMES
from caissa.learn import GateConfig, LearnConfig, Learner
from caissa.mcts import MCTSConfig
from caissa.network import NetworkConfig, NetworkEvaluator
from caissa.selfplay import SelfPlayConfig
from caissa.train import TrainConfig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--game", default="connect4", choices=sorted(GAMES))
    parser.add_argument("--iterations", type=int, default=40)
    parser.add_argument("--games", type=int, default=250)
    parser.add_argument("--simulations", type=int, default=50)
    parser.add_argument("--train-steps", type=int, default=500)
    parser.add_argument("--blocks", type=int, default=4)
    parser.add_argument("--channels", type=int, default=64)
    parser.add_argument("--buffer", type=int, default=120_000)
    parser.add_argument("--workers", type=int, default=10,
                        help="self-play processes; 1 runs in-process (easier to debug)")
    parser.add_argument("--device", default=None,
                        help="training device; default auto-detects (mps/cuda/cpu)")
    parser.add_argument("--eval-every", type=int, default=5,
                        help="iterations between progress matches; 0 disables")
    parser.add_argument("--eval-games", type=int, default=60)
    parser.add_argument("--gate", action="store_true",
                        help="only promote a network that beats the incumbent")
    parser.add_argument("--gate-games", type=int, default=40)
    parser.add_argument("--gate-threshold", type=float, default=0.55)
    parser.add_argument("--checkpoint-every", type=int, default=5,
                        help="iterations between kept generational checkpoints")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("models"))
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    game = GAMES[args.game]()

    config = LearnConfig(
        games_per_iteration=args.games,
        train_steps_per_iteration=args.train_steps,
        buffer_capacity=args.buffer,
        workers=args.workers,
        train_device=args.device,
        network=NetworkConfig(blocks=args.blocks, channels=args.channels),
        mcts=MCTSConfig(simulations=args.simulations),
        selfplay=SelfPlayConfig(),
        train=TrainConfig(),
        gate=GateConfig(enabled=args.gate, games=args.gate_games,
                        threshold=args.gate_threshold,
                        simulations=args.simulations),
    )

    if args.gate:
        floor = resolvable_elo(args.gate_games)
        print(f"gate: {args.gate_games} games resolves ~{floor:.0f} Elo; "
              f"threshold {args.gate_threshold:.0%} asks for "
              f"{-400 * np.log10(1 / args.gate_threshold - 1):.0f} Elo", flush=True)

    rng = np.random.default_rng(args.seed)
    anchor = None
    cumulative_elo = 0.0

    with Learner(game, config, seed=args.seed) as learner:
        print(f"{args.game}: {learner.net.parameter_count():,} parameters, "
              f"{args.simulations} simulations per move, {args.workers} workers")
        if args.eval_every:
            print(f"progress measured every {args.eval_every} iterations over "
                  f"{args.eval_games} games "
                  f"(resolves ~{resolvable_elo(args.eval_games):.0f} Elo)\n", flush=True)
        anchor = learner.cpu_net()

        for _ in range(args.iterations):
            stats = learner.run_iteration()
            print(stats.summary(), flush=True)

            if args.eval_every and stats.iteration % args.eval_every == 0:
                current = learner.cpu_net()
                result = play_match(
                    game,
                    Player(f"gen{stats.iteration}", NetworkEvaluator(current),
                           args.simulations),
                    Player("previous", NetworkEvaluator(anchor), args.simulations),
                    args.eval_games, rng,
                )
                cumulative_elo += result.elo
                print(f"         {result.summary()}   cumulative {cumulative_elo:+.0f} "
                      f"Elo (assumes transitivity)", flush=True)
                anchor = current

            if args.checkpoint_every and stats.iteration % args.checkpoint_every == 0:
                learner.save(args.out / f"{args.game}-gen{stats.iteration:04d}.pt")
            learner.save(args.out / f"{args.game}-latest.pt")

    print(f"\ncheckpoints in {args.out}/")


if __name__ == "__main__":
    main()
