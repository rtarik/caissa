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
import math
from pathlib import Path

import numpy as np
import torch

from caissa.arena import resolvable_elo
from caissa.games import GAMES
from caissa.learn import GateConfig, LearnConfig, Learner
from caissa.mcts import MCTSConfig
from caissa.network import NetworkConfig
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
    parser.add_argument("--policy-head", default="dense", choices=("dense", "conv"),
                        help="conv reads the action off the board; needs the action "
                             "space to be a multiple of the squares (Isola, chess)")
    parser.add_argument("--buffer", type=int, default=120_000)
    parser.add_argument("--min-buffer", type=int, default=4_000,
                        help="positions to collect before training starts; with --resume, "
                             "set it high to refill the buffer before the network moves")
    parser.add_argument("--random-openings", type=float, default=0.0,
                        help="share of self-play games that start from a random position, "
                             "reaching positions the agent's own play never would")
    parser.add_argument("--random-opening-plies", type=int, default=16,
                        help="longest random opening; each draws its length from 1 to this")
    parser.add_argument("--temperature-moves", type=int, default=8,
                        help="plies played at temperature 1 before greedy play; the "
                             "opening variety the buffer lives on (AlphaZero: 30 for chess)")
    parser.add_argument("--max-plies", type=int, default=None,
                        help="call a self-play game drawn after this many plies")
    parser.add_argument("--resign-below", type=float, default=None,
                        help="resign when search values the position below this twice running")
    parser.add_argument("--learning-rate", type=float, default=2e-3,
                        help="lower it to fine-tune a network that already knows something")
    parser.add_argument("--human", type=Path, default=None,
                        help="stored human positions (chess: data/chess) to rehearse "
                             "alongside self-play, so the value head keeps its calibration")
    parser.add_argument("--human-months", nargs="+", default=["2020-01"])
    parser.add_argument("--human-positions", type=int, default=100_000)
    parser.add_argument("--human-share", type=float, default=0.5,
                        help="share of every batch drawn from them; 0 is AlphaZero's")
    parser.add_argument("--save-buffer", action="store_true",
                        help="write the replay window beside each kept checkpoint, so the "
                             "next stage continues with it rather than an empty one")
    parser.add_argument("--workers", type=int, default=10,
                        help="self-play processes; 1 runs in-process (easier to debug)")
    parser.add_argument("--device", default=None,
                        help="training device; default auto-detects (mps/cuda/cpu)")
    parser.add_argument("--eval-every", type=int, default=5,
                        help="iterations between progress matches; 0 disables")
    parser.add_argument("--eval-games", type=int, default=100,
                        help="a match this size resolves ~69 Elo; smaller "
                             "samples cannot see a typical per-step gain")
    parser.add_argument("--gate", action="store_true",
                        help="only promote a network that beats the incumbent")
    parser.add_argument("--gate-games", type=int, default=40)
    parser.add_argument("--gate-threshold", type=float, default=0.55)
    parser.add_argument("--checkpoint-every", type=int, default=5,
                        help="iterations between kept generational checkpoints")
    parser.add_argument("--resume", type=Path, default=None,
                        help="continue from a checkpoint instead of starting over")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("models"))
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    game = GAMES[args.game]()

    config = LearnConfig(
        games_per_iteration=args.games,
        train_steps_per_iteration=args.train_steps,
        buffer_capacity=args.buffer,
        min_buffer_before_training=args.min_buffer,
        workers=args.workers,
        train_device=args.device,
        network=NetworkConfig(blocks=args.blocks, channels=args.channels,
                              policy_head=args.policy_head),
        mcts=MCTSConfig(simulations=args.simulations),
        selfplay=SelfPlayConfig(temperature_moves=args.temperature_moves,
                                random_opening_share=args.random_openings,
                                random_opening_plies=args.random_opening_plies,
                                max_plies=args.max_plies,
                                resign_below=args.resign_below),
        human_share=args.human_share if args.human else 0.0,
        train=TrainConfig(learning_rate=args.learning_rate),
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

    rehearsal = None
    if args.human:
        # Imported here: everything else in this script is game-agnostic, and
        # what a "human example" is happens to be chess-specific.
        from caissa.data.chess import human_samples, load_months

        positions, games = load_months(args.human, args.human_months)
        rehearsal = human_samples(positions, games, args.human_positions, rng)
        print(f"rehearsing {len(rehearsal):,} human positions from "
              f"{', '.join(args.human_months)}: {args.human_share:.0%} of every batch",
              flush=True)

    with Learner(game, config, seed=args.seed, human=rehearsal) as learner:
        if args.resume:
            # Restores the optimiser's moment estimates alongside the weights.
            # Resuming without them restarts AdamW cold, which shows up as a
            # visible stumble in training immediately after every resume.
            learner.load(args.resume)
            window = (f"its replay window came too ({len(learner.buffer):,} positions)"
                      if len(learner.buffer)
                      else f"no replay window beside it, so training waits until "
                           f"{args.min_buffer:,} positions have been played")
            print(f"resumed from {args.resume} at iteration {learner.iteration}; {window}",
                  flush=True)

        print(f"{args.game}: {learner.net.parameter_count():,} parameters, "
              f"{args.simulations} simulations per move, {args.workers} workers")
        if args.random_openings:
            print(f"{args.random_openings:.0%} of games start from a random opening of "
                  f"1-{args.random_opening_plies} plies", flush=True)
        if args.eval_every:
            print(f"progress measured every {args.eval_every} iterations over "
                  f"{args.eval_games} games "
                  f"(resolves ~{resolvable_elo(args.eval_games):.0f} Elo)\n", flush=True)
        anchor = learner.cpu_net()

        remaining = max(0, args.iterations - learner.iteration)
        for _ in range(remaining):
            stats = learner.run_iteration()
            print(stats.summary(), flush=True)

            if args.eval_every and stats.iteration % args.eval_every == 0:
                current = learner.cpu_net()
                result = learner.evaluate(
                    anchor, args.eval_games, args.simulations,
                    seed=int(rng.integers(0, 2**31 - 1)),
                    names=(f"gen{stats.iteration}", "previous"),
                )
                # A clean sweep's point estimate is the clamp, not a measurement.
                # Add the finite end of the interval instead, so the running
                # total stays a bound rather than becoming nonsense.
                low, high = result.interval
                if math.isinf(high):
                    cumulative_elo += low
                elif math.isinf(low):
                    cumulative_elo += high
                else:
                    cumulative_elo += result.elo
                print(f"         {result.summary()}   cumulative {cumulative_elo:+.0f} "
                      f"Elo (assumes transitivity)", flush=True)
                anchor = current

            if args.checkpoint_every and stats.iteration % args.checkpoint_every == 0:
                learner.save(args.out / f"{args.game}-gen{stats.iteration:04d}.pt",
                             buffer=args.save_buffer)
            learner.save(args.out / f"{args.game}-latest.pt", buffer=args.save_buffer)

    print(f"\ncheckpoints in {args.out}/")


if __name__ == "__main__":
    main()
