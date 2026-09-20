"""Is search actually improving on the network's own policy?

    python scripts/improvement.py models/chess-imitation1.pt --simulations 50 200 800

The assumption the whole method rests on. AlphaZero trains the policy head
towards search's visit counts, which only teaches the network anything if those
visits are *better* than the priors search started from - search as a policy
improvement operator. Every self-play iteration is that one assumption applied
again to its own output, so when it fails, training does not stall: it walks
downhill, quietly, while all its own losses keep falling.

Held-out human moves are the yardstick here. They are not best moves, so no
absolute number means much; what means something is the comparison, on the same
positions, between the network alone and the network plus search. Search that
agrees with strong players *less* than the raw policy does is not improving it,
and training towards its visits would pull the network away from what it knows.

The simulation counts are the dial that decides this. Reporting several at once
says whether a deficit is a property of the network or just of a search too
short to correct it.
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from caissa.data.chess import board_of, move_of
from caissa.data.heldout import load_months
from caissa.games.chess import Chess, action_of, position
from caissa.mcts import MCTS, MCTSConfig
from caissa.network import NetworkConfig, NetworkEvaluator, PolicyValueNet


@dataclass
class Task:
    config: NetworkConfig
    state: dict
    fens: list[str]
    #: The action index of the move the human played, per position.
    played: list[int]
    simulations: list[int]
    seed: int


def judge(task: Task) -> dict[int, np.ndarray]:
    """Whether the raw policy and each search depth found the human's move."""
    torch.set_num_threads(1)
    game = Chess()
    net = PolicyValueNet.for_game(game, task.config)
    net.load_state_dict(task.state)
    evaluator = NetworkEvaluator(net)

    # 0 stands for the raw policy: the network's own first choice, no search.
    correct = {depth: np.zeros(len(task.fens), dtype=bool)
               for depth in [0, *task.simulations]}
    for index, (fen, played) in enumerate(zip(task.fens, task.played)):
        state = position(fen)
        priors, _ = evaluator.evaluate(game, state)
        correct[0][index] = int(priors.argmax()) == played
        for depth in task.simulations:
            mcts = MCTS(game, evaluator, MCTSConfig(simulations=depth),
                        rng=np.random.default_rng(task.seed + index))
            # No noise, no temperature: this is search trying to find the move.
            visits, _ = mcts.run(state, temperature=0.0, add_noise=False)
            correct[depth][index] = int(visits.argmax()) == played
    return correct


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--data", type=Path, default=Path("data/chess"))
    parser.add_argument("--months", nargs="+", default=["2020-01"])
    parser.add_argument("--positions", type=int, default=400)
    parser.add_argument("--simulations", type=int, nargs="+", default=[200, 800])
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    stored, games = load_months(args.data, args.months)
    held_out = np.flatnonzero(games["validation"][stored["game"]].astype(bool))
    rng = np.random.default_rng(args.seed)
    rows = rng.choice(held_out, size=min(args.positions, len(held_out)), replace=False)

    game = Chess()
    fens, played = [], []
    for row in rows:
        board = board_of(stored[row])
        fens.append(board.fen())
        played.append(action_of(board, move_of(stored[row])))

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = NetworkConfig(**checkpoint["config"]["network"])
    net = PolicyValueNet.for_game(game, config)
    net.load_state_dict(checkpoint["network"])
    state = {key: value.cpu() for key, value in net.state_dict().items()}

    chunks = np.array_split(np.arange(len(fens)), args.workers)
    tasks = [Task(config, state, [fens[i] for i in chunk], [played[i] for i in chunk],
                  args.simulations, args.seed + number)
             for number, chunk in enumerate(chunks) if len(chunk)]
    print(f"{args.checkpoint.stem}: {len(fens)} held-out positions, "
          f"search at {', '.join(str(depth) for depth in args.simulations)} "
          f"simulations", flush=True)
    with mp.get_context("spawn").Pool(len(tasks)) as pool:
        results = pool.map(judge, tasks)

    merged = {depth: np.concatenate([result[depth] for result in results])
              for depth in [0, *args.simulations]}
    raw = merged[0].mean()
    print(f"\n{'':<14}{'agrees with the human':>24}{'vs raw policy':>16}")
    print(f"{'raw policy':<14}{raw:>23.1%}{'-':>16}")
    for depth in args.simulations:
        accuracy = merged[depth].mean()
        print(f"{f'{depth} sims':<14}{accuracy:>23.1%}{accuracy - raw:>+15.1%}")

    print("\nA search that agrees less than the raw policy is not improving it: "
          "\ntraining towards its visit counts would teach the network to play worse.")


if __name__ == "__main__":
    main()
