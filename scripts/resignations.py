"""How often resignation would throw a game away.

    python scripts/resignations.py models/chess-imitation1.pt --games 40

Resignation is a compute trade, and it is the only self-play setting that can put
a *wrong label* on a whole game: every position of a resigned game is labelled a
loss for the side that gave up, and if they were not actually lost, the network
is taught that a fine position is hopeless. AlphaZero handled this by playing a
tenth of its games out regardless and counting how often resignation would have
been wrong - keeping the threshold low enough that it almost never was.

This does the same measurement from the outside. The games are played with
resignation *off*, recording search's value at every move, and the rule is then
replayed over the record for several thresholds at once. One run answers both
questions: what a threshold costs in wrong labels, and what it buys in plies.

Read the false-positive rate, not the average. A threshold that is wrong on 1% of
games is a threshold worth using; one that is wrong on 10% is buying compute with
the training signal, which is the wrong direction.
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from caissa.games import GAMES
from caissa.mcts import MCTS, MCTSConfig
from caissa.network import NetworkConfig, NetworkEvaluator, PolicyValueNet
from caissa.selfplay import despairing

#: Thresholds reported side by side. -0.9 is AlphaZero's.
THRESHOLDS = (-0.99, -0.95, -0.9, -0.85, -0.8)


@dataclass
class Played:
    """One game, as the audit needs it."""

    #: Search's value for the player to move, one per ply.
    values: np.ndarray
    #: Who was to move, one per ply.
    seats: np.ndarray
    #: The game's result for each ply's mover: +1, 0 or -1.
    labels: np.ndarray


@dataclass
class Task:
    game_name: str
    config: NetworkConfig
    state: dict
    simulations: int
    temperature_moves: int
    max_plies: int
    seed: int


def play(task: Task) -> Played:
    """Play one game to its natural end, recording what search thought."""
    torch.set_num_threads(1)
    game = GAMES[task.game_name]()
    net = PolicyValueNet.for_game(game, task.config)
    net.load_state_dict(task.state)
    mcts = MCTS(game, NetworkEvaluator(net), MCTSConfig(simulations=task.simulations))
    rng = np.random.default_rng(task.seed)

    state = game.initial_state()
    values, seats = [], []
    while (outcome := game.terminal_value(state)) is None and len(values) < task.max_plies:
        temperature = 1.0 if len(values) < task.temperature_moves else 0.0
        policy, value = mcts.run(state, temperature=temperature, add_noise=True)
        values.append(value)
        seats.append(game.to_play(state))
        state = game.apply(state, int(rng.choice(len(policy), p=policy)))
    if outcome is None:
        outcome = 0.0  # hit the cap: a draw, as self-play would call it

    # The result from each mover's own side, exactly as self-play labels it.
    final_seat = game.to_play(state)
    seats = np.asarray(seats)
    labels = np.where(seats == final_seat, outcome, -outcome)
    return Played(np.asarray(values, dtype=np.float32), seats, labels.astype(np.float32))


def resignation(played: Played, threshold: float, moves: int) -> tuple[int, float] | None:
    """Where the rule would have fired: (ply, the result the resigner threw away)."""
    hopeless: dict[int, int] = {}
    for ply, (value, seat) in enumerate(zip(played.values, played.seats)):
        if despairing(hopeless, int(seat), float(value), threshold) >= moves:
            return ply, float(played.labels[ply])
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--game", default="chess", choices=sorted(GAMES))
    parser.add_argument("--games", type=int, default=40)
    parser.add_argument("--simulations", type=int, default=200)
    parser.add_argument("--temperature-moves", type=int, default=30)
    parser.add_argument("--max-plies", type=int, default=200)
    parser.add_argument("--moves", type=int, default=2,
                        help="dismal evaluations in a row, by the same player")
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    game = GAMES[args.game]()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = NetworkConfig(**checkpoint["config"]["network"])
    net = PolicyValueNet.for_game(game, config)
    net.load_state_dict(checkpoint["network"])
    state = {key: value.cpu() for key, value in net.state_dict().items()}

    tasks = [Task(args.game, config, state, args.simulations, args.temperature_moves,
                  args.max_plies, args.seed + number) for number in range(args.games)]
    print(f"{args.checkpoint.stem}: playing {args.games} games out in full "
          f"at {args.simulations} simulations", flush=True)
    with mp.get_context("spawn").Pool(args.workers) as pool:
        games = pool.map(play, tasks)

    plies = np.array([len(played.values) for played in games])
    results = np.array([played.labels[0] if len(played.labels) else 0.0 for played in games])
    print(f"{len(games)} games, {plies.mean():.0f} plies on average; "
          f"first mover won {np.mean(results > 0):.0%}, drew {np.mean(results == 0):.0%}\n")

    print(f"{'threshold':>10}{'resigned':>10}{'was lost':>10}{'was drawn':>11}"
          f"{'was won':>9}{'plies saved':>13}")
    for threshold in THRESHOLDS:
        fired = [(played, found) for played in games
                 if (found := resignation(played, threshold, args.moves))]
        if not fired:
            print(f"{threshold:>10.2f}{0:>10.0%}{'-':>10}{'-':>11}{'-':>9}{0:>13.0%}")
            continue
        thrown = np.array([result for _, (_, result) in fired])
        saved = sum(len(played.values) - ply for played, (ply, _) in fired)
        print(f"{threshold:>10.2f}{len(fired) / len(games):>10.0%}"
              f"{np.mean(thrown < 0):>10.0%}{np.mean(thrown == 0):>11.0%}"
              f"{np.mean(thrown > 0):>9.0%}{saved / plies.sum():>13.0%}")

    print("\n'was lost' is resignation being right. 'was drawn' and 'was won' are games "
          "\nmislabelled as losses in the training data - the cost of the compute saved.")


if __name__ == "__main__":
    main()
