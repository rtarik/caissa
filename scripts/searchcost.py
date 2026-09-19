"""What one search step costs: the rules against the network.

    python scripts/searchcost.py --game chess

Every expansion asks the rules whether the game is over and which moves are
legal, builds a child position for every legal move, and asks the network for
priors and a value. While the network dominates, the rules are a detail. When
the rules cost as much as the network - which pure-Python chess might - the
search should build a child only when it first visits it: most children built
are never visited at all.

Measured on one CPU thread, the way a self-play worker runs, with an untrained
network of the size training will use.
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import torch

from caissa.evaluator import UniformEvaluator
from caissa.games import GAMES
from caissa.mcts import MCTS, MCTSConfig
from caissa.network import NetworkConfig, NetworkEvaluator, PolicyValueNet


def sample_positions(game, count: int, rng: np.random.Generator) -> list:
    """Unfinished positions a random number of random moves into a game."""
    positions = []
    while len(positions) < count:
        state = game.initial_state()
        for _ in range(int(rng.integers(0, 60))):
            if game.terminal_value(state) is not None:
                break
            state = game.apply(state, int(rng.choice(np.flatnonzero(game.legal_actions(state)))))
        if game.terminal_value(state) is None:
            positions.append(state)
    return positions


def tree_size(root) -> tuple[int, int]:
    """(child positions built, children ever visited) below ``root``."""
    built = visited = 0
    stack = [root]
    while stack:
        node = stack.pop()
        for child in node.children.values():
            built += 1
            if child.visit_count:
                visited += 1
                stack.append(child)
    return built, visited


def per_simulation(game, evaluator, positions, simulations: int) -> tuple[float, int, int]:
    """Seconds per simulation, with the tree statistics of the searches."""
    mcts = MCTS(game, evaluator, MCTSConfig(simulations=simulations), rng=np.random.default_rng(0))
    built = visited = 0
    started = time.perf_counter()
    for state in positions:
        root = mcts.search(state, add_noise=False)
        b, v = tree_size(root)
        built, visited = built + b, visited + v
    return (time.perf_counter() - started) / (len(positions) * simulations), built, visited


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--game", default="chess", choices=sorted(GAMES))
    parser.add_argument("--positions", type=int, default=20)
    parser.add_argument("--simulations", type=int, default=200)
    parser.add_argument("--blocks", type=int, default=6)
    parser.add_argument("--channels", type=int, default=64)
    args = parser.parse_args()

    torch.set_num_threads(1)
    game = GAMES[args.game]()
    squares = game.board_shape[0] * game.board_shape[1]
    head = "conv" if game.action_size % squares == 0 else "dense"
    net = PolicyValueNet.for_game(
        game, NetworkConfig(blocks=args.blocks, channels=args.channels, policy_head=head))
    network = NetworkEvaluator(net)
    positions = sample_positions(game, args.positions, np.random.default_rng(0))

    encoded = [torch.from_numpy(game.encode(state)).unsqueeze(0) for state in positions]
    with torch.no_grad():
        net.eval()
        for x in encoded[:3]:
            net(x)  # warm-up
        started = time.perf_counter()
        for x in encoded:
            net(x)
        forward = (time.perf_counter() - started) / len(encoded)

    rules, built, visited = per_simulation(game, UniformEvaluator(), positions, args.simulations)
    total, _, _ = per_simulation(game, network, positions, args.simulations)
    simulations = len(positions) * args.simulations

    print(f"{args.game}: {args.blocks}x{args.channels} network, {head} policy head, "
          f"{net.parameter_count() / 1e6:.2f} M parameters, one CPU thread")
    print(f"  network forward pass alone     {forward * 1e6:8.0f} us")
    print(f"  search step, rules and tree    {rules * 1e6:8.0f} us   (uniform evaluator)")
    print(f"  search step, with the network  {total * 1e6:8.0f} us   "
          f"-> {1 / total:,.0f} simulations/s per worker")
    print(f"  rules and tree share           {rules / total:8.0%}")
    print(f"  child positions built per simulation {built / simulations:.1f}, "
          f"ever visited {visited / simulations:.2f}")


if __name__ == "__main__":
    main()
