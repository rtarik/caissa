"""How often a Dots & Boxes network takes a box it is handed early in the game.

    python scripts/gifts.py models/dotsandboxes-gen0040.pt models/dotsandboxes-latest.pt

Self-play only visits the positions its own play leads to, and competent play
almost never hands the opponent a box in the opening. So a network can grow strong
in every game it plays against itself and still not know what to do when a person
gives a box away - the one mistake every beginner makes.

The positions: a few plies of sensible play, then one line that leaves a box on
three sides. Taking that box is right. A capture does not pass the turn, so the
taker still gets to make the move they would have made anyway; declining only
leaves the box to the opponent. The real reason to decline a capture is to keep
control of the long chains at the end, which is what ``endgames.py`` grades - a
lone box this early is not that.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from caissa.games.dotsandboxes import DotsAndBoxes
from caissa.mcts import MCTS, MCTSConfig
from caissa.network import NetworkEvaluator
from endgames import closes_box, load, safe, sensible_move  # the sibling script


def blunders(game, rng, count: int, plies: tuple[int, int] = (2, 14)):
    """Positions just after one blunder, each with the set of lines that take the box."""
    found = []
    while len(found) < count:
        state = game.initial_state()
        target = int(rng.integers(plies[0], plies[1] + 1))
        while state.ply < target:
            state = game.apply(state, sensible_move(game, state, rng))
        legal = [int(a) for a in np.flatnonzero(game.legal_actions(state))]
        if any(closes_box(state, a) for a in legal):
            continue  # a box is already on offer, so this would not be a single blunder
        unsafe = [a for a in legal if not safe(state, a)]
        if not unsafe:
            continue
        state = game.apply(state, int(rng.choice(unsafe)))
        taking = {int(a) for a in np.flatnonzero(game.legal_actions(state))
                  if closes_box(state, int(a))}
        found.append((state, taking))
    return found


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoints", type=Path, nargs="+")
    parser.add_argument("--positions", type=int, default=200)
    parser.add_argument("--simulations", type=int, default=200,
                        help="the default matches the page's default strength")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.set_num_threads(1)  # one position at a time: faster on one thread
    game = DotsAndBoxes()
    positions = blunders(game, np.random.default_rng(args.seed), args.positions)
    print(f"{len(positions)} boxes handed over after 2-14 plies of sensible play; "
          f"how often each network takes the box\n")
    print(f"  {'':28s} {'policy alone':>12s} {f'{args.simulations} sims':>10s}")

    for path in args.checkpoints:
        net, _ = load(path, game)
        evaluator = NetworkEvaluator(net)
        alone = searched = 0
        for state, taking in positions:
            priors, _ = evaluator.evaluate(game, state)
            alone += int(priors.argmax()) in taking
            mcts = MCTS(game, evaluator, MCTSConfig(simulations=args.simulations),
                        rng=np.random.default_rng(0))
            policy, _ = mcts.run(state, temperature=0.0, add_noise=False)
            searched += int(policy.argmax()) in taking
        n = len(positions)
        print(f"  {path.name:28s} {100 * alone / n:11.1f}% {100 * searched / n:9.1f}%",
              flush=True)


if __name__ == "__main__":
    main()
