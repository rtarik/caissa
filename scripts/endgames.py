"""Grade a Dots & Boxes network against exact play, in endgames.

    python scripts/endgames.py models/dotsandboxes-latest.pt --positions 300

The training loss and the arena both measure the agent against itself, and this
project has watched both look healthy while the agent was weak. Dots & Boxes
endgames are small enough to solve exactly, so for them there is an answer that
owes nothing to the network: a move either keeps the best achievable result or
throws it away.

Two numbers come out, because one of them is the reason this game is on the ladder.

**All endgames** where some move is a mistake.

**Traps**: positions where a box is there for the taking and *every* way of taking
it loses. This is the double-dealing situation at the heart of the game - decline
the last two boxes of a chain so the opponent has to open the next one - and it is
exactly where greedy play goes wrong. A self-play agent can sit in the greedy
strategy for a long time, because greedy play beats weak play. A greedy player
scores zero here by construction, which is what makes the number worth watching.

Positions come from *sensible* play by default: take a box when one is available,
never hand one over while a safe line exists, otherwise open something. That is
the play that builds long chains, and chains are where traps live. Random play was
tried first and was useless for this: across 200 random endgames it produced zero
trap positions and a greedy player scored 100%, because random lines scatter
three-sided boxes everywhere and grabbing them is always right. The exam could not
ask the one question it was written for.

Either way the positions are the same exam for every network, not the ones a strong
player would reach. Treat the result as a floor on what the network understands,
not a rating.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from caissa.games.dotsandboxes import BOX_MASKS, LINE_BOXES, LINES, DotsAndBoxes
from caissa.mcts import MCTS, MCTSConfig
from caissa.network import NetworkConfig, NetworkEvaluator, PolicyValueNet


def closes_box(state, action: int) -> bool:
    lines = state.lines | (1 << action)
    return any((lines & BOX_MASKS[box]) == BOX_MASKS[box] for box in LINE_BOXES[action])


def safe(state, action: int) -> bool:
    """Whether a line leaves no neighbouring box with exactly three sides drawn."""
    lines = state.lines | (1 << action)
    return all((lines & BOX_MASKS[box]).bit_count() != 3 for box in LINE_BOXES[action])


def random_move(game, state, rng) -> int:
    return int(rng.choice(np.flatnonzero(game.legal_actions(state))))


def sensible_move(game, state, rng) -> int:
    """Take a box if possible; otherwise a safe line; otherwise anything."""
    legal = [int(a) for a in np.flatnonzero(game.legal_actions(state))]
    closing = [a for a in legal if closes_box(state, a)]
    if closing:
        return int(rng.choice(closing))
    safe_lines = [a for a in legal if safe(state, a)]
    return int(rng.choice(safe_lines or legal))


class Solver:
    """Exact results by memoised brute force - endgames only.

    The key leaves out the seat on purpose. In canonical perspective a position's
    value for the player to move does not depend on which of the two players that is,
    so both seats share one table entry and the table stays small.
    """

    def __init__(self, game: DotsAndBoxes):
        self.game = game
        self.table: dict[tuple[int, int, int], float] = {}

    def value(self, state) -> float:
        key = (state.lines, state.mine, state.theirs)
        cached = self.table.get(key)
        if cached is not None:
            return cached
        result = self.game.terminal_value(state)
        if result is None:
            result = -1.0
            for action in np.flatnonzero(self.game.legal_actions(state)):
                child = self.game.apply(state, int(action))
                outcome = self.value(child)
                # Keep the sign across a bonus move; flip it when the turn passes.
                result = max(result, outcome if child.seat == state.seat else -outcome)
                if result == 1.0:
                    break  # nothing beats a win, and 25 boxes allow no draw
        self.table[key] = result
        return result

    def moves(self, state) -> tuple[set[int], list[int]]:
        """(moves that keep the best result, all legal moves)."""
        scored = []
        for action in np.flatnonzero(self.game.legal_actions(state)):
            child = self.game.apply(state, int(action))
            outcome = self.value(child)
            scored.append((int(action), outcome if child.seat == state.seat else -outcome))
        best = max(outcome for _, outcome in scored)
        return {action for action, outcome in scored if outcome == best}, [a for a, _ in scored]


def sample(game, solver, rng, count: int, lines_left: list[int], play=sensible_move,
           min_traps: int = 0):
    """Endgames, reached by ``play``, in which at least one move throws the result away.

    Keeps going past ``count`` until ``min_traps`` traps have turned up. They are
    about one position in twenty, and a trap score over ten of them is noise - two
    right against four right is not a finding.
    """
    found, tried, traps = [], 0, 0
    while len(found) < count or traps < min_traps:
        tried += 1
        state = game.initial_state()
        target = LINES - int(rng.choice(lines_left))
        while state.ply < target:
            state = game.apply(state, play(game, state, rng))
        good, legal = solver.moves(state)
        if len(good) < len(legal):
            closing = [a for a in legal if closes_box(state, a)]
            trap = bool(closing) and not good.intersection(closing)
            traps += trap
            found.append((state, good, legal, closing, trap))
    return found, tried


def load(path: Path, game):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint["game"] != game.name:
        raise SystemExit(f"{path} is for {checkpoint['game']}, not {game.name}")
    net = PolicyValueNet.for_game(game, NetworkConfig(**checkpoint["config"]["network"]))
    net.load_state_dict(checkpoint["network"])
    return net, checkpoint["iteration"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--positions", type=int, default=300)
    parser.add_argument("--simulations", type=int, default=300)
    parser.add_argument("--lines-left", default="6,7,8,9,10,11,12",
                        help="comma-separated; how far from the end positions are drawn")
    parser.add_argument("--min-traps", type=int, default=50,
                        help="keep sampling until this many traps are found")
    parser.add_argument("--play", default="sensible", choices=("sensible", "random"),
                        help="how positions are reached; random rarely builds chains")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.set_num_threads(1)  # one position at a time: faster on one thread
    game = DotsAndBoxes()
    solver = Solver(game)
    net, generation = load(args.checkpoint, game)
    evaluator = NetworkEvaluator(net)

    rng = np.random.default_rng(args.seed)
    play = sensible_move if args.play == "sensible" else random_move
    sampled, tried = sample(game, solver, rng, args.positions,
                            [int(n) for n in args.lines_left.split(",")], play, args.min_traps)
    positions = sampled[: args.positions]
    traps = [p for p in sampled if p[4]]

    def expected(chooser, subset):
        return 100.0 * sum(chooser(p) for p in subset) / len(subset) if subset else float("nan")

    random_pick = lambda p: len(p[1]) / len(p[2])
    greedy_pick = lambda p: (len(p[1].intersection(p[3])) / len(p[3])) if p[3] else len(p[1]) / len(p[2])

    def policy_pick(p):
        priors, _ = evaluator.evaluate(game, p[0])
        return float(int(priors.argmax()) in p[1])

    def search_pick(p):
        mcts = MCTS(game, evaluator, MCTSConfig(simulations=args.simulations),
                    rng=np.random.default_rng(0))
        policy, _ = mcts.run(p[0], temperature=0.0, add_noise=False)
        return float(int(policy.argmax()) in p[1])

    print(f"{args.checkpoint.name}: generation {generation}")
    print(f"{len(positions)} endgames from {args.play} play with a mistake available, plus "
          f"{len(traps)} traps (from {tried} sampled, {args.lines_left} lines left)\n")
    print(f"  {'':24s} {'all endgames':>13s} {'traps':>9s}")
    for label, chooser in (("random play", random_pick), ("greedy: take any box", greedy_pick),
                           ("network alone", policy_pick),
                           (f"network + {args.simulations} sims", search_pick)):
        print(f"  {label:24s} {expected(chooser, positions):12.1f}% {expected(chooser, traps):8.1f}%",
              flush=True)


if __name__ == "__main__":
    main()
