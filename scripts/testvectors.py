"""Generate cross-language test vectors.

    python scripts/testvectors.py

The game rules exist twice - Python for training, TypeScript for the browser -
which is a deliberate trade (see PLAN.md), but two implementations of the same
rules drift. A board encoded one way in training and another in the browser gives
a network being fed inputs it was never trained on: legal moves, sensible-looking
play, quietly worse than it was measured to be.

So Python emits positions with the answers it computes, and the TypeScript tests
must reproduce them exactly. The encodings matter most: those are what the
network actually consumes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from caissa.evaluator import UniformEvaluator
from caissa.games import GAMES
from caissa.mcts import MCTS, MCTSConfig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--game", default="connect4", choices=sorted(GAMES))
    parser.add_argument("--cases", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=None,
                        help="defaults to web/test/<game>-vectors.json")
    args = parser.parse_args()

    game = GAMES[args.game]()
    out = args.out or Path(f"web/test/{game.name}-vectors.json")
    rng = np.random.default_rng(args.seed)
    cases = []
    terminal_cases = 0

    while len(cases) < args.cases:
        state = game.initial_state()
        moves: list[int] = []
        # Record every position along a game, not just the last: the interesting
        # disagreements are mid-game, and finished positions exercise the
        # terminal logic that a sign error would flip.
        while True:
            cases.append({
                "moves": list(moves),
                "legal": [int(v) for v in game.legal_actions(state)],
                "terminal": game.terminal_value(state),
                "encoded": [int(v) for v in game.encode(state).ravel()],
            })
            outcome = game.terminal_value(state)
            if outcome is not None:
                terminal_cases += 1
                break
            legal = np.flatnonzero(game.legal_actions(state))
            action = int(rng.choice(legal))
            state = game.apply(state, action)
            moves.append(action)

    cases = cases[: args.cases]

    # Search vectors. With a uniform evaluator and no root noise the search is
    # fully deterministic, so the TypeScript port can be compared to Python's
    # visit counts exactly rather than approximately - which is the only way to
    # be sure a re-implementation of an algorithm this fiddly is faithful.
    searches = []
    rng = np.random.default_rng(args.seed + 1)
    mcts_rng = np.random.default_rng(0)
    while len(searches) < 40:
        state = game.initial_state()
        moves = []
        for _ in range(int(rng.integers(0, 14))):
            if game.terminal_value(state) is not None:
                break
            action = int(rng.choice(np.flatnonzero(game.legal_actions(state))))
            state = game.apply(state, action)
            moves.append(action)
        if game.terminal_value(state) is not None:
            continue
        simulations = int(rng.choice([8, 25, 60, 150]))
        search = MCTS(game, UniformEvaluator(), MCTSConfig(simulations=simulations),
                      rng=mcts_rng)
        root = search.search(state, add_noise=False)
        searches.append({
            "moves": moves,
            "simulations": simulations,
            "visits": [root.children[a].visit_count if a in root.children else 0
                       for a in range(game.action_size)],
            "rootValue": root.value(),
        })
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "game": game.name,
        "boardShape": list(game.board_shape),
        "inputPlanes": game.input_planes,
        "actionSize": game.action_size,
        "cases": cases,
        "searches": searches,
    }) + "\n")

    decided = sum(1 for c in cases if c["terminal"] is not None)
    print(f"{len(cases)} cases and {len(searches)} searches -> {out} "
          f"({out.stat().st_size / 1024:.0f} KB, {decided} finished positions)")


if __name__ == "__main__":
    main()
