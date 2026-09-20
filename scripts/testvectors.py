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

#: How many moves into a game the search vectors start, per game.
#:
#: Early positions for most games. Late ones for Dots & Boxes, because the only
#: new thing its search does - keeping a value's sign across a bonus move - is
#: invisible until simulations reach finished games. With a knowledge-free
#: evaluator every unfinished position is worth exactly zero, and zero looks the
#: same whichever way it is flipped, so early-position vectors would agree with a
#: broken port.
SEARCH_PLIES = {"dotsandboxes": (54, 58), "chess": (10, 160)}

#: Simulation budgets for those searches, per game. Dots & Boxes gets larger ones
#: for the same reason it gets endgames: a search only exercises the bonus-move
#: sign rule once it reaches finished games, and eight simulations rarely do.
#: Gomoku needs a budget large enough to visit eighty-one children at all.
SEARCH_SIMULATIONS = {"dotsandboxes": [60, 150, 400], "gomoku": [100, 200, 400]}

#: Share of searches that must start where a move wins on the spot. The same
#: problem Dots & Boxes solved with endgames: a random chess position almost never
#: has a decided game within a knowledge-free search's reach, and the first chess
#: vectors reached a result in 0 of 40 searches - every root value zero, which
#: agrees with a port whatever it does with signs.
#: Gomoku joined this list when its rules stopped restricting play to the
#: neighbourhood of the stones: free-style, a knowledge-free search over eighty
#: empty points reaches a finished game in none of forty searches, exactly as the
#: restriction's own docstring predicts. Starting where a stone wins on the spot
#: puts the answers back.
SEARCH_WIN_IN_ONE = {"chess": 0.5, "gomoku": 0.75}


def wins_in_one(game, state) -> bool:
    """Whether some legal move ends the game in the mover's favour."""
    seat = game.to_play(state)
    for action in np.flatnonzero(game.legal_actions(state)):
        child = game.apply(state, int(action))
        result = game.terminal_value(child)
        if result is not None and (result if game.to_play(child) == seat else -result) > 0:
            return True
    return False

#: Rules cases per game, as the committed vector files were generated. Recorded
#: so that regenerating with the defaults reproduces those files exactly - which
#: is how a change meant to alter no result (lazy children in the search, say)
#: proves that it didn't.
CASES = {"connect4": 250, "reversi": 150, "gomoku": 150, "isola": 150, "chess": 300}

#: Games too long to record every position of. A random chess game runs to a few
#: hundred plies, so recording every position would fill the file from one or two
#: games; every tenth position, and always the last, spreads it over dozens.
CASE_STRIDE = {"chess": 10}

#: Above this many actions a dense mask per position would dominate the file -
#: 4,672 entries for every chess position - so legal actions, and the search's
#: visit counts, are listed by index instead.
SPARSE_ABOVE = 1000


def number(value) -> int | float:
    """A plane value for JSON: an integer where it is one, as every earlier game's
    are, and the exact float otherwise - chess's fifty-move counter is a fraction."""
    value = float(value)
    return int(value) if value.is_integer() else value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--game", default="connect4", choices=sorted(GAMES))
    parser.add_argument("--cases", type=int, default=None,
                        help="defaults to the count the committed file was made with")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=None,
                        help="defaults to web/test/<game>-vectors.json")
    args = parser.parse_args()

    game = GAMES[args.game]()
    out = args.out or Path(f"web/test/{game.name}-vectors.json")
    rng = np.random.default_rng(args.seed)
    cases = []
    terminal_cases = 0

    wanted = args.cases or CASES.get(game.name, 200)
    stride = CASE_STRIDE.get(game.name, 1)
    sparse = game.action_size > SPARSE_ABOVE
    while len(cases) < wanted:
        state = game.initial_state()
        moves: list[int] = []
        # Record positions all along a game, not just the last: the interesting
        # disagreements are mid-game, and finished positions exercise the
        # terminal logic that a sign error would flip.
        while True:
            outcome = game.terminal_value(state)
            if len(moves) % stride == 0 or outcome is not None:
                legal = game.legal_actions(state)
                case: dict = {"moves": list(moves)}
                if sparse:
                    case["legalIndices"] = [int(a) for a in np.flatnonzero(legal)]
                else:
                    case["legal"] = [int(v) for v in legal]
                case["toPlay"] = game.to_play(state)
                case["terminal"] = outcome
                case["encoded"] = [number(v) for v in game.encode(state).ravel()]
                cases.append(case)
            if outcome is not None:
                terminal_cases += 1
                break
            legal = np.flatnonzero(game.legal_actions(state))
            action = int(rng.choice(legal))
            state = game.apply(state, action)
            moves.append(action)

    cases = cases[:wanted]

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
        low, high = SEARCH_PLIES.get(game.name, (0, 14))
        for _ in range(int(rng.integers(low, high))):
            if game.terminal_value(state) is not None:
                break
            action = int(rng.choice(np.flatnonzero(game.legal_actions(state))))
            state = game.apply(state, action)
            moves.append(action)
        if game.terminal_value(state) is not None:
            continue
        if (len(searches) < round(40 * SEARCH_WIN_IN_ONE.get(game.name, 0))
                and not wins_in_one(game, state)):
            continue
        simulations = int(rng.choice(SEARCH_SIMULATIONS.get(game.name, [8, 25, 60, 150])))
        search = MCTS(game, UniformEvaluator(), MCTSConfig(simulations=simulations),
                      rng=mcts_rng)
        root = search.search(state, add_noise=False)
        if sparse:
            visits = {"visitsByAction": [[a, child.visit_count]
                                         for a, child in root.children.items() if child.visit_count]}
        else:
            visits = {"visits": [root.children[a].visit_count if a in root.children else 0
                                 for a in range(game.action_size)]}
        searches.append({"moves": moves, "simulations": simulations, **visits,
                         "rootValue": root.value()})
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
    resolved = sum(1 for s in searches if s["rootValue"] != 0)
    print(f"{len(cases)} cases and {len(searches)} searches -> {out} "
          f"({out.stat().st_size / 1024:.0f} KB, {decided} finished positions, "
          f"{resolved} searches reaching a result)")


if __name__ == "__main__":
    main()
