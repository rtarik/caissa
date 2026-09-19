"""Tests for turn order across the framework.

Until Dots & Boxes every game strictly alternated, and search, self-play and the
arena each flipped a value's sign once per move. They now ask ``to_play`` and
flip only when the seat changes. These tests pin both halves: the four
alternating games still alternate exactly, and the machinery gets a player moving
twice in a row right.
"""

from __future__ import annotations

import numpy as np
import pytest

from caissa.arena import play_game
from caissa.evaluator import UniformEvaluator
from caissa.games import GAMES
from caissa.games.dotsandboxes import LINES, DotsAndBoxes
from caissa.mcts import MCTS, MCTSConfig, Node
from caissa.selfplay import SelfPlayConfig
from caissa.selfplay import play_game as self_play

ALTERNATING = [name for name in GAMES if name != "dotsandboxes"]


def random_move(game, state, rng):
    return int(rng.choice(np.flatnonzero(game.legal_actions(state))))


# ----------------------------------------------------------------- the rules


@pytest.mark.parametrize("name", ALTERNATING)
def test_alternating_games_pass_the_turn_on_every_move(name):
    game = GAMES[name]()
    rng = np.random.default_rng(0)
    for _ in range(10):
        state = game.initial_state()
        assert game.to_play(state) == 0
        while game.terminal_value(state) is None:
            after = game.apply(state, random_move(game, state, rng))
            assert game.to_play(after) != game.to_play(state)
            state = after


def test_dots_and_boxes_keeps_the_turn_exactly_when_a_box_closes():
    game = DotsAndBoxes()
    rng = np.random.default_rng(1)
    for _ in range(10):
        state = game.initial_state()
        while game.terminal_value(state) is None:
            after = game.apply(state, random_move(game, state, rng))
            closed = sum(game.score(after)) > sum(game.score(state))
            assert (game.to_play(after) == game.to_play(state)) == closed
            state = after


# ---------------------------------------------------------------- the search


def test_backup_keeps_the_sign_across_a_bonus_move():
    """root -> a (turn passed) -> b (bonus move) -> c (turn passed).

    c's mover has the result. b's mover is c's opponent. a's mover is the *same*
    player as b's, because the move from a to b closed a box. The root's mover is
    a's opponent.
    """
    mcts = MCTS(DotsAndBoxes(), UniformEvaluator())
    root, a, b, c = Node(0.0), Node(0.0, flip=True), Node(0.0, flip=False), Node(0.0, flip=True)
    mcts._backup([root, a, b, c], 1.0)
    assert [n.value() for n in (root, a, b, c)] == [1.0, -1.0, -1.0, 1.0]


def test_children_are_marked_for_whether_they_keep_the_turn():
    game = DotsAndBoxes()
    # Three sides of box (0, 0): exactly one line closes it.
    state = game.initial_state()
    for line in (0, 5, 30):
        state = game.apply(state, line)
    mcts = MCTS(game, UniformEvaluator(), MCTSConfig(simulations=1))
    root = mcts.search(state, add_noise=False)
    # Expansion records the moves; a child's position is built on its first visit.
    assert all(child.state is None for child in root.children.values())
    for child in root.children.values():
        mcts._build(root, child)
    keeps = [action for action, child in root.children.items() if not child.flip]
    assert keeps == [31]


def solve(game, state):
    """Exact result for the player to move, by brute force - tiny endgames only."""
    value = game.terminal_value(state)
    if value is not None:
        return value
    seat = game.to_play(state)
    best = -1.0
    for action in np.flatnonzero(game.legal_actions(state)):
        child = game.apply(state, int(action))
        result = solve(game, child)
        best = max(best, result if game.to_play(child) == seat else -result)
    return best


def endgames(game, rng, count, lines_left):
    """Random endgames in which at least one move throws the result away."""
    found = []
    while len(found) < count:
        state = game.initial_state()
        while state.ply < LINES - lines_left:
            state = game.apply(state, random_move(game, state, rng))
        seat = game.to_play(state)
        scored = []
        for action in np.flatnonzero(game.legal_actions(state)):
            child = game.apply(state, int(action))
            result = solve(game, child)
            scored.append((int(action), result if game.to_play(child) == seat else -result))
        best = max(result for _, result in scored)
        good = {action for action, result in scored if result == best}
        if len(good) < len(scored):
            found.append((state, good))
    return found


def test_search_matches_exact_play_in_endgames_with_bonus_moves():
    """The end-to-end check of everything the bonus move touches in search.

    Endgames are small enough to solve by brute force and to search completely,
    so the right answer is known. With a knowledge-free evaluator every value the
    search holds came from a finished game, carried back up through box-closing
    lines - so a sign error on a bonus move shows up as a wrong move, not as a
    subtly weaker one.
    """
    game = DotsAndBoxes()
    for state, good in endgames(game, np.random.default_rng(7), count=25, lines_left=5):
        mcts = MCTS(game, UniformEvaluator(), MCTSConfig(simulations=2000),
                    rng=np.random.default_rng(0))
        policy, _ = mcts.run(state, temperature=0.0, add_noise=False)
        assert int(policy.argmax()) in good


# ----------------------------------------------------------------- self-play


class Replay:
    """Plays a fixed sequence of moves, so the game is known in advance."""

    def __init__(self, game, actions):
        self.game, self.actions, self.index = game, list(actions), 0

    def run(self, state, temperature=1.0, add_noise=True):
        policy = np.zeros(self.game.action_size)
        policy[self.actions[self.index]] = 1.0
        self.index += 1
        return policy, 0.0


def test_value_labels_follow_the_seat_not_the_move_count():
    game = DotsAndBoxes()
    rng = np.random.default_rng(8)
    actions, seats, state = [], [], game.initial_state()
    while game.terminal_value(state) is None:
        seats.append(game.to_play(state))
        actions.append(random_move(game, state, rng))
        state = game.apply(state, actions[-1])
    outcome, final_seat = game.terminal_value(state), game.to_play(state)

    samples = self_play(game, Replay(game, actions), SelfPlayConfig(temperature_moves=0),
                        rng=np.random.default_rng(0))

    expected = [outcome if seat == final_seat else -outcome for seat in seats]
    assert [sample.value for sample in samples] == expected

    # And the old rule - flip once per move - would have labelled it differently.
    counted = [outcome if (len(seats) - i) % 2 == 0 else -outcome for i in range(len(seats))]
    assert counted != expected


# ------------------------------------------------------------------- arena


def test_arena_gives_bonus_moves_to_the_player_who_earned_them():
    """Both who moves and who gets the credit must follow the seat.

    Played over several games on purpose. Every Dots & Boxes game is exactly sixty
    moves, so an arena crediting the result by counting moves agrees with the seat
    whenever a game ends with seat 0 to move - and a test built on one such game
    lets that bug straight through. This one was, until a mutation proved it.
    """
    game = DotsAndBoxes()
    final_seats = set()
    for seed in range(12):
        log: list[tuple[int, int]] = []

        class Recorder:
            def __init__(self, seat, rng):
                self.seat, self.rng = seat, rng

            def choose(self, game, state, rng):
                assert game.to_play(state) == self.seat, "asked to move out of turn"
                action = random_move(game, state, self.rng)
                log.append((self.seat, action))
                return action

        score = play_game(game, Recorder(0, np.random.default_rng(100 + seed)),
                          Recorder(1, np.random.default_rng(200 + seed)),
                          game.initial_state(), np.random.default_rng(seed))

        seats = [seat for seat, _ in log]
        assert any(a == b for a, b in zip(seats, seats[1:])), "no bonus move happened"

        # Credit must follow the seat that finished ahead, found independently.
        state = game.initial_state()
        for _, action in log:
            state = game.apply(state, action)
        final_seats.add(game.to_play(state))
        value = game.terminal_value(state)
        result = value if game.to_play(state) == 0 else -value
        assert score == {1.0: 1.0, -1.0: 0.0}[result]

    assert final_seats == {0, 1}, "games must finish on both seats, or credit by parity slips through"
