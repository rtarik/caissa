"""Tests for Monte Carlo tree search.

The theme throughout: search is tested with evaluators that have *no* useful
knowledge, so anything the search finds was found by searching. That keeps
"the search is broken" clearly separable from "the network is untrained", which
is the diagnosis that matters most while building a reinforcement learner.
"""

from __future__ import annotations

import numpy as np
import pytest

from caissa.evaluator import UniformEvaluator
from caissa.games.connect4 import COLS, ROWS, Connect4
from caissa.mcts import MCTS, MCTSConfig, Node


@pytest.fixture
def game() -> Connect4:
    return Connect4()


def play(game: Connect4, *columns: int):
    state = game.initial_state()
    for column in columns:
        state = game.apply(state, column)
    return state


def make_mcts(game, simulations=200, evaluator=None, seed=0, **kwargs):
    return MCTS(
        game,
        evaluator or UniformEvaluator(),
        MCTSConfig(simulations=simulations, **kwargs),
        rng=np.random.default_rng(seed),
    )


# Three in column 0 for the player to move, who wins by playing column 0 again.
WIN_IN_ONE = (0, 1, 0, 1, 0, 1)
# Three in column 0 for the *opponent*; the player to move must block there.
MUST_BLOCK = (0, 1, 0, 1, 0)


def test_finds_a_win_in_one_without_any_knowledge(game):
    """Search alone, with a knowledge-free evaluator, must see an immediate win.

    If this fails, nothing downstream can be rescued by training.
    """
    policy, _ = make_mcts(game, simulations=100).run(
        play(game, *WIN_IN_ONE), temperature=1.0
    )
    assert policy.argmax() == 0
    assert policy[0] > 0.5


def test_finds_the_only_blocking_move(game):
    """Two-ply lookahead: every other move loses immediately."""
    policy, _ = make_mcts(game, simulations=600).run(
        play(game, *MUST_BLOCK), temperature=0.0
    )
    assert policy.argmax() == 0


def test_values_carry_the_mover_relative_sign(game):
    """A won position is positive for its mover; its winning child is -1.

    The child records the result from the *opponent's* perspective, because that
    is who moves there. Getting this backwards is the defining silent failure of
    an AlphaZero implementation, so it is asserted directly.
    """
    mcts = make_mcts(game, simulations=100)
    root = mcts.search(play(game, *WIN_IN_ONE), add_noise=False)

    assert root.value() > 0.5, "player to move is winning, so the root is positive"
    assert root.children[0].value() == -1.0, "the losing side sees the win as -1"


def test_search_overrules_a_misleading_evaluator(game):
    """Search improves on its priors - the property the whole algorithm rests on.

    An evaluator that is confidently wrong steers a shallow search astray. Given
    enough simulations, search discovers the truth regardless. This is precisely
    what makes the visit distribution a better training target than the priors
    that produced it.
    """

    class Misleading:
        """Insists on column 6 and claims the position is lost."""

        def evaluate(self, game, state):
            legal = game.legal_actions(state)
            priors = np.full(game.action_size, 0.001, dtype=np.float32)
            priors[COLS - 1] = 1.0
            priors *= legal
            return priors / priors.sum(), -1.0

    state = play(game, *WIN_IN_ONE)

    shallow, _ = make_mcts(game, 4, Misleading()).run(state, temperature=0.0)
    assert shallow.argmax() == COLS - 1, "a shallow search just follows the prior"

    deep, _ = make_mcts(game, 400, Misleading()).run(state, temperature=0.0)
    assert deep.argmax() == 0, "a deeper search finds the win despite the prior"


def test_more_simulations_concentrate_the_policy(game):
    state = play(game, *WIN_IN_ONE)
    few, _ = make_mcts(game, 30).run(state, temperature=1.0)
    many, _ = make_mcts(game, 600).run(state, temperature=1.0)
    assert many.max() > few.max()


def test_visits_are_only_spent_on_legal_moves(game):
    state = play(game, *([0] * ROWS))  # column 0 is full
    policy, _ = make_mcts(game, 100).run(state, temperature=1.0)
    assert policy[0] == 0.0
    assert np.isclose(policy.sum(), 1.0)


def test_simulation_budget_is_respected(game):
    mcts = make_mcts(game, simulations=50)
    root = mcts.search(game.initial_state(), add_noise=False)
    assert root.visit_count == 50
    # Every simulation after the root's own expansion descends into a child.
    assert sum(c.visit_count for c in root.children.values()) == 49


def test_cannot_search_a_finished_game(game):
    finished = play(game, 0, 1, 0, 1, 0, 1, 0)
    with pytest.raises(ValueError):
        make_mcts(game).search(finished)


# --------------------------------------------------------------------- backup


def test_backup_alternates_sign_up_the_tree(game):
    """A win for the leaf's mover is a loss for their parent, and so on."""
    mcts = make_mcts(game)
    path = [Node(0.0) for _ in range(4)]
    mcts._backup(path, 1.0)

    assert [n.value() for n in path] == [-1.0, 1.0, -1.0, 1.0]
    assert all(n.visit_count == 1 for n in path)


# ----------------------------------------------------------------- temperature


def fake_root(counts: dict[int, int]) -> Node:
    root = Node(0.0)
    for action, visits in counts.items():
        child = Node(0.0)
        child.visit_count = visits
        root.children[action] = child
    return root


def test_temperature_zero_is_greedy(game):
    policy = make_mcts(game).policy(fake_root({0: 10, 1: 90, 2: 5}), temperature=0.0)
    expected = np.zeros(COLS)
    expected[1] = 1.0
    np.testing.assert_array_equal(policy, expected)


def test_temperature_one_is_proportional_to_visits(game):
    policy = make_mcts(game).policy(fake_root({0: 10, 1: 90, 2: 0}), temperature=1.0)
    assert policy[0] == pytest.approx(0.1)
    assert policy[1] == pytest.approx(0.9)
    assert policy[2] == 0.0


def test_low_temperature_sharpens_without_overflowing(game):
    root = fake_root({0: 40, 1: 60})
    sharp = make_mcts(game).policy(root, temperature=0.02)
    assert np.isfinite(sharp).all()
    assert sharp[1] > 0.99


# -------------------------------------------------------------- dirichlet noise


def test_noise_perturbs_root_priors_but_keeps_a_distribution(game):
    state = play(game, 3)
    clean = make_mcts(game, 1).search(state, add_noise=False)
    noisy = make_mcts(game, 1, seed=5).search(state, add_noise=True)

    clean_priors = np.array([c.prior for c in clean.children.values()])
    noisy_priors = np.array([c.prior for c in noisy.children.values()])

    assert not np.allclose(clean_priors, noisy_priors)
    assert noisy_priors.sum() == pytest.approx(1.0)
    assert (noisy_priors > 0).all()


def test_noise_respects_its_epsilon(game):
    """With epsilon 0 the priors must be untouched, however noisy the draw."""
    state = play(game, 3)
    clean = make_mcts(game, 1).search(state, add_noise=False)
    zeroed = make_mcts(game, 1, dirichlet_epsilon=0.0).search(state, add_noise=True)

    np.testing.assert_allclose(
        [c.prior for c in clean.children.values()],
        [c.prior for c in zeroed.children.values()],
    )


def test_noise_widens_the_moves_that_get_explored(game):
    """The point of root noise: moves the evaluator dismissed still get visits.

    Without it, a confidently wrong evaluator's rejected moves would never be
    played, never enter training data, and never be reconsidered.
    """

    class Narrow:
        def evaluate(self, game, state):
            legal = game.legal_actions(state)
            priors = np.full(game.action_size, 1e-6, dtype=np.float32)
            priors[3] = 1.0
            priors *= legal
            return priors / priors.sum(), 0.0

    state = game.initial_state()
    without = make_mcts(game, 60, Narrow(), seed=1).search(state, add_noise=False)
    with_noise = make_mcts(game, 60, Narrow(), seed=1).search(state, add_noise=True)

    explored_without = sum(c.visit_count > 0 for c in without.children.values())
    explored_with = sum(c.visit_count > 0 for c in with_noise.children.values())
    assert explored_with > explored_without


def test_search_is_deterministic_without_noise(game):
    state = play(game, 3, 3, 4)
    first, _ = make_mcts(game, 120, seed=1).run(state, add_noise=False)
    second, _ = make_mcts(game, 120, seed=99).run(state, add_noise=False)
    np.testing.assert_array_equal(first, second)


def test_noise_is_applied_only_at_the_root(game):
    """Deeper nodes must keep the evaluator's untouched priors.

    Root noise exists to vary the *games that get played*. Injecting it further
    down would instead corrupt the search's judgement inside a line, degrading
    the very evaluation the training target is built from.
    """
    root = make_mcts(game, 200, seed=3).search(game.initial_state(), add_noise=True)

    perturbed = np.array([c.prior for c in root.children.values()])
    assert perturbed.std() > 0, "root priors should have been perturbed"

    # UniformEvaluator gives every legal move the same prior, so any spread at a
    # deeper node means noise leaked past the root.
    checked = 0
    for child in root.children.values():
        if not child.children:
            continue
        deeper = np.array([c.prior for c in child.children.values()])
        np.testing.assert_allclose(deeper, deeper[0])
        checked += 1
    assert checked > 0, "search never expanded a node below the root"
