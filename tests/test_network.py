"""Tests for the policy/value network and its evaluator adapter."""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn

from caissa.evaluator import Evaluator, UniformEvaluator
from caissa.games.connect4 import COLS, ROWS, Connect4
from caissa.mcts import MCTS, MCTSConfig
from caissa.network import (
    NetworkConfig,
    NetworkEvaluator,
    PolicyValueNet,
    ResidualBlock,
)


@pytest.fixture
def game() -> Connect4:
    return Connect4()


@pytest.fixture
def net(game) -> PolicyValueNet:
    torch.manual_seed(0)
    return PolicyValueNet.for_game(game, NetworkConfig(blocks=2, channels=16))


def play(game: Connect4, *columns: int):
    state = game.initial_state()
    for column in columns:
        state = game.apply(state, column)
    return state


def batch(game, states) -> torch.Tensor:
    return torch.from_numpy(np.stack([game.encode(s) for s in states]))


# ------------------------------------------------------------------- the network


def test_forward_shapes(game, net):
    logits, value = net(batch(game, [game.initial_state(), play(game, 3)]))
    assert logits.shape == (2, game.action_size)
    assert value.shape == (2,), "value is squeezed, so callers need not remember to"


def test_value_is_bounded_to_the_outcome_range(game, net):
    """tanh keeps the value inside [-1, 1], the only range a game can produce.

    An untrained network happens to emit small numbers anyway, so checking a
    fresh one proves nothing. The final layer is scaled up hard first, to force
    the squashing function to be the thing doing the work.
    """
    with torch.no_grad():
        final = [m for m in net.value_head if isinstance(m, nn.Linear)][-1]
        final.weight *= 500.0
        final.bias += 200.0

    states = [play(game, *cols) for cols in [(), (3,), (3, 3), (0, 1, 2)]]
    _, value = net(batch(game, states))

    assert value.abs().max() > 0.9, "the scaling should have saturated the output"
    assert value.min() >= -1.0 and value.max() <= 1.0


def test_residual_block_passes_its_input_through():
    """With the convolutions zeroed, a residual block must be the identity.

    That is the whole point of the skip connection: each block learns a
    *correction* to its input rather than a replacement for it, so gradients
    reach the early layers and adding depth does not make training harder.
    Without the skip, zeroed convolutions would output zeros instead.
    """
    block = ResidualBlock(4)
    with torch.no_grad():
        nn.init.zeros_(block.conv1.weight)
        nn.init.zeros_(block.conv2.weight)
    block.eval()

    x = torch.rand(2, 4, 5, 5)  # strictly positive, so the final relu is identity
    torch.testing.assert_close(block(x), x)


def test_is_game_agnostic(game):
    """Nothing in the network is specific to Connect 4's shape."""

    class Hypothetical:
        input_planes, board_shape, action_size = 7, (5, 9), 23

    other = Hypothetical()
    net = PolicyValueNet(other.input_planes, other.board_shape, other.action_size)
    logits, value = net(torch.randn(3, 7, 5, 9))
    assert logits.shape == (3, 23)
    assert value.shape == (3,)


def test_both_heads_receive_gradients(game, net):
    """Gradient must reach the shared trunk from both heads.

    The loss here is the real one from Phase 2 - cross-entropy on the policy
    against a search-derived distribution, squared error on the value. Note that
    a naive ``logits.sum().backward()`` gives *exactly zero* gradient at the
    trunk: batch normalisation makes its output invariant to shifts in its
    input, so a loss that depends only on the normalised statistics carries no
    signal backwards. Loss choice is not incidental to whether learning happens.
    """
    states = [
        play(game, *cols)
        for cols in [(), (3,), (3, 3), (0, 1), (2,), (4, 4), (1, 1), (5,)]
    ]
    logits, value = net(batch(game, states))

    target_policy = torch.zeros(len(states), game.action_size)
    target_policy[:, 0] = 1.0
    target_value = torch.ones(len(states))

    policy_loss = -(target_policy * torch.log_softmax(logits, dim=1)).sum(1).mean()
    value_loss = torch.nn.functional.mse_loss(value, target_value)
    (policy_loss + value_loss).backward()

    trunk_grad = net.stem[0].weight.grad
    assert trunk_grad is not None and trunk_grad.abs().sum() > 0, "trunk got no signal"
    for head in (net.policy_head, net.value_head):
        last = [m for m in head if isinstance(m, nn.Linear)][-1]
        assert last.weight.grad is not None and last.weight.grad.abs().sum() > 0


# ----------------------------------------------------------------- the evaluator


def test_evaluator_satisfies_the_protocol(net):
    assert isinstance(NetworkEvaluator(net), Evaluator)
    assert isinstance(UniformEvaluator(), Evaluator)


def test_priors_are_a_distribution_over_legal_moves_only(game, net):
    evaluator = NetworkEvaluator(net)
    state = play(game, *([0] * ROWS))  # column 0 full

    priors, value = evaluator.evaluate(game, state)

    assert priors.shape == (game.action_size,)
    assert priors[0] == 0.0, "a full column must get exactly zero, not merely little"
    assert priors[1:].sum() == pytest.approx(1.0)
    assert -1.0 <= value <= 1.0


def test_untrained_priors_are_roughly_uniform(game, net):
    """An untrained network should be ignorant, not confidently wrong.

    Default initialisation gives small logits, so the softmax starts near
    uniform. That matters: a fresh network that was strongly opinionated would
    steer early self-play down arbitrary lines before it knew anything.
    """
    priors, _ = NetworkEvaluator(net).evaluate(game, game.initial_state())
    assert priors.max() < 0.35, f"suspiciously opinionated for an untrained net: {priors}"


def test_evaluation_is_deterministic(game, net):
    evaluator = NetworkEvaluator(net)
    state = play(game, 3, 3, 4)
    first_priors, first_value = evaluator.evaluate(game, state)
    second_priors, second_value = evaluator.evaluate(game, state)
    np.testing.assert_array_equal(first_priors, second_priors)
    assert first_value == second_value


# ------------------------------------------------- the batch-normalisation trap


def test_evaluator_forces_eval_mode(game, net):
    net.train()
    evaluator = NetworkEvaluator(net)
    assert not evaluator.net.training


def test_evaluating_does_not_mutate_the_network(game, net):
    """Searching must not change the weights or the running statistics.

    In training mode, batch-norm layers update their running averages on every
    forward pass. Since search calls the network hundreds of times per move, a
    missing ``eval()`` means the act of *playing* steadily corrupts the network -
    with no error, and no obviously wrong output, until strength collapses.
    """
    net.train()  # deliberately wrong on the way in
    evaluator = NetworkEvaluator(net)

    norms = [m for m in net.modules() if isinstance(m, nn.BatchNorm2d)]
    assert norms, "no batch-norm layers, so this test proves nothing"
    before = [(m.running_mean.clone(), m.running_var.clone()) for m in norms]

    for columns in [(), (3,), (3, 3), (0, 1), (2, 2, 4)]:
        evaluator.evaluate(game, play(game, *columns))

    for module, (mean, var) in zip(norms, before):
        assert torch.equal(module.running_mean, mean)
        assert torch.equal(module.running_var, var)


def test_training_mode_really_would_give_different_answers(game, net):
    """The eval() call is load-bearing, not defensive tidiness."""
    net.train()
    with torch.no_grad():  # let the running statistics diverge from the batch's
        net(torch.randn(32, game.input_planes, *game.board_shape))

    x = batch(game, [play(game, 3, 3, 4)])
    net.eval()
    with torch.no_grad():
        eval_logits, eval_value = net(x)
    net.train()
    with torch.no_grad():
        train_logits, train_value = net(x)

    assert not torch.allclose(eval_logits, train_logits)
    assert not torch.allclose(eval_value, train_value)


# ------------------------------------------------------------------ integration


def test_search_works_with_the_network(game, net):
    """An untrained network must not stop search from seeing a win in one.

    Near-uniform priors make this equivalent to the knowledge-free case, so a
    failure here points at the evaluator adapter rather than at the search.
    """
    mcts = MCTS(
        game,
        NetworkEvaluator(net),
        MCTSConfig(simulations=150),
        rng=np.random.default_rng(0),
    )
    policy, value = mcts.run(play(game, 0, 1, 0, 1, 0, 1), temperature=0.0)
    assert policy.argmax() == 0
    assert value > 0.5


def test_network_and_uniform_evaluators_are_interchangeable(game, net):
    """Search does not care where its opinions come from."""
    state = play(game, 0, 1, 0, 1, 0)
    for evaluator in (UniformEvaluator(), NetworkEvaluator(net)):
        mcts = MCTS(game, evaluator, MCTSConfig(simulations=600),
                    rng=np.random.default_rng(1))
        policy, _ = mcts.run(state, temperature=0.0)
        assert policy.argmax() == 0, f"{type(evaluator).__name__} missed the block"


def test_eval_mode_is_reasserted_on_every_call(game, net):
    """Training mode can be switched on *after* the evaluator was built.

    The evaluator holds a reference to the network, and the training step puts
    that same object back into training mode. Checking eval() only at
    construction would leave the very sequence that occurs in the real loop -
    build evaluator, train, play - unprotected.
    """
    evaluator = NetworkEvaluator(net)
    net.train()  # as a training step leaves it

    norms = [m for m in net.modules() if isinstance(m, nn.BatchNorm2d)]
    before = [(m.running_mean.clone(), m.running_var.clone()) for m in norms]

    evaluator.evaluate(game, play(game, 3, 3))

    assert not net.training
    for module, (mean, var) in zip(norms, before):
        assert torch.equal(module.running_mean, mean)
        assert torch.equal(module.running_var, var)
