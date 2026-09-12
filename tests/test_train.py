"""Tests for the training step."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from caissa.games.connect4 import COLS, Connect4
from caissa.network import NetworkConfig, PolicyValueNet
from caissa.train import TrainConfig, alphazero_loss, make_optimizer, train_step


@pytest.fixture
def game() -> Connect4:
    return Connect4()


@pytest.fixture
def net(game) -> PolicyValueNet:
    torch.manual_seed(0)
    return PolicyValueNet.for_game(game, NetworkConfig(blocks=1, channels=16))


def fixed_batch(game, size=32, seed=0):
    """A batch of *distinct* positions with random targets.

    Distinctness matters. Repeat a position with conflicting value labels and the
    value loss acquires a floor of its own, because no function can output two
    numbers for one input - it converges to their mean instead. That is exactly
    what happens in real self-play data, where the same opening appears in many
    games with different outcomes, and it is the desired behaviour: the value
    head should learn the *expected* result, not memorise individual games. Here
    it would just make the test's target unreachable, so the duplicates go.
    """
    rng = np.random.default_rng(seed)
    states, seen = [], set()
    while len(states) < size:
        state = game.initial_state()
        for _ in range(int(rng.integers(1, 10))):
            if game.terminal_value(state) is not None:
                break
            legal = np.flatnonzero(game.legal_actions(state))
            state = game.apply(state, int(rng.choice(legal)))
        key = state.board.tobytes()
        if key in seen:
            continue
        seen.add(key)
        states.append(state)

    positions = torch.from_numpy(np.stack([game.encode(s) for s in states]))
    policy = torch.from_numpy(rng.dirichlet(np.ones(COLS), size=size).astype(np.float32))
    value = torch.from_numpy(rng.choice([-1.0, 0.0, 1.0], size=size).astype(np.float32))
    return positions, policy, value


def test_policy_loss_bottoms_out_at_the_target_entropy(game, net):
    """Cross-entropy against a soft target cannot reach zero.

    Its minimum is the entropy of the target itself. Expecting a policy loss to
    fall toward zero - as it would with one-hot labels - leads to concluding that
    a perfectly healthy run has stalled.
    """
    target = torch.tensor([[0.5, 0.25, 0.25, 0.0, 0.0, 0.0, 0.0]])
    perfect_logits = torch.log(target.clamp_min(1e-9))

    _, policy_loss, _ = alphazero_loss(
        perfect_logits, torch.zeros(1), target, torch.zeros(1)
    )
    entropy = -(target * torch.log(target.clamp_min(1e-9))).sum()
    assert policy_loss.item() == pytest.approx(entropy.item(), abs=1e-5)


def test_value_loss_is_zero_on_exact_predictions(game):
    value = torch.tensor([1.0, -1.0, 0.0])
    _, _, value_loss = alphazero_loss(
        torch.zeros(3, COLS), value, torch.full((3, COLS), 1 / COLS), value
    )
    assert value_loss.item() == pytest.approx(0.0)


def test_training_converges_toward_the_irreducible_floor(game, net):
    """Training must reach the best loss the targets actually allow.

    The floor is not zero. The value term can vanish, but the policy term bottoms
    out at the mean entropy of the target distributions, so a ratio-based
    threshold would be arbitrary and would fail for reasons unrelated to whether
    learning worked. Measuring the gap to the floor is the meaningful check.
    """
    batch = fixed_batch(game)
    _, target_policy, _ = batch
    floor = -(target_policy * torch.log(target_policy.clamp_min(1e-9))).sum(1).mean()

    optimizer = make_optimizer(net, TrainConfig(learning_rate=3e-3))
    first = train_step(net, optimizer, batch)
    for _ in range(300):
        last = train_step(net, optimizer, batch)

    assert first.policy > floor.item() + 0.2, "started already at the floor"
    assert last.policy < floor.item() + 0.1, (
        f"policy loss {last.policy:.4f} did not approach floor {floor.item():.4f}"
    )
    assert last.value < 0.05
    assert last.total < first.total


def test_weight_decay_is_applied(game, net):
    """L2 regularisation goes through the optimiser, not the loss function."""
    optimizer = make_optimizer(net, TrainConfig(weight_decay=0.123))
    assert all(g["weight_decay"] == 0.123 for g in optimizer.param_groups)


def test_step_leaves_the_network_in_training_mode(game, net):
    """Documents the coupling that NetworkEvaluator has to defend against."""
    net.eval()
    train_step(net, make_optimizer(net, TrainConfig()), fixed_batch(game, 8))
    assert net.training
