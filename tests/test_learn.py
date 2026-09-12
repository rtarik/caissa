"""Tests for the self-improvement loop and checkpointing."""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn

from caissa.games.connect4 import Connect4
from caissa.learn import LearnConfig, Learner
from caissa.mcts import MCTSConfig
from caissa.network import NetworkConfig
from caissa.selfplay import SelfPlayConfig, generate
from caissa.train import TrainConfig


@pytest.fixture
def game() -> Connect4:
    return Connect4()


def tiny(**overrides) -> LearnConfig:
    config = LearnConfig(
        games_per_iteration=2,
        train_steps_per_iteration=2,
        buffer_capacity=500,
        min_buffer_before_training=1,
        network=NetworkConfig(blocks=1, channels=8),
        mcts=MCTSConfig(simulations=6),
        selfplay=SelfPlayConfig(temperature_moves=4),
        train=TrainConfig(batch_size=8),
    )
    for key, value in overrides.items():
        setattr(config, key, value)
    return config


def test_an_iteration_generates_data_and_trains(game):
    learner = Learner(game, tiny(), seed=0)
    stats = learner.run_iteration()

    assert stats.iteration == 1
    assert stats.positions > 0
    assert len(learner.buffer) == stats.positions
    assert stats.losses is not None
    assert stats.seconds > 0


def test_training_waits_for_enough_data(game):
    """Batches drawn from a nearly empty buffer are not diverse enough to learn
    from, and would just overfit the first handful of games."""
    learner = Learner(game, tiny(min_buffer_before_training=10_000), seed=0)
    stats = learner.run_iteration()

    assert stats.positions > 0
    assert stats.losses is None, "should still be filling the buffer"
    assert "filling buffer" in stats.summary()


def test_iterations_accumulate(game):
    learner = Learner(game, tiny(), seed=0)
    first = learner.run_iteration()
    second = learner.run_iteration()

    assert second.iteration == 2
    assert len(learner.buffer) > first.buffer
    assert len(learner.history) == 2


def test_buffer_capacity_bounds_growth(game):
    learner = Learner(game, tiny(buffer_capacity=20), seed=0)
    for _ in range(3):
        learner.run_iteration()
    assert len(learner.buffer) == 20


# ---------------------------------------------------------- the coupling bug


def test_self_play_after_training_does_not_corrupt_the_network(game):
    """The bug that only exists once self-play and training share a network.

    ``train_step`` leaves the network in training mode. The evaluator holds a
    reference to that same object, so the next self-play game would run its
    batch-norm layers in training mode - using each single position's own
    statistics and overwriting the learned running averages as it goes. Playing
    would degrade the network. Nothing would raise, and the loss would keep
    falling.

    Neither a test of self-play alone nor a test of training alone can catch
    this, because it is created by their interaction.
    """
    learner = Learner(game, tiny(), seed=0)
    learner.run_iteration()
    assert learner.net.training, "train_step is expected to leave training mode on"

    norms = [m for m in learner.net.modules() if isinstance(m, nn.BatchNorm2d)]
    assert norms, "no batch-norm layers, so this test proves nothing"
    before = [(m.running_mean.clone(), m.running_var.clone()) for m in norms]

    generate(game, learner._mcts(), games=1, rng=np.random.default_rng(1))

    for module, (mean, var) in zip(norms, before):
        assert torch.equal(module.running_mean, mean), "self-play moved running_mean"
        assert torch.equal(module.running_var, var), "self-play moved running_var"


# ------------------------------------------------------------------ checkpoints


def test_checkpoint_round_trip_restores_weights(game, tmp_path):
    learner = Learner(game, tiny(), seed=0)
    learner.run_iteration()
    path = learner.save(tmp_path / "check.pt")
    assert path.exists()

    restored = Learner(game, tiny(), seed=1)
    # Confirm they genuinely differ before loading, or this proves nothing.
    assert not torch.equal(
        next(iter(learner.net.state_dict().values())),
        next(iter(restored.net.state_dict().values())),
    )

    restored.load(path)
    assert restored.iteration == learner.iteration
    for key, tensor in learner.net.state_dict().items():
        assert torch.equal(restored.net.state_dict()[key], tensor)


def test_checkpoint_restores_optimizer_state(game, tmp_path):
    """AdamW carries per-parameter moment estimates.

    Resuming without them restarts the optimiser cold, which shows up as a
    visible stumble in training immediately after every resume.
    """
    learner = Learner(game, tiny(), seed=0)
    learner.run_iteration()
    path = learner.save(tmp_path / "check.pt")

    restored = Learner(game, tiny(), seed=1)
    restored.load(path)

    original_state = learner.optimizer.state_dict()["state"]
    assert original_state, "optimiser had no state to restore"
    assert restored.optimizer.state_dict()["state"].keys() == original_state.keys()


def test_checkpoints_are_game_specific(game, tmp_path):
    learner = Learner(game, tiny(), seed=0)
    path = learner.save(tmp_path / "check.pt")

    class Impostor(Connect4):
        name = "gomoku"

    with pytest.raises(ValueError, match="gomoku"):
        Learner(Impostor(), tiny()).load(path)
