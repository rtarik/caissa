"""Tests for the replay buffer."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from caissa.games.connect4 import COLS, Connect4
from caissa.replay import ReplayBuffer
from caissa.selfplay import Sample


@pytest.fixture
def game() -> Connect4:
    return Connect4()


def make_samples(game, count, value=1.0):
    state = game.initial_state()
    return [
        Sample(
            encoded=game.encode(state),
            policy=np.full(COLS, 1 / COLS, dtype=np.float32),
            value=value,
        )
        for _ in range(count)
    ]


def test_capacity_is_enforced(game):
    buffer = ReplayBuffer(capacity=10)
    buffer.extend(make_samples(game, 25))
    assert len(buffer) == 10
    assert buffer.full


def test_oldest_samples_are_evicted_first(game):
    """The buffer is a window on *recent* play, so old data must fall out.

    Data from much weaker versions of the network is not merely stale, it is
    actively misleading, and would hold the network back if it never expired.
    """
    buffer = ReplayBuffer(capacity=3)
    buffer.extend(make_samples(game, 3, value=-1.0))
    buffer.extend(make_samples(game, 2, value=1.0))

    _, _, values = buffer.sample(64, np.random.default_rng(0))
    assert set(values.tolist()) <= {-1.0, 1.0}
    assert (values == 1.0).sum() > 0, "newest samples must be present"
    assert len(buffer) == 3


def test_sample_returns_batched_tensors(game):
    buffer = ReplayBuffer(capacity=100)
    buffer.extend(make_samples(game, 50))

    positions, policies, values = buffer.sample(16, np.random.default_rng(1))

    assert positions.shape == (16, game.input_planes, *game.board_shape)
    assert policies.shape == (16, game.action_size)
    assert values.shape == (16,)
    assert positions.dtype == torch.float32
    assert policies.dtype == torch.float32
    assert values.dtype == torch.float32


def test_sampling_an_empty_buffer_is_an_error(game):
    with pytest.raises(ValueError):
        ReplayBuffer(capacity=10).sample(4, np.random.default_rng(0))


def test_sampling_is_random(game):
    """Batches must mix across the buffer, not follow a game's move order.

    Successive positions within a game differ by one piece; a batch drawn from a
    contiguous run is effectively one position repeated, and the gradient step it
    produces is correspondingly narrow.
    """
    buffer = ReplayBuffer(capacity=200)
    rng = np.random.default_rng(0)
    for _ in range(200):
        state = game.initial_state()
        for _ in range(int(rng.integers(1, 10))):
            legal = np.flatnonzero(game.legal_actions(state))
            state = game.apply(state, int(rng.choice(legal)))
        buffer.extend([Sample(
            encoded=game.encode(state),
            policy=np.full(COLS, 1 / COLS, dtype=np.float32),
            value=0.0,
        )])

    first, _, _ = buffer.sample(32, np.random.default_rng(2))
    second, _, _ = buffer.sample(32, np.random.default_rng(3))
    assert not torch.equal(first, second), "two draws produced identical batches"

    # And a single batch must not be one position repeated.
    unique = {tuple(row.flatten().tolist()) for row in first}
    assert len(unique) > 8
