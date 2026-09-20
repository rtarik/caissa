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


# ------------------------------------------------- compact storage and persistence


def varied_samples(game, count):
    """Every sample distinguishable, so a round trip that shuffles is caught."""
    shape = (game.input_planes, *game.board_shape)
    samples = []
    for index in range(count):
        policy = np.zeros(COLS, dtype=np.float32)
        policy[index % COLS] = 1.0
        samples.append(Sample(encoded=np.full(shape, index / 100, dtype=np.float16),
                              policy=policy, value=index / count - 0.5))
    return samples


def test_only_the_visited_moves_are_stored(game):
    """A chess policy is 4,672 numbers with about thirty of them non-zero.

    Keeping the zeros is what makes a sample 42 KB instead of 2.7 KB, so the
    storage itself is the thing under test here, not only what comes back out.
    """
    policy = np.zeros(COLS, dtype=np.float32)
    policy[2], policy[5] = 0.25, 0.75
    buffer = ReplayBuffer(capacity=4)
    buffer.extend([Sample(encoded=game.encode(game.initial_state()),
                          policy=policy, value=0.5)])

    _, actions, probabilities, _ = buffer._samples[0]
    assert actions.tolist() == [2, 5]
    assert probabilities.tolist() == [0.25, 0.75]


def test_dense_policies_are_rebuilt_exactly(game):
    """Training reads a full row per sample; sparse storage must be invisible."""
    policy = np.zeros(COLS, dtype=np.float32)
    policy[2], policy[5] = 0.25, 0.75
    buffer = ReplayBuffer(capacity=4)
    buffer.extend([Sample(encoded=game.encode(game.initial_state()),
                          policy=policy, value=0.5)])

    _, policies, values = buffer.sample(3, np.random.default_rng(0))
    assert policies.shape == (3, COLS)
    for row in policies:
        np.testing.assert_array_equal(row.numpy(), policy)
    assert values.tolist() == [0.5, 0.5, 0.5]


def test_a_batch_mixes_policies_of_different_widths(game):
    """Rows are scattered in one go from concatenated runs of different lengths.

    Off-by-one bookkeeping there would write one sample's visits into another's
    row - silently, since every row would still be a valid distribution.
    """
    narrow, wide = np.zeros(COLS, dtype=np.float32), np.zeros(COLS, dtype=np.float32)
    narrow[0] = 1.0
    wide[1], wide[3], wide[6] = 0.5, 0.25, 0.25
    encoded = game.encode(game.initial_state())
    buffer = ReplayBuffer(capacity=4)
    buffer.extend([Sample(encoded=encoded, policy=narrow, value=1.0),
                   Sample(encoded=encoded, policy=wide, value=-1.0)])

    _, policies, values = buffer.sample(32, np.random.default_rng(3))
    for row, value in zip(policies, values):
        expected = narrow if value > 0 else wide
        np.testing.assert_array_equal(row.numpy(), expected)


def test_planes_are_kept_in_half_precision(game):
    """Half precision halves a window again, at a bounded cost in accuracy.

    A plane carrying something continuous - the fifty-move counter as a fraction -
    comes back within a thousandth, which is nothing beside the noise already in a
    Monte Carlo value target. The inexactness is asserted too: if these numbers
    happened to be representable exactly, the tolerance below would prove nothing.
    """
    encoded = np.full((game.input_planes, *game.board_shape), 0.3, dtype=np.float32)
    encoded[0, 0, 0] = 1 / 3
    buffer = ReplayBuffer(capacity=2)
    buffer.extend([Sample(encoded=encoded,
                          policy=np.full(COLS, 1 / COLS, dtype=np.float32), value=0.0)])

    positions, _, _ = buffer.sample(1, np.random.default_rng(0))
    assert positions.dtype == torch.float32
    np.testing.assert_allclose(positions[0].numpy(), encoded, atol=1e-3)
    assert not np.array_equal(positions[0].numpy(), encoded), "nothing was rounded"


def test_a_saved_window_comes_back_the_same(game, tmp_path):
    """The window is run state, not a cache.

    A stage that resumes with an empty buffer trains its first iterations on a
    single iteration's games - correlated, narrow, and measured in Phase 8 as a
    real cost - so a run that is meant to continue writes its window out.
    """
    buffer = ReplayBuffer(capacity=50)
    buffer.extend(varied_samples(game, 20))
    path = buffer.save(tmp_path / "window.npz")

    restored = ReplayBuffer(capacity=50)
    restored.restore(path)

    assert len(restored) == 20
    before = buffer.sample(16, np.random.default_rng(7))
    after = restored.sample(16, np.random.default_rng(7))
    for original, copy in zip(before, after):
        assert torch.equal(original, copy)


def test_a_restored_window_obeys_the_capacity_it_is_poured_into(game, tmp_path):
    """Capacity belongs to the new run's config, and the newest data wins."""
    buffer = ReplayBuffer(capacity=50)
    buffer.extend(varied_samples(game, 20))
    path = buffer.save(tmp_path / "window.npz")

    small = ReplayBuffer(capacity=5)
    small.restore(path)

    assert len(small) == 5
    _, policies, _ = small.sample(64, np.random.default_rng(1))
    kept = set(policies.argmax(dim=1).tolist())
    assert kept == {index % COLS for index in range(15, 20)}


def test_an_empty_window_survives_the_round_trip(game, tmp_path):
    """The first checkpoint of a run is written before any game is played."""
    path = ReplayBuffer(capacity=4).save(tmp_path / "empty.npz")
    restored = ReplayBuffer(capacity=4)
    restored.extend(make_samples(game, 2))
    restored.restore(path)
    assert len(restored) == 0
