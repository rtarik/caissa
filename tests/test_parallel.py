"""Tests for parallel self-play.

The load-bearing test here is equivalence: one worker must reproduce a
sequential run exactly, seed for seed. A fast path that quietly generates
slightly different data is the worst kind of optimisation, because the
difference shows up only as a training run that mysteriously behaves worse.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from caissa.games.connect4 import Connect4
from caissa.learn import LearnConfig, Learner
from caissa.mcts import MCTS, MCTSConfig
from caissa.network import NetworkConfig, NetworkEvaluator, PolicyValueNet
from caissa.parallel import ParallelSelfPlay, single_threaded, split_games
from caissa.selfplay import SelfPlayConfig, generate
from caissa.train import TrainConfig

NET = NetworkConfig(blocks=1, channels=8)
SEARCH = MCTSConfig(simulations=6)
PLAY = SelfPlayConfig(temperature_moves=4)


@pytest.fixture
def game() -> Connect4:
    return Connect4()


# ------------------------------------------------------------------ pure logic


@pytest.mark.parametrize(
    ("games", "workers", "expected"),
    [
        (10, 5, [2, 2, 2, 2, 2]),
        (7, 3, [3, 2, 2]),
        (3, 10, [1, 1, 1]),      # idle workers are dropped, not given zero games
        (1, 8, [1]),
        (100, 6, [17, 17, 17, 17, 16, 16]),
    ],
)
def test_games_are_split_evenly(games, workers, expected):
    assert split_games(games, workers) == expected
    assert sum(split_games(games, workers)) == games


def test_single_threaded_restores_the_previous_setting():
    before = torch.get_num_threads()
    with single_threaded():
        assert torch.get_num_threads() == 1
    assert torch.get_num_threads() == before


def test_single_threaded_restores_after_an_exception():
    before = torch.get_num_threads()
    with pytest.raises(RuntimeError):
        with single_threaded():
            raise RuntimeError("boom")
    assert torch.get_num_threads() == before


def test_pool_must_be_entered(game):
    pool = ParallelSelfPlay(game.name, NET, SEARCH, PLAY, workers=1)
    with pytest.raises(RuntimeError, match="context manager"):
        pool.generate({}, games=1, seed=0)


# --------------------------------------------------------------- equivalence


def sequential(game, net, games, seed):
    """Exactly what one worker does, run in this process.

    One generator drives both the search and the move sampling. Using two would
    consume the random stream differently and the games would diverge - which is
    precisely how this test failed the first time it was written.
    """
    with single_threaded():
        rng = np.random.default_rng(seed)
        mcts = MCTS(game, NetworkEvaluator(net), SEARCH, rng=rng)
        return generate(game, mcts, games, PLAY, rng)


@pytest.mark.slow
@pytest.mark.parametrize("workers", [1, 2])
def test_parallel_reproduces_sequential_runs_exactly(game, workers):
    """The fast path must generate the data the slow path would, in order.

    This covers three things at once: that a worker is equivalent to a sequential
    run, that each worker is given a *distinct* seed derived from the base (or
    the pool would replay one game N times and the extra processes would buy
    nothing), and that results are reassembled in worker order.

    A fast path that silently produces slightly different data is the worst kind
    of optimisation: it shows up only as a training run that behaves worse for no
    visible reason.
    """
    torch.manual_seed(0)
    net = PolicyValueNet.for_game(game, NET)

    expected = []
    for index, count in enumerate(split_games(2, workers)):
        expected.extend(sequential(game, net, count, 100 + index))

    with ParallelSelfPlay(game.name, NET, SEARCH, PLAY, workers=workers) as pool:
        actual = pool.generate(net.state_dict(), games=2, seed=100)

    assert len(actual) == len(expected)
    for got, want in zip(actual, expected):
        np.testing.assert_array_equal(got.encoded, want.encoded)
        np.testing.assert_allclose(got.policy, want.policy, rtol=1e-6, atol=1e-8)
        assert got.value == want.value

    if workers > 1:
        # And the two workers must not have played the same game.
        halves = [sequential(game, net, 1, 100 + i) for i in range(2)]
        assert not np.array_equal(halves[0][0].policy, halves[1][0].policy)


@pytest.mark.slow
def test_learner_runs_an_iteration_in_parallel(game, tmp_path):
    config = LearnConfig(
        games_per_iteration=4,
        train_steps_per_iteration=2,
        buffer_capacity=500,
        min_buffer_before_training=1,
        workers=2,
        network=NET,
        mcts=SEARCH,
        selfplay=PLAY,
        train=TrainConfig(batch_size=8),
    )
    with Learner(game, config, seed=0) as learner:
        first = learner.run_iteration()
        second = learner.run_iteration()

        assert first.positions > 0
        assert second.positions > 0
        assert first.losses is not None
        # A fresh seed per iteration, or every iteration replays the same games.
        assert learner.buffer.__len__() == first.positions + second.positions
        learner.save(tmp_path / "parallel.pt")
    assert learner._pool is None, "pool should be closed on exit"


def test_pool_refuses_to_start_inside_a_worker(game, monkeypatch):
    """Turns an accidental fork bomb into an immediate, explained failure.

    The spawn start method re-imports the caller's main module in every new
    interpreter. If the pool is started at module level rather than behind an
    `if __name__ == "__main__":` guard, each child re-runs that line and starts
    its own pool. It takes the machine down in seconds, and the traceback it
    produces points at multiprocessing internals rather than at the real cause.
    """
    import multiprocessing

    monkeypatch.setattr(multiprocessing, "parent_process", lambda: object())
    with pytest.raises(RuntimeError, match="__main__"):
        ParallelSelfPlay(game.name, NET, SEARCH, PLAY, workers=2).__enter__()
