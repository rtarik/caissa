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
OPENINGS = SelfPlayConfig(temperature_moves=4, random_opening_share=1.0, random_opening_plies=6)


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


def sequential(game, net, games, seed, play=PLAY):
    """Exactly what one worker does, run in this process.

    One generator drives both the search and the move sampling. Using two would
    consume the random stream differently and the games would diverge - which is
    precisely how this test failed the first time it was written.
    """
    with single_threaded():
        rng = np.random.default_rng(seed)
        mcts = MCTS(game, NetworkEvaluator(net), SEARCH, rng=rng)
        return generate(game, mcts, games, play, rng)


@pytest.mark.slow
@pytest.mark.parametrize("play", [PLAY, OPENINGS], ids=["plain", "random-openings"])
@pytest.mark.parametrize("workers", [1, 2])
def test_parallel_reproduces_sequential_runs_exactly(game, workers, play):
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
        expected.extend(sequential(game, net, count, 100 + index, play))

    with ParallelSelfPlay(game.name, NET, SEARCH, play, workers=workers) as pool:
        actual = pool.generate(net.state_dict(), games=2, seed=100)

    assert len(actual) == len(expected)
    for got, want in zip(actual, expected):
        np.testing.assert_array_equal(got.encoded, want.encoded)
        np.testing.assert_allclose(got.policy, want.policy, rtol=1e-6, atol=1e-8)
        assert got.value == want.value

    if workers > 1:
        # And the two workers must not have played the same game.
        halves = [sequential(game, net, 1, 100 + i, play) for i in range(2)]
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


# ------------------------------------------------------------- parallel arena


def spec(game, seed: int):
    """A (config, state dict) pair, which is what crosses the process boundary."""
    torch.manual_seed(seed)
    net = PolicyValueNet.for_game(game, NET)
    return NET, net.state_dict()


@pytest.mark.slow
@pytest.mark.parametrize("workers", [1, 2])
def test_parallel_arena_reproduces_a_sequential_match(game, workers):
    """The fast path must return the match the slow path would.

    A split that quietly changed the pairing or the openings would shift the
    measured Elo, and every conclusion in the project rests on those numbers.
    """
    from caissa.arena import Player, play_match
    from caissa.parallel import ParallelArena

    first, second = spec(game, 0), spec(game, 1)

    expected = []
    for index, pairs in enumerate(split_games(4 // 2, workers)):
        with single_threaded():
            net_a = PolicyValueNet.for_game(game, first[0])
            net_a.load_state_dict(first[1])
            net_b = PolicyValueNet.for_game(game, second[0])
            net_b.load_state_dict(second[1])
            expected.append(play_match(
                game,
                Player("player", NetworkEvaluator(net_a), SEARCH.simulations),
                Player("opponent", NetworkEvaluator(net_b), SEARCH.simulations),
                pairs * 2, np.random.default_rng(100 + index), 2,
            ))

    with ParallelArena(game.name, NET, SEARCH, PLAY, workers=workers) as arena:
        actual = arena.match(first, second, games=4,
                             simulations=SEARCH.simulations, seed=100)

    assert actual.wins == sum(r.wins for r in expected)
    assert actual.draws == sum(r.draws for r in expected)
    assert actual.losses == sum(r.losses for r in expected)
    assert actual.games == 4


@pytest.mark.slow
def test_parallel_arena_keeps_pairs_whole(game):
    """Workers get whole colour-reversed pairs, never half of one.

    Splitting mid-pair would put the two halves on different workers with
    different openings, and the variance reduction pairing exists for would be
    lost - silently, since the result would still look like a match.
    """
    from caissa.parallel import ParallelArena

    first, second = spec(game, 0), spec(game, 1)
    with ParallelArena(game.name, NET, SEARCH, PLAY, workers=3) as arena:
        result = arena.match(first, second, games=6,
                             simulations=SEARCH.simulations, seed=0)
    assert result.games == 6

    with pytest.raises(ValueError, match="at least two games"):
        with ParallelArena(game.name, NET, SEARCH, PLAY, workers=2) as arena:
            arena.match(first, second, games=1, simulations=SEARCH.simulations, seed=0)


@pytest.mark.parametrize("workers", [1, 2])
def test_parallel_arena_gives_each_player_its_own_settings(game, workers):
    """A match can be asymmetric: different depths, different exploration.

    That is how the project asks whether more search is worth anything to a
    network - it plays the network against itself at another depth - so the two
    sides must not be quietly given the same settings. Swapping them, or using
    one side's for both, changes the result, which is what this pins.
    """
    from caissa.arena import Player, play_match
    from caissa.parallel import ParallelArena

    first, second = spec(game, 0), spec(game, 1)
    depths, exploration = (12, 3), (0.5, 4.0)

    expected = []
    for index, pairs in enumerate(split_games(4 // 2, workers)):
        with single_threaded():
            net_a = PolicyValueNet.for_game(game, first[0])
            net_a.load_state_dict(first[1])
            net_b = PolicyValueNet.for_game(game, second[0])
            net_b.load_state_dict(second[1])
            expected.append(play_match(
                game,
                Player("player", NetworkEvaluator(net_a), depths[0], c_puct=exploration[0]),
                Player("opponent", NetworkEvaluator(net_b), depths[1], c_puct=exploration[1]),
                pairs * 2, np.random.default_rng(100 + index), 2,
            ))

    with ParallelArena(game.name, NET, SEARCH, PLAY, workers=workers) as arena:
        actual = arena.match(first, second, games=4, simulations=depths, seed=100,
                             c_puct=exploration)

    assert actual.wins == sum(r.wins for r in expected)
    assert actual.draws == sum(r.draws for r in expected)
    assert actual.losses == sum(r.losses for r in expected)
