"""Self-play across several processes.

Self-play is ~98% network evaluation and ~2% tree walking, measured on this
machine: 952 microseconds per simulation, of which 24 are spent descending and
expanding the tree. That number decides the design. Batching leaves across
concurrent games - the obvious optimisation - can at best remove the network
cost, leaving the tree rate of ~42,000 evaluations per second as a ceiling.
Running whole games in parallel processes lifts *both* halves, is far simpler,
and gets close to the same place.

Two things make it work, neither of them obvious:

**One torch thread per worker.** PyTorch splits individual operations across
threads by default. For a single 6x7 position that is pure loss - coordinating
ten threads costs more than the arithmetic - and measured here it is 2.5x
*slower* than one thread. Left unset, ten workers would also each try to use ten
threads on ten cores, and spend their time fighting each other.

**Processes, not threads.** Python's global interpreter lock means threads cannot
run the tree search concurrently. Processes each get their own interpreter. The
cost is that everything crossing the boundary must be pickled, which is why
workers are handed a network *state dict* and rebuild the model themselves rather
than receiving a live object.

.. warning::

   Any script that starts a pool must guard its entry point::

       if __name__ == "__main__":
           main()

   The spawn start method re-imports the main module in each new interpreter. If
   starting the pool is module-level code, every child re-runs it and starts its
   own pool - a fork bomb that will take the machine down in seconds. This is
   easy to do by accident, so :class:`ParallelSelfPlay` refuses to start a pool
   from inside a worker process and says why.
"""

from __future__ import annotations

import contextlib
import multiprocessing as mp
from dataclasses import dataclass

import numpy as np
import torch

from caissa.games import GAMES
from caissa.mcts import MCTS, MCTSConfig
from caissa.network import NetworkConfig, NetworkEvaluator, PolicyValueNet
from caissa.selfplay import Sample, SelfPlayConfig, generate


@contextlib.contextmanager
def single_threaded():
    """Run torch on one thread inside this block, then restore the setting.

    For a single small position, PyTorch's intra-op threading costs more than it
    saves - measured at 2.5x slower with ten threads than with one. Batched work
    such as the training step wants the threads back, hence the restore rather
    than a global setting.
    """
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


@dataclass
class WorkerTask:
    game_name: str
    state_dict: dict
    network: NetworkConfig
    mcts: MCTSConfig
    selfplay: SelfPlayConfig
    games: int
    seed: int


def _run_worker(task: WorkerTask) -> list[Sample]:
    """Play ``task.games`` games in this process and return their samples.

    Deliberately a plain function at module scope: the spawn start method has to
    import it by name in the fresh interpreter, so a closure or a bound method
    would not survive the trip.
    """
    torch.set_num_threads(1)

    game = GAMES[task.game_name]()
    net = PolicyValueNet.for_game(game, task.network)
    net.load_state_dict(task.state_dict)

    # One generator drives both the search and the move sampling, exactly as the
    # sequential path does, so a single worker reproduces a sequential run seed
    # for seed. That equivalence is what makes the fast path testable.
    rng = np.random.default_rng(task.seed)
    mcts = MCTS(game, NetworkEvaluator(net), task.mcts, rng=rng)
    return generate(game, mcts, task.games, task.selfplay, rng)


def split_games(games: int, workers: int) -> list[int]:
    """Divide games as evenly as possible, dropping workers with nothing to do."""
    base, extra = divmod(games, workers)
    counts = [base + (1 if i < extra else 0) for i in range(workers)]
    return [c for c in counts if c > 0]


class ParallelSelfPlay:
    """A reusable pool of self-play workers.

    The pool is created once and kept, because spawning a process means importing
    torch in a fresh interpreter - a second or two each. Rebuilding it every
    iteration would cost more than the parallelism saves.

    Used as a context manager so the processes are always cleaned up::

        with ParallelSelfPlay("connect4", net_config, mcts_config, sp_config) as pool:
            samples = pool.generate(net.state_dict(), games=100, seed=0)
    """

    def __init__(self, game_name: str, network: NetworkConfig, mcts: MCTSConfig,
                 selfplay: SelfPlayConfig, workers: int | None = None):
        self.game_name = game_name
        self.network = network
        self.mcts = mcts
        self.selfplay = selfplay
        # Default to the performance cores. Counting every logical core would
        # put workers on the efficiency cores, which are several times slower and
        # would hold up each iteration's slowest worker.
        self.workers = workers or max(1, (mp.cpu_count() or 2) - 4)
        self._pool: mp.pool.Pool | None = None

    def __enter__(self) -> ParallelSelfPlay:
        # Starting a pool from inside a worker means the caller's module-level
        # code is being re-executed by the spawn import, and every child is about
        # to start its own pool. Failing here turns a fork bomb into a message.
        if mp.parent_process() is not None:
            raise RuntimeError(
                "ParallelSelfPlay was started from inside a worker process. "
                "This usually means the calling script starts the pool at module "
                "level; put it behind `if __name__ == \"__main__\":` so the spawn "
                "re-import does not run it again."
            )
        self._pool = mp.get_context("spawn").Pool(processes=self.workers)
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def close(self) -> None:
        if self._pool is not None:
            self._pool.terminate()
            self._pool.join()
            self._pool = None

    def generate(self, state_dict: dict, games: int, seed: int) -> list[Sample]:
        """Play ``games`` games across the pool and return all their samples.

        Each worker gets a distinct seed derived from ``seed``, so a run is
        reproducible given the same worker count. Change the worker count and the
        games change too - the split is different - which is worth knowing before
        wondering why results moved.
        """
        if self._pool is None:
            raise RuntimeError("use ParallelSelfPlay as a context manager")

        state_dict = {k: v.detach().cpu() for k, v in state_dict.items()}
        tasks = [
            WorkerTask(
                game_name=self.game_name,
                state_dict=state_dict,
                network=self.network,
                mcts=self.mcts,
                selfplay=self.selfplay,
                games=count,
                seed=seed + index,
            )
            for index, count in enumerate(split_games(games, self.workers))
        ]

        samples: list[Sample] = []
        for result in self._pool.map(_run_worker, tasks):
            samples.extend(result)
        return samples
