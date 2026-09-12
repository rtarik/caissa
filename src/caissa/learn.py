"""The self-improvement loop.

    play games with the current network
        -> train the network on what search found in them
            -> play games with the improved network
                -> ...

The loop is the algorithm. Each half is unremarkable on its own: search is a
planning routine, training is supervised learning on a batch of labelled
positions. What makes it reinforcement learning is that the labels come from the
learner's own behaviour, so improving the network changes the data, which changes
what the network learns next.

That feedback is also what makes it fragile. In supervised learning the dataset
is fixed and a bug shows up as a loss that will not fall. Here the network can
happily minimise a loss computed against targets that are themselves wrong, or
narrow, or drifting - and every number on the console will look fine. Phase 3
exists because of this: strength has to be measured against something outside the
loop, not inferred from the loop's own reports.

**On devices.** Everything here runs on CPU, which is measured to be the right
choice while self-play is sequential. Per iteration at the default settings,
self-play is ~75,000 single-position evaluations (72s on CPU, 113s on the GPU,
since one small tensor at a time does not repay the round trip) against ~200
training steps (3.6s on CPU, 0.2s on the GPU). Self-play dominates by an order of
magnitude, so the faster self-play wins. That inverts once Phase 2b batches
self-play across parallel games.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch

from caissa.mcts import MCTS, MCTSConfig
from caissa.network import NetworkConfig, NetworkEvaluator, PolicyValueNet
from caissa.replay import ReplayBuffer
from caissa.selfplay import SelfPlayConfig, generate
from caissa.train import Losses, TrainConfig, make_optimizer, train_step


@dataclass
class LearnConfig:
    games_per_iteration: int = 40
    train_steps_per_iteration: int = 200
    #: Positions, not games. Sized so it spans several iterations' worth of play,
    #: which is what gives the window its smoothing effect.
    buffer_capacity: int = 60_000
    #: Wait until there is enough data for batches to be meaningfully diverse.
    min_buffer_before_training: int = 4_000

    network: NetworkConfig = field(default_factory=NetworkConfig)
    mcts: MCTSConfig = field(default_factory=MCTSConfig)
    selfplay: SelfPlayConfig = field(default_factory=SelfPlayConfig)
    train: TrainConfig = field(default_factory=TrainConfig)


@dataclass
class IterationStats:
    iteration: int
    games: int
    positions: int
    buffer: int
    losses: Losses | None
    seconds: float

    def summary(self) -> str:
        head = (f"iter {self.iteration:>3}  {self.games:>4} games  "
                f"{self.positions:>6} new  buffer {self.buffer:>6}  "
                f"{self.seconds:>5.1f}s")
        if self.losses is None:
            return head + "   (filling buffer)"
        return (head + f"   loss {self.losses.total:.4f}"
                f"  policy {self.losses.policy:.4f}  value {self.losses.value:.4f}")


class Learner:
    """Owns the network, the optimiser and the replay buffer across iterations."""

    def __init__(self, game, config: LearnConfig | None = None,
                 seed: int | None = None):
        self.game = game
        self.config = config or LearnConfig()
        self.rng = np.random.default_rng(seed)

        self.net = PolicyValueNet.for_game(game, self.config.network)
        self.optimizer = make_optimizer(self.net, self.config.train)
        self.buffer = ReplayBuffer(self.config.buffer_capacity)
        self.iteration = 0
        self.history: list[IterationStats] = []

    def _mcts(self) -> MCTS:
        # Rebuilt each iteration so it always wraps the current network. The
        # evaluator keeps a reference rather than a copy, so this is about
        # clarity rather than correctness.
        return MCTS(self.game, NetworkEvaluator(self.net), self.config.mcts,
                    rng=self.rng)

    def run_iteration(self) -> IterationStats:
        started = time.perf_counter()
        self.iteration += 1

        samples = generate(
            self.game, self._mcts(), self.config.games_per_iteration,
            self.config.selfplay, self.rng,
        )
        self.buffer.extend(samples)

        losses = None
        if len(self.buffer) >= self.config.min_buffer_before_training:
            totals = [0.0, 0.0, 0.0]
            for _ in range(self.config.train_steps_per_iteration):
                batch = self.buffer.sample(self.config.train.batch_size, self.rng)
                step = train_step(self.net, self.optimizer, batch)
                totals[0] += step.total
                totals[1] += step.policy
                totals[2] += step.value
            n = self.config.train_steps_per_iteration
            losses = Losses(totals[0] / n, totals[1] / n, totals[2] / n)

        stats = IterationStats(
            iteration=self.iteration,
            games=self.config.games_per_iteration,
            positions=len(samples),
            buffer=len(self.buffer),
            losses=losses,
            seconds=time.perf_counter() - started,
        )
        self.history.append(stats)
        return stats

    # ------------------------------------------------------------- persistence

    def save(self, path: str | Path) -> Path:
        """Write a checkpoint.

        The optimiser state goes in alongside the weights. AdamW carries
        per-parameter moment estimates, and resuming without them restarts the
        optimiser cold, which shows up as a visible stumble in training right
        after every resume.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "iteration": self.iteration,
                "game": self.game.name,
                "network": self.net.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "config": json.loads(json.dumps(asdict(self.config))),
            },
            path,
        )
        return path

    def load(self, path: str | Path) -> None:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        if checkpoint["game"] != self.game.name:
            raise ValueError(
                f"checkpoint is for {checkpoint['game']}, not {self.game.name}"
            )
        self.net.load_state_dict(checkpoint["network"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.iteration = checkpoint["iteration"]
