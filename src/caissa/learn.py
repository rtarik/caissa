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

**On devices, the two halves want opposite things.** Self-play evaluates one
small position at a time, where the GPU round trip costs more than the work, so
it runs on CPU - spread across worker processes, one torch thread each. Training
is batched, where the GPU is ~39x faster (59,757 against 1,525 positions per
second at batch 512; convolutions over a 6x7 board have too little arithmetic per
byte to use a CPU well, and the backward pass is six times the forward).

So the network lives on the GPU and a CPU mirror is made for self-play. Before
parallelism the split was 72s of play to 3.6s of training; afterwards it was 11s
to 72s, and the bottleneck had moved to the other half entirely. Measure again
after any change that shifts the balance.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch

from caissa.arena import MatchResult, Player, play_match
from caissa.mcts import MCTS, MCTSConfig
from caissa.parallel import ParallelSelfPlay, single_threaded
from caissa.network import (
    NetworkConfig,
    NetworkEvaluator,
    PolicyValueNet,
    best_device,
)
from caissa.replay import ReplayBuffer
from caissa.selfplay import SelfPlayConfig, generate
from caissa.train import Losses, TrainConfig, make_optimizer, train_step


@dataclass
class GateConfig:
    """Champion-challenger promotion, as in AlphaGo Zero.

    A new network only replaces the one generating self-play data if it beats it
    by a margin. The protection is real - a bad iteration cannot poison the data
    for every iteration after it - but it is not free, and AlphaZero dropped it
    entirely in favour of always using the latest network.

    The trade is worth understanding. Gating costs a match every iteration, and
    the match has to be large enough to mean something: AlphaGo Zero used 400
    games at a 55% threshold, which is almost exactly the sample needed to
    resolve 35 Elo at 95% confidence. A cheap 40-game gate resolves only about
    110 Elo, so it will reject genuine improvements smaller than that and let the
    agent stall while reporting nothing wrong.
    """

    enabled: bool = False
    games: int = 40
    #: Score the challenger must exceed. 0.5 promotes on any edge, including one
    #: indistinguishable from noise.
    threshold: float = 0.55
    simulations: int = 50
    opening_plies: int = 2


@dataclass
class LearnConfig:
    games_per_iteration: int = 40
    train_steps_per_iteration: int = 200
    #: Positions, not games. Sized so it spans several iterations' worth of play,
    #: which is what gives the window its smoothing effect.
    buffer_capacity: int = 60_000
    #: Wait until there is enough data for batches to be meaningfully diverse.
    min_buffer_before_training: int = 4_000
    #: Self-play worker processes. 1 (or None) runs in this process, which is
    #: slower but far easier to debug. See src/caissa/parallel.py.
    workers: int | None = None
    #: Where the gradient steps happen. None auto-detects. Training is batched,
    #: so the GPU wins by ~39x here; self-play is one position at a time, so it
    #: stays on CPU regardless. See the device note in this module's docstring.
    train_device: str | None = None

    network: NetworkConfig = field(default_factory=NetworkConfig)
    mcts: MCTSConfig = field(default_factory=MCTSConfig)
    selfplay: SelfPlayConfig = field(default_factory=SelfPlayConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    gate: GateConfig = field(default_factory=GateConfig)


@dataclass
class IterationStats:
    iteration: int
    games: int
    positions: int
    buffer: int
    losses: Losses | None
    seconds: float
    #: Result of the promotion match, when gating is on.
    gate: MatchResult | None = None
    promoted: bool = False
    #: Split out so it stays visible which half is the bottleneck. It moves:
    #: self-play dominated by 10x before parallelism, much less so after.
    selfplay_seconds: float = 0.0
    train_seconds: float = 0.0

    def summary(self) -> str:
        head = (f"iter {self.iteration:>3}  {self.games:>4} games  "
                f"{self.positions:>6} new  buffer {self.buffer:>6}  "
                f"{self.seconds:>5.1f}s "
                f"(play {self.selfplay_seconds:>4.1f} train {self.train_seconds:>4.1f})")
        if self.losses is None:
            return head + "   (filling buffer)"
        text = (head + f"   loss {self.losses.total:.4f}"
                f"  policy {self.losses.policy:.4f}  value {self.losses.value:.4f}")
        if self.gate is not None:
            verdict = "promoted" if self.promoted else "kept incumbent"
            low, high = self.gate.interval
            text += (f"\n         gate {self.gate.score:.1%} over {self.gate.games} "
                     f"games ({self.gate.elo:+.0f} Elo [{low:+.0f}, {high:+.0f}]) "
                     f"-> {verdict}")
        return text


class Learner:
    """Owns the network, the optimiser and the replay buffer across iterations."""

    def __init__(self, game, config: LearnConfig | None = None,
                 seed: int | None = None):
        self.game = game
        self.config = config or LearnConfig()
        self.rng = np.random.default_rng(seed)

        self.device = (
            torch.device(self.config.train_device)
            if self.config.train_device
            else best_device()
        )
        self.net = PolicyValueNet.for_game(game, self.config.network).to(self.device)
        self.optimizer = make_optimizer(self.net, self.config.train)
        self.buffer = ReplayBuffer(self.config.buffer_capacity)
        self.iteration = 0
        self.history: list[IterationStats] = []
        self._pool: ParallelSelfPlay | None = None
        # The network that generates self-play data. Without gating it is simply
        # the latest one; with gating it is the last one to win a promotion
        # match, so a bad iteration cannot poison every iteration after it.
        self.best_net = self.cpu_net() if self.config.gate.enabled else None

    def __enter__(self) -> Learner:
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def close(self) -> None:
        if self._pool is not None:
            self._pool.close()
            self._pool = None

    def cpu_net(self) -> PolicyValueNet:
        """A CPU copy of the network, for one-position-at-a-time evaluation.

        Self-play evaluates single positions, which is faster on CPU than on the
        GPU - the round trip costs more than the work. When training lives on the
        GPU the two need different copies, and the copy is a megabyte or so, so
        the cost is nothing next to what it saves.
        """
        if self.device.type == "cpu":
            return self.net
        mirror = PolicyValueNet.for_game(self.game, self.config.network)
        mirror.load_state_dict(
            {k: v.detach().cpu() for k, v in self.net.state_dict().items()}
        )
        return mirror

    def playing_net(self) -> PolicyValueNet:
        """The network self-play uses: the incumbent if gating, else the latest."""
        return self.best_net if self.best_net is not None else self.cpu_net()

    def _mcts(self) -> MCTS:
        # Rebuilt each iteration so it always wraps the current network.
        return MCTS(self.game, NetworkEvaluator(self.playing_net()),
                    self.config.mcts, rng=self.rng)

    def _generate(self):
        """Produce this iteration's self-play data, in parallel if configured."""
        if not self.config.workers or self.config.workers == 1:
            with single_threaded():
                return generate(
                    self.game, self._mcts(), self.config.games_per_iteration,
                    self.config.selfplay, self.rng,
                )

        if self._pool is None:
            self._pool = ParallelSelfPlay(
                self.game.name, self.config.network, self.config.mcts,
                self.config.selfplay, self.config.workers,
            )
            self._pool.__enter__()

        # A fresh seed per iteration. Reusing one would make every iteration
        # replay the same games with a network that had moved on - the buffer
        # would fill with variations on a single opening.
        seed = int(self.rng.integers(0, 2**31 - 1))
        return self._pool.generate(
            self.playing_net().state_dict(),
            self.config.games_per_iteration,
            seed,
        )

    def run_iteration(self) -> IterationStats:
        started = time.perf_counter()
        self.iteration += 1

        play_started = time.perf_counter()
        samples = self._generate()
        self.buffer.extend(samples)
        selfplay_seconds = time.perf_counter() - play_started

        train_started = time.perf_counter()
        losses = None
        if len(self.buffer) >= self.config.min_buffer_before_training:
            totals = [0.0, 0.0, 0.0]
            for _ in range(self.config.train_steps_per_iteration):
                batch = self.buffer.sample(
                    self.config.train.batch_size, self.rng, self.device
                )
                step = train_step(self.net, self.optimizer, batch)
                totals[0] += step.total
                totals[1] += step.policy
                totals[2] += step.value
            n = self.config.train_steps_per_iteration
            losses = Losses(totals[0] / n, totals[1] / n, totals[2] / n)
        train_seconds = time.perf_counter() - train_started

        gate_result, promoted = self._run_gate(trained=losses is not None)

        stats = IterationStats(
            iteration=self.iteration,
            games=self.config.games_per_iteration,
            positions=len(samples),
            buffer=len(self.buffer),
            losses=losses,
            seconds=time.perf_counter() - started,
            selfplay_seconds=selfplay_seconds,
            train_seconds=train_seconds,
            gate=gate_result,
            promoted=promoted,
        )
        self.history.append(stats)
        return stats

    def _run_gate(self, trained: bool) -> tuple[MatchResult | None, bool]:
        """Play the challenger against the incumbent and decide on promotion."""
        gate = self.config.gate
        if not gate.enabled or self.best_net is None or not trained:
            return None, False

        challenger = Player("challenger", NetworkEvaluator(self.cpu_net()),
                            gate.simulations)
        incumbent = Player("incumbent", NetworkEvaluator(self.best_net),
                           gate.simulations)
        result = play_match(self.game, challenger, incumbent, gate.games,
                            self.rng, gate.opening_plies)

        promoted = result.score > gate.threshold
        if promoted:
            self.best_net = self.cpu_net()
        return result, promoted

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
                # Saved on CPU so a checkpoint can be loaded anywhere.
                "network": {k: v.detach().cpu()
                            for k, v in self.net.state_dict().items()},
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
        self.net.to(self.device)
        if self.best_net is not None:
            self.best_net = self.cpu_net()
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.iteration = checkpoint["iteration"]
