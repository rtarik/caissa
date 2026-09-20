"""A sliding window over recent self-play data.

Two problems this solves, both of which will silently wreck training.

**Correlation.** Successive positions in a game are nearly identical - one piece
moves. A gradient step taken on a contiguous run of them is a step taken on
essentially one position, repeated. Sampling at random from a large pool breaks
that up, so each batch covers many games and many stages of play.

**Non-stationarity.** This is the part with no equivalent in supervised learning.
The data distribution is produced by the network being trained, so it *moves as
the network moves*. Training only on the newest games means chasing a target that
runs away, and the network can forget what it knew two generations ago. Keeping a
window of recent games smooths the shift.

The window size is a genuine trade-off, not a tuning detail. Too small and
training is unstable and forgetful. Too large and most batches come from
noticeably weaker versions of the network, so learning slows. AlphaZero kept the
most recent 500,000 games.

**What a sample costs.** Positions are kept in half precision and policies as
only the moves that actually received visits. Written out in full a chess sample
is 42 KB - its policy alone is 4,672 numbers, all but about thirty of them zero -
and 2.7 KB kept this way, which is the difference between a 200,000-position
window costing 8 GB and costing half of one. Half precision loses the last digit
of a plane like the fifty-move counter, which is nothing beside the noise already
in a Monte Carlo value target.

**A window is run state, not a cache.** A stage that resumes with an empty buffer
spends its first iterations training on a single iteration's games - Phase 8
measured what that costs - so it can be written beside the checkpoint.
"""

from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Iterable

import numpy as np
import torch

from caissa.selfplay import Sample

#: (planes, the actions search visited, their share of the visits, the result).
Stored = tuple[np.ndarray, np.ndarray, np.ndarray, float]


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self._samples: deque[Stored] = deque(maxlen=capacity)
        self._action_size = 0

    def extend(self, samples: Iterable[Sample]) -> None:
        for sample in samples:
            policy = np.asarray(sample.policy)
            self._action_size = max(self._action_size, len(policy))
            visited = np.flatnonzero(policy)
            self._samples.append((
                np.asarray(sample.encoded, dtype=np.float16),
                visited.astype(np.int32),
                policy[visited].astype(np.float32),
                float(sample.value),
            ))

    def __len__(self) -> int:
        return len(self._samples)

    @property
    def full(self) -> bool:
        return len(self._samples) == self.capacity

    def sample(self, batch_size: int, rng: np.random.Generator,
               device: torch.device | None = None
               ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Draw a random batch as ``(positions, policy targets, value targets)``.

        Sampling is with replacement, which at these buffer sizes is
        indistinguishable from without and avoids any bookkeeping.
        """
        if not self._samples:
            raise ValueError("replay buffer is empty")

        indices = rng.integers(0, len(self._samples), size=batch_size)
        chosen = [self._samples[int(index)] for index in indices]

        positions = torch.from_numpy(np.stack([c[0] for c in chosen]).astype(np.float32))
        # The stored policies are rebuilt into dense rows in one scatter.
        counts = np.fromiter((len(c[1]) for c in chosen), dtype=np.int64, count=batch_size)
        rows = torch.from_numpy(np.repeat(np.arange(batch_size), counts))
        columns = torch.from_numpy(np.concatenate([c[1] for c in chosen]).astype(np.int64))
        policies = torch.zeros(batch_size, self._action_size)
        policies[rows, columns] = torch.from_numpy(np.concatenate([c[2] for c in chosen]))
        values = torch.tensor([c[3] for c in chosen], dtype=torch.float32)

        if device is not None:
            positions = positions.to(device)
            policies = policies.to(device)
            values = values.to(device)
        return positions, policies, values

    def save(self, path: str | Path) -> Path:
        """Write the window out, to be picked up by the next stage."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        empty_float, empty_int = np.zeros(0, np.float32), np.zeros(0, np.int32)
        np.savez(
            path,
            planes=(np.stack([s[0] for s in self._samples]) if self._samples
                    else np.zeros(0, np.float16)),
            counts=np.fromiter((len(s[1]) for s in self._samples), dtype=np.int64,
                               count=len(self._samples)),
            actions=(np.concatenate([s[1] for s in self._samples]) if self._samples
                     else empty_int),
            probabilities=(np.concatenate([s[2] for s in self._samples]) if self._samples
                           else empty_float),
            values=np.array([s[3] for s in self._samples], dtype=np.float32),
            action_size=self._action_size,
        )
        return path

    def restore(self, path: str | Path) -> None:
        """Refill this buffer from a saved window, keeping its own capacity."""
        with np.load(path) as saved:
            self._action_size = int(saved["action_size"])
            offsets = np.concatenate([[0], np.cumsum(saved["counts"])])
            planes, actions = saved["planes"], saved["actions"]
            probabilities, values = saved["probabilities"], saved["values"]
            self._samples.clear()
            for index in range(len(values)):
                start, end = int(offsets[index]), int(offsets[index + 1])
                self._samples.append((planes[index], actions[start:end],
                                      probabilities[start:end], float(values[index])))
