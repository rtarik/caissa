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
"""

from __future__ import annotations

from collections import deque
from typing import Iterable

import numpy as np
import torch

from caissa.selfplay import Sample


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self._samples: deque[Sample] = deque(maxlen=capacity)

    def extend(self, samples: Iterable[Sample]) -> None:
        self._samples.extend(samples)

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
        chosen = [self._samples[int(i)] for i in indices]

        positions = torch.from_numpy(np.stack([s.encoded for s in chosen]))
        policies = torch.from_numpy(
            np.stack([s.policy for s in chosen]).astype(np.float32)
        )
        values = torch.tensor([s.value for s in chosen], dtype=torch.float32)

        if device is not None:
            positions = positions.to(device)
            policies = policies.to(device)
            values = values.to(device)
        return positions, policies, values
