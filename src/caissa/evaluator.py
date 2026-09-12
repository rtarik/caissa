"""The boundary between search and knowledge.

Monte Carlo tree search needs exactly two things from a position: a prior
probability for each legal move, and an estimate of how good the position is for
the player to move. Where those come from is not the search's concern.

Keeping that boundary explicit buys three things:

* The search can be tested without training anything, using the uniform
  evaluator below or a hand-written oracle.
* A trained network drops in later without the search changing.
* The difference between "the search is broken" and "the network is untrained"
  stays diagnosable, which matters enormously in reinforcement learning, where
  the usual failure mode is a system that runs perfectly and learns nothing.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np


class Evaluator(Protocol):
    """Supplies priors and a position value to the search."""

    def evaluate(self, game, state) -> tuple[np.ndarray, float]:
        """Return ``(priors, value)`` for ``state``.

        ``priors`` is a probability distribution of shape ``(action_size,)``
        that is already masked to legal moves and sums to 1. Masking belongs
        here rather than in the search so that every evaluator is forced to
        respect the rules, and the search can treat its input as trustworthy.

        ``value`` is in ``[-1, 1]`` and follows the project-wide convention: it
        is the expected outcome **for the player to move**, not for any fixed
        player.
        """


class UniformEvaluator:
    """Knows nothing: every legal move equally likely, every position a draw.

    This is the honest starting point - it is what an untrained network
    approximates - and it makes a useful test fixture. Search driven by this
    evaluator has no positional understanding whatsoever, so anything it finds
    was found by *search alone*. If tree search cannot spot a win in one with a
    uniform evaluator, the search is broken, and no amount of training will
    rescue it.
    """

    def evaluate(self, game, state) -> tuple[np.ndarray, float]:
        legal = game.legal_actions(state)
        priors = legal.astype(np.float32)
        return priors / priors.sum(), 0.0
