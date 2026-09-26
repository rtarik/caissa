"""Fitting a rating to games against opponents whose ratings are known.

The arena compares two versions of the agent and reports the gap between them.
That gap is real, but it floats: a ladder of gaps says nothing about where the
ladder stands. Anchoring it takes opponents with ratings already attached, and a
way to turn a pile of results against several of them into one number.

The number is the maximum-likelihood rating under the Elo model. A player rated
R is expected to score 1 / (1 + 10^((R_opp - R) / 400)) against an opponent
rated R_opp; the fit is the R that makes the observed scores most probable, all
opponents at once. Draws count as half a win and half a loss, which is the
standard treatment and exactly what the expected-score formula assumes.
"""

from __future__ import annotations

import numpy as np

#: How far the log-likelihood falls from its peak at the edge of a 95% interval:
#: half the 0.95 quantile of a chi-squared distribution with one degree of freedom.
INTERVAL_DROP = 1.92


def expected(rating: float, opponent: float) -> float:
    """The score a player of ``rating`` expects against ``opponent``."""
    return 1.0 / (1.0 + 10 ** ((opponent - rating) / 400))


def fit(results: list[tuple[float, float]],
        lowest: float = 0.0, highest: float = 4000.0) -> tuple[float, float, float]:
    """The maximum-likelihood rating from (opponent rating, score) pairs.

    Returns (rating, low, high), the ends of a 95% likelihood-ratio interval.
    Maximised over a one-point grid, which for a single parameter is both exact
    enough and impossible to get subtly wrong. When every game was won, or every
    game lost, the likelihood never turns over and the estimate runs to the edge
    of the grid - the honest answer is a bound, and ``low`` or ``high`` says so.
    """
    if not results:
        raise ValueError("no games to fit a rating to")
    grid = np.arange(lowest, highest + 1.0, 1.0)
    opponents = np.array([opponent for opponent, _ in results], dtype=float)
    scores = np.array([score for _, score in results], dtype=float)

    p = 1.0 / (1.0 + 10 ** ((opponents[None, :] - grid[:, None]) / 400))
    p = np.clip(p, 1e-12, 1 - 1e-12)
    loglik = (scores * np.log(p) + (1 - scores) * np.log(1 - p)).sum(axis=1)

    best = int(loglik.argmax())
    inside = grid[loglik >= loglik[best] - INTERVAL_DROP]
    return float(grid[best]), float(inside.min()), float(inside.max())


def fit_pool(games: list[tuple[str, str, float]], anchor: str,
             prior: float = 0.5, iterations: int = 2000) -> dict[str, float]:
    """Ratings for every player at once, from (player, opponent, player's score) games.

    The Bradley-Terry model, which is the Elo model for a whole pool: each player
    has a strength, and the expected score between two players depends only on
    the gap. Fitted by the minorise-maximise update (Hunter, 2004), which only
    ever increases the likelihood and needs no step size to tune.

    Two details matter here. Ratings are only defined up to a common shift - add
    100 to everyone and every expected score is unchanged - so one player, the
    ``anchor``, is pinned at zero and the rest are read against it. And a player
    who won or lost every game would have no finite rating, so each pairing gets
    ``prior`` of a virtual draw: a small pull towards the middle that keeps every
    rating finite without moving a well-measured one.
    """
    players = sorted({p for g in games for p in g[:2]})
    index = {p: i for i, p in enumerate(players)}
    n = len(players)
    played = np.zeros((n, n))
    scored = np.zeros(n)
    for a, b, score in games:
        i, j = index[a], index[b]
        played[i, j] += 1
        played[j, i] += 1
        scored[i] += score
        scored[j] += 1 - score
    # The virtual draws: half a game's worth each way, for every pair that met.
    met = played > 0
    played = played + prior * met
    scored = scored + (prior / 2) * met.sum(axis=1)

    strength = np.ones(n)
    for _ in range(iterations):
        denominator = (played / (strength[:, None] + strength[None, :])).sum(axis=1)
        updated = scored / denominator
        updated /= updated[index[anchor]]
        if np.allclose(updated, strength, rtol=1e-10, atol=0):
            strength = updated
            break
        strength = updated
    return {p: float(400 * np.log10(strength[index[p]])) for p in players}
