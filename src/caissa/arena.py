"""Measuring strength by playing games, rather than by reading the loss.

Everything the training loop reports about itself is circular: the loss measures
agreement with targets the loop generated. A network can minimise it beautifully
while getting no stronger, and in this project it already has. The only way to
know whether an agent improved is to make it play something and count.

Three things make a match measure what it is supposed to.

**Alternate who moves first.** Connect 4 is a first-player win with perfect play,
so a match where one side always starts measures the advantage of starting, not
the difference between the players.

**Play paired games from a shared opening.** Both players face the same position,
once from each side. That removes the luck of the draw: if an opening is simply
good for whoever starts, both players collect that equally and it cancels.

**Do not let determinism collapse the match.** Two deterministic players from the
same start play one game, N times. Variety has to come from somewhere, and a
random opening of a few plies is a cleaner source than randomising the play
itself - the players are then measured making their best moves, not their noisy
ones.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from caissa.evaluator import Evaluator
from caissa.mcts import MCTS, MCTSConfig


@dataclass
class Player:
    """An evaluator plus how much search it is allowed."""

    name: str
    evaluator: Evaluator
    #: 0 plays straight from the policy head, with no search at all. Useful for
    #: asking what the *network* learned rather than what search can rescue.
    simulations: int = 50

    def choose(self, game, state, rng: np.random.Generator) -> int:
        if self.simulations == 0:
            priors, _ = self.evaluator.evaluate(game, state)
            return int(priors.argmax())
        mcts = MCTS(game, self.evaluator, MCTSConfig(simulations=self.simulations),
                    rng=rng)
        # No Dirichlet noise and no temperature: this is the player trying to
        # win, not generating training data.
        policy, _ = mcts.run(state, temperature=0.0, add_noise=False)
        return int(policy.argmax())


@dataclass
class MatchResult:
    player: str
    opponent: str
    wins: int
    draws: int
    losses: int

    @property
    def games(self) -> int:
        return self.wins + self.draws + self.losses

    @property
    def score(self) -> float:
        """Points per game: a win is 1, a draw is 0.5, a loss is 0."""
        return (self.wins + 0.5 * self.draws) / self.games

    @property
    def elo(self) -> float:
        return elo_difference(self.score)

    @property
    def interval(self) -> tuple[float, float]:
        """95% confidence interval on the Elo difference.

        Reported alongside the point estimate because the point estimate on its
        own invites conclusions the sample size does not support. A 20-game match
        won 60-40 has an interval hundreds of Elo wide and means almost nothing.
        """
        n = self.games
        mean = self.score
        # Variance of the per-game score, from the win/draw/loss counts.
        squares = self.wins * 1.0 + self.draws * 0.25
        variance = squares / n - mean * mean
        if variance <= 0 or n < 2:
            return (elo_difference(mean), elo_difference(mean))
        error = 1.96 * math.sqrt(variance / (n - 1))
        return (
            elo_difference(min(max(mean - error, 1e-9), 1 - 1e-9)),
            elo_difference(min(max(mean + error, 1e-9), 1 - 1e-9)),
        )

    @property
    def significant(self) -> bool:
        """Whether the confidence interval excludes zero.

        A match that fails this has not shown a difference, whatever its score
        says. Promoting on an insignificant result is promoting on noise, and the
        agent then wanders rather than improves while every report looks fine.
        """
        low, high = self.interval
        return low > 0.0 or high < 0.0

    def summary(self) -> str:
        low, high = self.interval
        mark = "" if self.significant else "  (not significant)"
        return (f"{self.player} vs {self.opponent}: "
                f"+{self.wins} ={self.draws} -{self.losses} "
                f"({self.score:.1%}, {self.elo:+.0f} Elo "
                f"[{low:+.0f}, {high:+.0f}]){mark}")


def elo_difference(score: float) -> float:
    """Convert a score in (0, 1) into an Elo difference.

    Elo assumes a logistic relationship between rating difference and expected
    score, and - more importantly - it assumes *transitivity*: that if A beats B
    and B beats C then A beats C. Self-play agents routinely break that, learning
    a style that beats their immediate predecessor while losing to something
    older. A rising Elo curve computed only against the previous generation can
    therefore describe an agent going round in circles. Anchoring to a fixed
    opponent is what keeps the number meaningful.
    """
    score = min(max(score, 1e-9), 1 - 1e-9)
    return -400.0 * math.log10(1.0 / score - 1.0)


def expected_score(elo: float) -> float:
    """The inverse: what an Elo difference predicts as a score."""
    return 1.0 / (1.0 + 10.0 ** (-elo / 400.0))


def games_needed(elo: float, confidence: float = 1.96) -> int:
    """Games required before a difference of ``elo`` is distinguishable from zero.

    This function exists to be sobering. Near even, an Elo point is worth about
    0.00144 of a point per game, while a single game's score has a standard
    deviation of about 0.5 - so the noise is enormous relative to the signal, and
    the required sample grows as the *square* of the precision wanted::

        100 Elo      ~50 games
         50 Elo     ~190 games
         20 Elo   ~1,200 games
         10 Elo   ~4,600 games

    Which is why a 20-game match settles nothing, and why anyone tuning an engine
    on eyeballed results is usually tuning noise.
    """
    difference = expected_score(elo) - 0.5
    if difference <= 0:
        raise ValueError("elo must be positive")
    return math.ceil((confidence * 0.5 / difference) ** 2)


def resolvable_elo(games: int, confidence: float = 1.96) -> float:
    """The smallest Elo difference ``games`` games can distinguish from zero.

    The inverse of :func:`games_needed`, and the more useful direction when
    choosing a promotion threshold: a gate whose threshold sits below what its
    sample can resolve will promote on noise. AlphaGo Zero's 400 games at 55%
    corresponds to 35 Elo, and 400 games resolves 34 - the threshold was chosen
    to match the sample, not picked as a round number.
    """
    if games < 2:
        raise ValueError("need at least two games")
    target = confidence * 0.5 / math.sqrt(games)
    low, high = 0.0, 4000.0
    for _ in range(100):
        middle = (low + high) / 2
        if expected_score(middle) - 0.5 < target:
            low = middle
        else:
            high = middle
    return low


def random_opening(game, plies: int, rng: np.random.Generator):
    """A random position ``plies`` moves in, guaranteed not to be over.

    Retries rather than returning a finished game: a match played from terminal
    positions would score without either player making a move.
    """
    while True:
        state = game.initial_state()
        for _ in range(plies):
            if game.terminal_value(state) is not None:
                break
            legal = np.flatnonzero(game.legal_actions(state))
            state = game.apply(state, int(rng.choice(legal)))
        else:
            if game.terminal_value(state) is None:
                return state


def play_game(game, first: Player, second: Player, state,
              rng: np.random.Generator) -> float:
    """Play out ``state`` and return the result **for** ``first``.

    +1 win, 0.5 draw, 0 loss. The sign work is the fiddly part: when the loop
    ends, ``terminal_value`` describes the position for whoever is *to move*,
    which is the player who did not just move - so it is the loser's view of a
    decisive game.
    """
    players = (first, second)
    turn = 0
    while (outcome := game.terminal_value(state)) is None:
        state = game.apply(state, players[turn % 2].choose(game, state, rng))
        turn += 1

    # ``outcome`` belongs to players[turn % 2], the side to move at the end.
    result = outcome if turn % 2 == 0 else -outcome
    return {1.0: 1.0, 0.0: 0.5, -1.0: 0.0}[result]


def play_match(game, player: Player, opponent: Player, games: int,
               rng: np.random.Generator, opening_plies: int = 2) -> MatchResult:
    """Play ``games`` games in colour-reversed pairs and tally the result.

    An odd count is rounded down to the nearest pair, because an unpaired game
    would reintroduce exactly the first-move bias the pairing removes.
    """
    pairs = games // 2
    if pairs < 1:
        raise ValueError("a match needs at least two games, to make one pair")

    wins = draws = losses = 0
    for _ in range(pairs):
        opening = random_opening(game, opening_plies, rng)
        for scored in (
            play_game(game, player, opponent, opening, rng),
            1.0 - play_game(game, opponent, player, opening, rng),
        ):
            if scored == 1.0:
                wins += 1
            elif scored == 0.5:
                draws += 1
            else:
                losses += 1

    return MatchResult(player.name, opponent.name, wins, draws, losses)
