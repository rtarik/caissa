"""Generating training data by having the agent play itself.

Each move produces one training example: the position, the visit distribution
that search settled on, and - once the game is over - how the game turned out for
whoever was to move in that position.

The value target deserves a paragraph, because it is a real choice and not the
obvious one. Most reinforcement learning *bootstraps*: it trains V(s) toward
``reward + V(s')``, using its own estimate of the next state as part of the
target. AlphaZero does not. It uses the **actual final result of the game**,
applied back to every position in it.

The trade is bias against variance. A bootstrapped target is low-variance but
biased, because it inherits whatever the network currently gets wrong, and those
errors can circulate and reinforce themselves. The final outcome is unbiased - it
is what actually happened - but noisy, since one blunder on move 40 relabels
every earlier position in that game as a loss. AlphaZero can afford the noise:
self-play data is cheap, and the tree search already supplies the lookahead that
bootstrapping would otherwise provide. So it takes the unbiased target and drowns
the variance in volume.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from caissa.mcts import MCTS


@dataclass
class SelfPlayConfig:
    #: Plies played at temperature 1 before switching to greedy play. Early
    #: randomness is what stops every game being the same game; without it the
    #: buffer fills with near-duplicates and the network sees a narrow slice of
    #: the state space. AlphaZero used 30 plies for chess.
    temperature_moves: int = 8
    #: Temperature applied during those opening plies.
    temperature: float = 1.0


@dataclass
class Sample:
    """One training example."""

    #: Encoded position, canonical perspective, shape ``(planes, H, W)``.
    encoded: np.ndarray
    #: Search's visit distribution over actions - the policy target.
    policy: np.ndarray
    #: Final game result for the player to move in this position.
    value: float


def play_game(game, mcts: MCTS, config: SelfPlayConfig | None = None,
              rng: np.random.Generator | None = None) -> list[Sample]:
    """Play one game against itself and return its training examples.

    Both sides are the same network with the same search. There is no opponent
    to speak of - the agent is simply asked to find the best moves it can from
    both sides of the board, and the record of that attempt becomes the lesson.
    """
    config = config or SelfPlayConfig()
    rng = rng if rng is not None else np.random.default_rng()

    state = game.initial_state()
    positions: list[tuple[np.ndarray, np.ndarray]] = []

    ply = 0
    while (outcome := game.terminal_value(state)) is None:
        temperature = config.temperature if ply < config.temperature_moves else 0.0
        # Noise is on: this is data generation, and the point is variety. During
        # evaluation or real play it would be turned off.
        policy, _ = mcts.run(state, temperature=temperature, add_noise=True)
        positions.append((game.encode(state), policy))

        # Sample from the search policy rather than taking its argmax. At
        # temperature 0 the policy is already one-hot so this is the greedy move
        # anyway; during the opening plies it is what produces the variety.
        action = int(rng.choice(len(policy), p=policy))
        state = game.apply(state, action)
        ply += 1

    # ``outcome`` is from the perspective of the player to move in the *final*
    # position. Walking backwards and flipping the sign at each step gives every
    # earlier position the result as seen by whoever was to move there. This is
    # the same alternation as the search backup, and gets broken the same way:
    # silently, with training proceeding happily in the wrong direction.
    samples: list[Sample] = []
    value = outcome
    for encoded, policy in reversed(positions):
        value = -value
        samples.append(Sample(encoded=encoded, policy=policy, value=value))
    samples.reverse()
    return samples


def augment(game, samples: list[Sample]) -> list[Sample]:
    """Expand samples using the board's symmetries.

    Self-play positions are expensive - each one cost a few hundred network
    evaluations - so anything that multiplies them for free is worth taking.
    Connect 4 doubles; a square board with the full dihedral group gives eight.

    The value is untouched: reflecting a board does not change who is winning.
    """
    expanded: list[Sample] = []
    for sample in samples:
        for encoded, policy in game.symmetries(sample.encoded, sample.policy):
            expanded.append(Sample(encoded=encoded, policy=policy, value=sample.value))
    return expanded


def generate(game, mcts: MCTS, games: int, config: SelfPlayConfig | None = None,
             rng: np.random.Generator | None = None,
             augment_samples: bool = True) -> list[Sample]:
    """Play ``games`` self-play games and return all their training examples."""
    rng = rng if rng is not None else np.random.default_rng()
    samples: list[Sample] = []
    for _ in range(games):
        played = play_game(game, mcts, config, rng)
        samples.extend(augment(game, played) if augment_samples else played)
    return samples
