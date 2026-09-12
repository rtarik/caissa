"""Tests for self-play data generation.

The value target gets the most attention here. It is the one label in the system
that depends on something that happens *after* the position it describes, which
makes it the easiest to get subtly backwards - and a sign error produces training
data that is perfectly consistent, perfectly wrong, and completely silent.
"""

from __future__ import annotations

import numpy as np
import pytest

from caissa.evaluator import UniformEvaluator
from caissa.games.connect4 import COLS, Connect4
from caissa.mcts import MCTS, MCTSConfig
from caissa.selfplay import SelfPlayConfig, Sample, augment, generate, play_game


@pytest.fixture
def game() -> Connect4:
    return Connect4()


class ScriptedMCTS:
    """Plays a fixed sequence, so a game's outcome is known in advance."""

    def __init__(self, moves):
        self.moves = list(moves)
        self.index = 0

    def run(self, state, temperature=1.0, add_noise=True):
        policy = np.zeros(COLS)
        policy[self.moves[self.index]] = 1.0
        self.index += 1
        return policy, 0.0


# Player one drops into column 0 on plies 0, 2, 4 and 6, winning vertically.
SCRIPTED_WIN = [0, 1, 0, 1, 0, 1, 0]
# Fills the board with no line of four.
SCRIPTED_DRAW = [0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0,
                 2, 3, 2, 3, 2, 3, 3, 2, 3, 2, 3, 2,
                 4, 5, 4, 5, 4, 5, 5, 4, 5, 4, 5, 4,
                 6, 6, 6, 6, 6, 6]


def test_value_targets_follow_the_winner(game):
    """The winner's positions are +1, the loser's -1, alternating strictly.

    Player one moves on even plies and wins, so every even-indexed sample must be
    +1 and every odd one -1. Inverting the assignment would still produce a tidy
    alternating sequence, which is why the parity is pinned to the actual winner
    rather than merely checked for alternation.
    """
    samples = play_game(game, ScriptedMCTS(SCRIPTED_WIN))

    assert len(samples) == len(SCRIPTED_WIN)
    values = [s.value for s in samples]
    assert values == [1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0]


def test_drawn_games_label_every_position_zero(game):
    samples = play_game(game, ScriptedMCTS(SCRIPTED_DRAW))
    assert len(samples) == len(SCRIPTED_DRAW)
    assert all(s.value == 0.0 for s in samples)


def test_values_are_only_ever_outcomes(game):
    """Monte Carlo targets, not bootstrapped estimates: only -1, 0 or +1 appear."""
    mcts = MCTS(game, UniformEvaluator(), MCTSConfig(simulations=15),
                rng=np.random.default_rng(0))
    for _ in range(5):
        for sample in play_game(game, mcts, rng=np.random.default_rng(1)):
            assert sample.value in (-1.0, 0.0, 1.0)


def test_policies_are_distributions_over_legal_moves(game):
    mcts = MCTS(game, UniformEvaluator(), MCTSConfig(simulations=20),
                rng=np.random.default_rng(2))
    for sample in play_game(game, mcts, rng=np.random.default_rng(3)):
        assert sample.policy.sum() == pytest.approx(1.0)
        assert (sample.policy >= 0).all()


def test_positions_are_encoded_not_raw_states(game):
    samples = play_game(game, ScriptedMCTS(SCRIPTED_WIN))
    assert samples[0].encoded.shape == (game.input_planes, *game.board_shape)
    assert samples[0].encoded.dtype == np.float32


def test_temperature_produces_varied_games(game):
    """Without opening randomness every game would be the same game."""
    def first_moves(seed):
        mcts = MCTS(game, UniformEvaluator(), MCTSConfig(simulations=25),
                    rng=np.random.default_rng(seed))
        samples = play_game(game, mcts, SelfPlayConfig(temperature_moves=8),
                            rng=np.random.default_rng(seed))
        return tuple(int(s.policy.argmax()) for s in samples[:4])

    assert len({first_moves(s) for s in range(8)}) > 1


def test_greedy_play_after_the_temperature_window(game):
    """Once the window closes, the recorded policy must be one-hot."""
    mcts = MCTS(game, UniformEvaluator(), MCTSConfig(simulations=25),
                rng=np.random.default_rng(4))
    samples = play_game(game, mcts, SelfPlayConfig(temperature_moves=2),
                        rng=np.random.default_rng(4))
    for sample in samples[2:]:
        assert sample.policy.max() == 1.0


# ----------------------------------------------------------------- augmentation


def test_augmentation_multiplies_by_the_symmetry_count(game):
    samples = play_game(game, ScriptedMCTS(SCRIPTED_WIN))
    expanded = augment(game, samples)
    assert len(expanded) == 2 * len(samples)  # Connect 4 has the mirror only


def test_augmentation_preserves_values(game):
    """Reflecting a board does not change who is winning."""
    samples = play_game(game, ScriptedMCTS(SCRIPTED_WIN))
    expanded = augment(game, samples)
    assert sorted(s.value for s in expanded) == sorted(
        v for s in samples for v in (s.value, s.value)
    )


def test_augmentation_mirrors_board_and_policy_together(game):
    """The two must be permuted identically or the pairing is corrupted."""
    original = Sample(
        encoded=game.encode(game.apply(game.initial_state(), 0)),
        policy=np.eye(COLS, dtype=np.float32)[0],
        value=1.0,
    )
    _, mirrored = augment(game, [original])

    np.testing.assert_array_equal(mirrored.encoded, original.encoded[:, :, ::-1])
    assert mirrored.policy.argmax() == COLS - 1


def test_generate_collects_from_several_games(game):
    mcts = MCTS(game, UniformEvaluator(), MCTSConfig(simulations=15),
                rng=np.random.default_rng(5))
    samples = generate(game, mcts, games=3, rng=np.random.default_rng(6))
    assert len(samples) > 3
    assert all(isinstance(s, Sample) for s in samples)


class BiasedMCTS:
    """Always reports the same preference, so only the sampling can vary."""

    def __init__(self, game, weights):
        self.game = game
        self.weights = np.asarray(weights, dtype=float)

    def run(self, state, temperature=1.0, add_noise=True):
        legal = self.game.legal_actions(state)
        policy = self.weights * legal
        if policy.sum() == 0:  # preferred columns exhausted; fall back to legal
            policy = legal.astype(float)
        return policy / policy.sum(), 0.0


def test_moves_are_sampled_from_the_policy_not_argmaxed(game):
    """Inside the temperature window, moves are drawn *from* the distribution.

    Taking the argmax instead would collapse the opening to a single line per
    network, and the buffer would fill with near-duplicate games. The search
    here always returns the same preference, so any variation in what gets
    played can only come from sampling.
    """
    weights = np.zeros(COLS)
    weights[0], weights[1] = 0.6, 0.4

    played = []
    for seed in range(40):
        samples = play_game(
            game,
            BiasedMCTS(game, weights),
            SelfPlayConfig(temperature_moves=99),
            rng=np.random.default_rng(seed),
        )
        # In the position after the first move, plane 1 holds the pieces of the
        # player who just moved - a single piece, in the column they chose.
        opponent_plane = samples[1].encoded[1]
        played.append(int(np.argwhere(opponent_plane)[0][1]))

    assert set(played) == {0, 1}, f"only ever played {set(played)}"
    assert 0.2 < played.count(1) / len(played) < 0.6, "not following the weights"
