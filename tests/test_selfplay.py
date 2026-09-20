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
from caissa.games.dotsandboxes import LINES, DotsAndBoxes
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


# -------------------------------------------------------------- random openings


class FirstLine:
    """Draws the lowest-numbered open line: legal, instant and deterministic."""

    def __init__(self, game):
        self.game = game

    def run(self, state, temperature=1.0, add_noise=True):
        policy = np.zeros(self.game.action_size)
        policy[np.flatnonzero(self.game.legal_actions(state))[0]] = 1.0
        return policy, 0.0


def opening_lengths(config, seeds):
    """How many plies each game spent in a random opening.

    Dots & Boxes makes this exact: every game is sixty lines, so whatever the
    recorded positions fall short by is the random opening - and the first
    recorded board must already show that many lines drawn.
    """
    game = DotsAndBoxes()
    lengths = []
    for seed in seeds:
        samples = play_game(game, FirstLine(game), config, rng=np.random.default_rng(seed))
        skipped = LINES - len(samples)
        assert samples[0].encoded[0].sum() == skipped, "recording began before the opening ended"
        lengths.append(skipped)
    return lengths


def test_random_openings_are_played_but_not_recorded():
    """Nobody searched the random moves, so they must not become training examples."""
    config = SelfPlayConfig(random_opening_share=1.0, random_opening_plies=12)
    lengths = opening_lengths(config, range(100))
    assert min(lengths) == 1
    assert max(lengths) == 12


def test_random_openings_start_only_the_configured_share_of_games():
    lengths = opening_lengths(SelfPlayConfig(random_opening_share=0.25), range(200))
    share = sum(length > 0 for length in lengths) / len(lengths)
    assert 0.15 < share < 0.35


def test_no_random_openings_unless_asked():
    assert set(opening_lengths(SelfPlayConfig(), range(20))) == {0}


# ------------------------------------------------ game length caps and resignation


class Doomed:
    """A *consistent* evaluator: one seat is lost, so the other one is winning.

    Values here are always the mover's own, so an evaluator that tells one player
    -0.95 tells their opponent about +0.95 on the very next ply. A stub that hands
    the same dismal value to both sides is the one case where counting plies and
    counting a player's own moves agree - which is how a resignation rule that can
    never fire in a real game passes a test that looks thorough.
    """

    def __init__(self, game, value=-0.95, seat=0, relent=()):
        self.game = game
        self.value = value
        self.seat = seat
        #: Moves of the doomed seat, counting from zero, where it sees hope instead.
        self.relent = set(relent)
        self.moves = 0

    def run(self, state, temperature=1.0, add_noise=True):
        policy = np.zeros(self.game.action_size)
        policy[np.flatnonzero(self.game.legal_actions(state))[0]] = 1.0
        if self.game.to_play(state) != self.seat:
            return policy, -self.value
        doomed = self.moves not in self.relent
        self.moves += 1
        return policy, self.value if doomed else -self.value


def test_a_player_who_sees_no_hope_resigns(game):
    """Resignation buys back the compute spent playing out lost games.

    Two things have to be right. The count is of a player's *own* moves - their
    opponent's confidence on the ply between is not a reprieve - so the doomed
    player resigns on their second move, at ply 2. And the labels: the resigning
    player is the one to move in the final recorded position, so that sample is
    -1 and their opponent's +1. Backwards, it would teach the network that
    hopeless positions are won, in silence.
    """
    config = SelfPlayConfig(resign_below=-0.9, resign_moves=2)
    samples = play_game(game, Doomed(game), config)

    assert len(samples) == 3, "resigned on the doomed player's second move"
    assert [s.value for s in samples] == [-1.0, 1.0, -1.0]


def test_the_opponent_s_confidence_is_not_a_reprieve(game):
    """The same game with the second player doomed instead, to pin the seat.

    If the counter were per ply rather than per player, both of these would run
    to the end: one player's -0.95 would be cancelled by the other's +0.95 every
    time, and resignation would quietly never happen.
    """
    config = SelfPlayConfig(resign_below=-0.9, resign_moves=2)
    samples = play_game(game, Doomed(game, seat=1), config)

    assert len(samples) == 4, "the second player's second move is ply 3"
    assert [s.value for s in samples] == [1.0, -1.0, 1.0, -1.0]


#: Filling columns left to right decides this game on ply 19.
PLAYED_OUT = 19


def test_one_bad_evaluation_is_not_enough_to_resign(game):
    """A single low value is noise; a player's counter resets when it sees hope."""
    config = SelfPlayConfig(resign_below=-0.9, resign_moves=2)
    # Dismal on every second move of its own, so never twice running.
    samples = play_game(game, Doomed(game, relent=range(1, 20, 2)), config)

    assert len(samples) == PLAYED_OUT, "played on to a real result"
    assert samples[-1].value != 0.0


def test_games_are_played_out_unless_resignation_is_asked_for(game):
    """The same hopeless evaluations, with the setting off, decide nothing."""
    samples = play_game(game, Doomed(game), SelfPlayConfig())
    assert len(samples) == PLAYED_OUT


def test_a_player_can_resign_on_consecutive_plies_when_it_moves_twice():
    """Dots & Boxes hands a player several moves in a row.

    Those are consecutive plies *and* consecutive moves by the same player, so
    resignation is allowed to fire across them - the rule is about a player's own
    moves, not about how many plies went by.
    """
    game = DotsAndBoxes()
    config = SelfPlayConfig(resign_below=-0.9, resign_moves=2)
    samples = play_game(game, Doomed(game), config)

    assert 0 < len(samples) < LINES
    assert samples[-1].value == -1.0


def test_a_capped_game_is_recorded_as_a_draw():
    """Self-play games lengthen as a network improves.

    A network that cannot force a win will shuffle until the rules stop it, which
    in chess takes hundreds of moves of nothing. Calling it a draw at the cap
    spends that compute on new games instead, and labels honestly: neither side
    proved anything.
    """
    game = DotsAndBoxes()
    samples = play_game(game, FirstLine(game), SelfPlayConfig(max_plies=10))

    assert len(samples) == 10, "one sample per ply, then the cap"
    assert [s.value for s in samples] == [0.0] * 10


def test_an_uncapped_game_runs_to_the_end():
    game = DotsAndBoxes()
    samples = play_game(game, FirstLine(game), SelfPlayConfig())
    assert len(samples) == LINES
    assert any(s.value != 0.0 for s in samples), "a decided game, so the cap matters"


def test_the_cap_counts_searched_plies_not_random_opening_ones():
    """The random opening is not the agent's play, and must not eat its budget."""
    game = DotsAndBoxes()
    config = SelfPlayConfig(max_plies=10, random_opening_share=1.0,
                            random_opening_plies=6)
    samples = play_game(game, FirstLine(game), config, rng=np.random.default_rng(0))
    assert len(samples) == 10
