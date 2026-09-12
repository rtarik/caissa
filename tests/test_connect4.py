"""Tests for Connect 4 and, through it, for the Game contract itself."""

from __future__ import annotations

import numpy as np
import pytest

from caissa.games.base import Game
from caissa.games.connect4 import COLS, ROWS, Connect4, Connect4State


@pytest.fixture
def game() -> Connect4:
    return Connect4()


def play(game: Connect4, *columns: int):
    """Apply a sequence of column drops, starting from the initial position."""
    state = game.initial_state()
    for column in columns:
        state = game.apply(state, column)
    return state


def test_satisfies_game_protocol(game):
    assert isinstance(game, Game)


def test_initial_state_is_empty_and_open(game):
    state = game.initial_state()
    assert state.board.shape == (ROWS, COLS)
    assert not state.board.any()
    assert game.legal_actions(state).all()
    assert game.terminal_value(state) is None


def test_pieces_fall_to_the_lowest_empty_cell(game):
    state = game.apply(game.initial_state(), 3)
    # The mover's piece is -1 from the next player's perspective.
    assert state.board[ROWS - 1, 3] == -1
    assert np.count_nonzero(state.board) == 1

    state = game.apply(state, 3)
    assert state.board[ROWS - 2, 3] == -1  # stacked on top
    assert state.board[ROWS - 1, 3] == 1  # and the first piece flipped sign


def test_full_column_becomes_illegal(game):
    state = play(game, *([0] * ROWS))
    assert not game.legal_actions(state)[0]
    assert game.legal_actions(state)[1:].all()
    with pytest.raises(ValueError):
        game.apply(state, 0)


@pytest.mark.parametrize(
    ("columns", "label"),
    [
        ((0, 1, 0, 1, 0, 1, 0), "vertical"),
        ((0, 0, 1, 1, 2, 2, 3), "horizontal"),
        ((0, 1, 1, 2, 2, 3, 2, 3, 3, 4, 3), "rising diagonal"),
        ((3, 2, 2, 1, 1, 0, 1, 0, 0, 6, 0), "falling diagonal"),
    ],
)
def test_detects_win(game, columns, label):
    state = play(game, *columns)
    # The winner just moved, so the player to move has lost.
    assert game.terminal_value(state) == -1.0, f"{label} win not detected"


def test_no_premature_termination(game):
    state = play(game, 0, 1, 0, 1, 0, 1)  # three in a column, not four
    assert game.terminal_value(state) is None


def test_value_is_never_positive_for_the_player_to_move(game):
    """A win is always detected on the loser's turn, never the winner's.

    If the mover had just won, the game would have ended a ply earlier, so +1
    is unreachable here. This encodes the mover-relative value convention.
    """
    rng = np.random.default_rng(0)
    for _ in range(300):
        state = game.initial_state()
        while (value := game.terminal_value(state)) is None:
            legal = np.flatnonzero(game.legal_actions(state))
            state = game.apply(state, int(rng.choice(legal)))
        assert value in (-1.0, 0.0)


def test_draw_on_full_board(game):
    # Column-ordered filling that reaches 42 pieces with no line of four.
    order = [0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0,
             2, 3, 2, 3, 2, 3, 3, 2, 3, 2, 3, 2,
             4, 5, 4, 5, 4, 5, 5, 4, 5, 4, 5, 4,
             6, 6, 6, 6, 6, 6]
    state = play(game, *order)
    assert np.count_nonzero(state.board) == ROWS * COLS
    assert game.terminal_value(state) == 0.0


def test_random_games_always_terminate(game):
    rng = np.random.default_rng(7)
    for _ in range(200):
        state = game.initial_state()
        for _ in range(ROWS * COLS + 1):
            if game.terminal_value(state) is not None:
                break
            legal = np.flatnonzero(game.legal_actions(state))
            assert legal.size > 0
            state = game.apply(state, int(rng.choice(legal)))
        else:
            pytest.fail("game exceeded the maximum possible number of plies")


def test_encoding_separates_the_two_players(game):
    state = play(game, 3, 4)
    encoded = game.encode(state)

    assert encoded.shape == (game.input_planes, ROWS, COLS)
    assert encoded.dtype == np.float32
    # Plane 0 is always the mover's pieces, plane 1 the opponent's.
    assert encoded[0, ROWS - 1, 3] == 1.0
    assert encoded[1, ROWS - 1, 4] == 1.0
    assert encoded.sum() == 2.0


def test_applying_a_move_swaps_the_two_planes(game):
    """The canonical flip, stated precisely.

    If it is P's turn in state ``s``, then plane 0 holds P's pieces and plane 1
    holds the opponent's. After P moves, it is the opponent's turn, so the two
    planes must exchange roles - and the only difference is P's new piece.
    """
    rng = np.random.default_rng(3)
    for _ in range(200):
        state = game.initial_state()
        for _ in range(int(rng.integers(0, 10))):
            if game.terminal_value(state) is not None:
                break
            legal = np.flatnonzero(game.legal_actions(state))
            state = game.apply(state, int(rng.choice(legal)))
        if game.terminal_value(state) is not None:
            continue

        before = game.encode(state)
        action = int(rng.choice(np.flatnonzero(game.legal_actions(state))))
        after = game.encode(game.apply(state, action))

        # The opponent's pieces are untouched, and are now plane 0.
        np.testing.assert_array_equal(after[0], before[1])
        # The mover's pieces are now plane 1, with exactly one added.
        added = after[1] - before[0]
        assert added.sum() == 1.0
        assert set(np.unique(added)) <= {0.0, 1.0}


def test_symmetries_include_identity_and_mirror(game):
    state = play(game, 0, 1, 0)
    encoded = game.encode(state)
    policy = np.arange(COLS, dtype=np.float32)

    variants = game.symmetries(encoded, policy)
    assert len(variants) == 2

    identity_board, identity_policy = variants[0]
    np.testing.assert_array_equal(identity_board, encoded)
    np.testing.assert_array_equal(identity_policy, policy)

    mirror_board, mirror_policy = variants[1]
    np.testing.assert_array_equal(mirror_board, encoded[:, :, ::-1])
    np.testing.assert_array_equal(mirror_policy, policy[::-1])


def mirror(state: Connect4State) -> Connect4State:
    """The left-right reflection of a position, used to check the symmetry."""
    last = state.last_move
    return Connect4State(
        board=state.board[:, ::-1].copy(),
        last_move=None if last is None else (last[0], COLS - 1 - last[1]),
        ply=state.ply,
    )


def test_mirroring_commutes_with_playing(game):
    """Mirror-then-play must equal play-then-mirror, for every legal action.

    This is the property that makes symmetry augmentation sound. If the policy
    permutation did not match the board permutation, augmentation would pair
    positions with the wrong move distributions and quietly poison training.
    """
    rng = np.random.default_rng(11)
    checked = 0
    for _ in range(400):
        state = game.initial_state()
        for _ in range(int(rng.integers(0, 20))):
            if game.terminal_value(state) is not None:
                break
            legal = np.flatnonzero(game.legal_actions(state))
            state = game.apply(state, int(rng.choice(legal)))
        if game.terminal_value(state) is not None:
            continue

        for action in np.flatnonzero(game.legal_actions(state)):
            action = int(action)
            play_then_mirror = mirror(game.apply(state, action))
            mirror_then_play = game.apply(mirror(state), COLS - 1 - action)

            np.testing.assert_array_equal(
                mirror_then_play.board, play_then_mirror.board
            )
            assert mirror_then_play.last_move == play_then_mirror.last_move
            assert game.terminal_value(mirror_then_play) == game.terminal_value(
                play_then_mirror
            )
            checked += 1

    assert checked > 1000, "symmetry check did not exercise enough positions"


def test_mirrored_legal_actions_match(game):
    rng = np.random.default_rng(13)
    for _ in range(200):
        state = game.initial_state()
        for _ in range(int(rng.integers(0, 30))):
            if game.terminal_value(state) is not None:
                break
            legal = np.flatnonzero(game.legal_actions(state))
            state = game.apply(state, int(rng.choice(legal)))
        np.testing.assert_array_equal(
            game.legal_actions(mirror(state)), game.legal_actions(state)[::-1]
        )


def test_legal_actions_mirror_correctly(game):
    state = play(game, *([0] * ROWS))  # column 0 full
    legal = game.legal_actions(state)
    assert not legal[0]
    # Under the mirror, the full column moves to the far side.
    assert not legal[::-1][COLS - 1]


def test_render_shows_the_grid(game):
    rendered = game.render(play(game, 3))
    lines = rendered.splitlines()
    assert len(lines) == ROWS + 1  # board plus the column ruler
    assert lines[ROWS - 1].split() == [".", ".", ".", "o", ".", ".", "."]
