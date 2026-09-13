"""Tests for Gomoku.

Structurally close to Connect 4 - place a stone, make a line - so the suite
follows the same shape. What differs is the scale, and the two places where the
similarity to Connect 4 is a trap: the line is five rather than four, and every
square is playable rather than only the column tops.
"""

from __future__ import annotations

import numpy as np
import pytest

from caissa.games.base import Game
from caissa.games.gomoku import CONNECT, SIZE, SQUARES, Gomoku, GomokuState


@pytest.fixture
def game() -> Gomoku:
    return Gomoku()


def square(row: int, col: int) -> int:
    return row * SIZE + col


def play(game: Gomoku, *actions: int) -> GomokuState:
    state = game.initial_state()
    for action in actions:
        state = game.apply(state, action)
    return state


def line(game: Gomoku, cells: list[int], spare: list[int]) -> GomokuState:
    """Alternate the mover's stones through ``cells`` and the opponent's through
    ``spare``, so the mover completes the line on the final ply."""
    actions = []
    for i, cell in enumerate(cells):
        actions.append(cell)
        if i < len(cells) - 1:
            actions.append(spare[i])
    return play(game, *actions)


def test_satisfies_game_protocol(game):
    assert isinstance(game, Game)


def test_starts_empty_with_every_square_open(game):
    state = game.initial_state()
    assert state.board.shape == (SIZE, SIZE)
    assert not state.board.any()
    assert game.legal_actions(state).all()
    assert game.legal_actions(state).size == SQUARES
    assert game.terminal_value(state) is None


def test_a_played_square_is_no_longer_legal(game):
    state = game.apply(game.initial_state(), square(4, 4))
    assert not game.legal_actions(state)[square(4, 4)]
    assert game.legal_actions(state).sum() == SQUARES - 1
    with pytest.raises(ValueError):
        game.apply(state, square(4, 4))


def test_stones_stay_where_they_are_put(game):
    """No gravity, unlike Connect 4 - the whole board is reachable."""
    state = game.apply(game.initial_state(), square(0, 8))
    assert state.board[0, 8] == -1  # the mover's stone, seen by the opponent
    assert np.count_nonzero(state.board) == 1


@pytest.mark.parametrize(
    ("cells", "label"),
    [
        ([square(4, c) for c in range(2, 7)], "horizontal"),
        ([square(r, 4) for r in range(1, 6)], "vertical"),
        ([square(i, i) for i in range(2, 7)], "falling diagonal"),
        ([square(6 - i, 2 + i) for i in range(5)], "rising diagonal"),
    ],
)
def test_five_in_a_row_wins(game, cells, label):
    spare = [square(8, c) for c in range(4)]
    state = line(game, cells, spare)
    # The winner just moved, so the player to move has lost.
    assert game.terminal_value(state) == -1.0, f"{label} five not detected"


def test_four_in_a_row_is_not_enough(game):
    """The trap for anyone arriving from Connect 4."""
    cells = [square(4, c) for c in range(2, 2 + CONNECT - 1)]
    spare = [square(8, c) for c in range(4)]
    state = line(game, cells, spare)
    assert len(cells) == 4
    assert game.terminal_value(state) is None


def test_a_line_broken_by_an_opponent_stone_does_not_win(game):
    # The mover takes (4,2), (4,3), (4,5), (4,6) while the opponent holds (4,4).
    state = play(
        game,
        square(4, 2), square(4, 4),
        square(4, 3), square(8, 0),
        square(4, 5), square(8, 1),
        square(4, 6), square(8, 2),
    )
    assert game.terminal_value(state) is None


def test_value_is_never_positive_for_the_player_to_move(game):
    """Like Connect 4 and unlike Reversi: the game ends the instant someone wins,
    so it always ends on the loser's turn."""
    rng = np.random.default_rng(0)
    for _ in range(40):
        state = game.initial_state()
        while (value := game.terminal_value(state)) is None:
            legal = np.flatnonzero(game.legal_actions(state))
            state = game.apply(state, int(rng.choice(legal)))
        assert value in (-1.0, 0.0)


def test_random_games_terminate(game):
    rng = np.random.default_rng(1)
    for _ in range(20):
        state = game.initial_state()
        for _ in range(SQUARES + 1):
            if game.terminal_value(state) is not None:
                break
            legal = np.flatnonzero(game.legal_actions(state))
            assert legal.size > 0
            state = game.apply(state, int(rng.choice(legal)))
        else:
            pytest.fail("game exceeded the maximum possible number of plies")


def test_a_full_board_with_no_line_is_a_draw(game):
    """Constructed directly: reaching a genuine 81-stone draw by play is rare."""
    board = np.zeros((SIZE, SIZE), dtype=np.int8)
    # Stripes four rows deep alternate ownership, so no line of five exists in
    # any direction while every square is occupied.
    for r in range(SIZE):
        for c in range(SIZE):
            board[r, c] = 1 if ((r // 2) + (c // 2)) % 2 == 0 else -1
    state = GomokuState(board=board, last_move=square(0, 0), ply=SQUARES)
    assert not (board == 0).any()
    assert game.terminal_value(state) == 0.0


# -------------------------------------------------------------------- encoding


def test_encoding_separates_the_two_players(game):
    state = play(game, square(4, 4), square(0, 0))
    encoded = game.encode(state)
    assert encoded.shape == (game.input_planes, SIZE, SIZE)
    assert encoded.dtype == np.float32
    # Two plies in, the player who opened is on move again - so (4,4) is theirs
    # and sits in plane 0, while the reply at (0,0) is the opponent's in plane 1.
    assert encoded[0, 4, 4] == 1.0
    assert encoded[1, 0, 0] == 1.0
    assert encoded.sum() == 2.0


def test_applying_a_move_swaps_the_two_planes(game):
    before = game.encode(play(game, square(4, 4), square(0, 0)))
    after = game.encode(play(game, square(4, 4), square(0, 0), square(8, 8)))
    np.testing.assert_array_equal(after[0], before[1])
    added = after[1] - before[0]
    assert added.sum() == 1.0


# ------------------------------------------------------------------ symmetries


def transform_board(board: np.ndarray, turns: int, mirror: bool) -> np.ndarray:
    grid = np.rot90(board, turns)
    if mirror:
        grid = grid[:, ::-1]
    return np.ascontiguousarray(grid)


def transform_action(action: int, turns: int, mirror: bool) -> int:
    labels = transform_board(np.arange(SQUARES).reshape(SIZE, SIZE), turns, mirror)
    row, col = np.argwhere(labels == action)[0]
    return int(row) * SIZE + int(col)


VARIANTS = [(turns, mirror) for turns in range(4) for mirror in (False, True)]


def test_a_square_board_has_eight_symmetries(game):
    state = play(game, square(2, 3), square(5, 1))
    variants = game.symmetries(game.encode(state), np.zeros(SQUARES, np.float32))
    assert len(variants) == 8
    assert len({v[0].tobytes() for v in variants}) == 8


def test_symmetries_include_the_identity(game):
    encoded = game.encode(play(game, square(2, 3)))
    policy = np.arange(SQUARES, dtype=np.float32)
    board, permuted = game.symmetries(encoded, policy)[0]
    np.testing.assert_array_equal(board, encoded)
    np.testing.assert_array_equal(permuted, policy)


def test_policy_permutation_matches_the_board_permutation(game):
    encoded = game.encode(play(game, square(2, 3)))
    for index, (turns, mirror) in enumerate(VARIANTS):
        for action in (square(0, 0), square(2, 5), square(8, 3)):
            policy = np.zeros(SQUARES, dtype=np.float32)
            policy[action] = 1.0
            _, permuted = game.symmetries(encoded, policy)[index]
            assert permuted.argmax() == transform_action(action, turns, mirror)


def test_transforming_commutes_with_playing(game):
    """Eight variants per position, so a mismatch corrupts data eight times over."""
    rng = np.random.default_rng(3)
    checked = 0
    for _ in range(15):
        state = game.initial_state()
        for _ in range(int(rng.integers(0, 20))):
            if game.terminal_value(state) is not None:
                break
            legal = np.flatnonzero(game.legal_actions(state))
            state = game.apply(state, int(rng.choice(legal)))
        if game.terminal_value(state) is not None:
            continue

        for action in rng.choice(np.flatnonzero(game.legal_actions(state)), size=6):
            action = int(action)
            for turns, mirror in VARIANTS:
                rotated = GomokuState(
                    board=transform_board(state.board, turns, mirror),
                    last_move=None if state.last_move is None
                    else transform_action(state.last_move, turns, mirror),
                    ply=state.ply,
                )
                play_then_transform = transform_board(
                    game.apply(state, action).board, turns, mirror
                )
                moved = game.apply(rotated, transform_action(action, turns, mirror))
                np.testing.assert_array_equal(moved.board, play_then_transform)
                checked += 1

    assert checked > 400, "symmetry check did not exercise enough positions"


def test_terminal_detection_survives_the_symmetries(game):
    """A win must stay a win however the board is turned."""
    cells = [square(i, i) for i in range(2, 7)]
    won = line(game, cells, [square(8, c) for c in range(4)])
    assert game.terminal_value(won) == -1.0

    for turns, mirror in VARIANTS:
        rotated = GomokuState(
            board=transform_board(won.board, turns, mirror),
            last_move=transform_action(won.last_move, turns, mirror),
            ply=won.ply,
        )
        assert game.terminal_value(rotated) == -1.0


def test_render_shows_the_grid(game):
    text = game.render(play(game, square(4, 4)))
    assert len(text.splitlines()) == SIZE + 1
