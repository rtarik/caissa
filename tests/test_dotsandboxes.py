"""Tests for Dots and Boxes.

Two things get most of the attention. The bonus move, because this is the first
game where the player to move can stay the same, and a mistake there is a silent
sign error that trains toward giving boxes away. And the symmetries, because a
quarter-turn swaps horizontal lines for vertical ones - a new way for an
augmented sample to be quietly wrong.
"""

from __future__ import annotations

import numpy as np
import pytest

from caissa.games.base import Game
from caissa.games.dotsandboxes import (
    BOX_LINES,
    BOXES,
    DOTS,
    HORIZONTAL_LINES,
    LATTICE,
    LINE_BOXES,
    LINES,
    SQUARES,
    DotsAndBoxes,
    horizontal,
    vertical,
)


@pytest.fixture
def game() -> DotsAndBoxes:
    return DotsAndBoxes()


def play(game, *lines):
    state = game.initial_state()
    for line in lines:
        state = game.apply(state, line)
    return state


def random_game(game, rng):
    """Every state of one random game, the final one included."""
    state = game.initial_state()
    states = [state]
    while game.terminal_value(state) is None:
        legal = np.flatnonzero(game.legal_actions(state))
        state = game.apply(state, int(rng.choice(legal)))
        states.append(state)
    return states


def cell_to_line(row: int, col: int) -> int:
    """A line from its lattice coordinates - derived here, not from the module."""
    if row % 2 == 0 and col % 2 == 1:
        return (row // 2) * BOXES + (col - 1) // 2
    if row % 2 == 1 and col % 2 == 0:
        return HORIZONTAL_LINES + ((row - 1) // 2) * DOTS + col // 2
    raise AssertionError(f"({row}, {col}) is not a line cell")


def test_satisfies_game_protocol(game):
    assert isinstance(game, Game)


# -------------------------------------------------------------------- geometry


def test_standard_board_geometry():
    assert (BOXES, DOTS, LINES, SQUARES, LATTICE) == (5, 6, 60, 25, 11)


def test_every_box_has_four_distinct_lines():
    for lines in BOX_LINES:
        assert len(set(lines)) == 4


def test_perimeter_lines_border_one_box_and_inner_lines_two():
    bordering = [len(boxes) for boxes in LINE_BOXES]
    assert bordering.count(1) == 4 * BOXES
    assert bordering.count(2) == LINES - 4 * BOXES
    assert sum(bordering) == 4 * SQUARES


# --------------------------------------------------------------------- turns


def test_starts_with_every_line_open_and_seat_zero_to_move(game):
    state = game.initial_state()
    assert game.legal_actions(state).all()
    assert game.to_play(state) == 0
    assert game.terminal_value(state) is None
    assert game.score(state) == (0, 0)


def test_a_line_that_closes_nothing_passes_the_turn(game):
    state = play(game, horizontal(0, 0))
    assert game.to_play(state) == 1
    assert game.to_play(game.apply(state, vertical(2, 3))) == 0


def test_closing_a_box_claims_it_and_keeps_the_turn(game):
    # Top, bottom and left of box (0, 0), by alternating players...
    three_sides = play(game, horizontal(0, 0), horizontal(1, 0), vertical(0, 0))
    assert game.to_play(three_sides) == 1
    assert game.score(three_sides) == (0, 0)

    # ...then the right side closes it, for whoever draws it.
    closed = game.apply(three_sides, vertical(0, 1))
    assert game.to_play(closed) == 1, "a closed box earns another move"
    assert game.score(closed) == (1, 0), "and the box belongs to the mover"


def test_one_line_can_close_two_boxes(game):
    # Three sides each of boxes (0, 0) and (0, 1), leaving the side they share.
    state = play(game, horizontal(0, 0), horizontal(1, 0), vertical(0, 0),
                 horizontal(0, 1), horizontal(1, 1), vertical(0, 2))
    assert game.to_play(state) == 0 and game.score(state) == (0, 0)

    both = game.apply(state, vertical(0, 1))
    assert game.to_play(both) == 0
    assert game.score(both) == (2, 0)


def test_boxes_change_hands_only_when_the_turn_passes(game):
    """Canonical perspective follows the seat, not the move."""
    rng = np.random.default_rng(0)
    bonuses = passes = 0
    for _ in range(20):
        states = random_game(game, rng)
        for before, after in zip(states, states[1:]):
            mine, theirs = game.score(before)
            if game.to_play(after) == game.to_play(before):
                assert game.score(after)[0] - mine in (1, 2)
                assert game.score(after)[1] == theirs
                bonuses += 1
            else:
                assert game.score(after) == (theirs, mine)
                passes += 1
    assert bonuses > 50 and passes > 500


def test_to_play_reports_the_seat(game):
    state = play(game, horizontal(0, 0))
    assert game.to_play(state) == state.seat == 1


# ------------------------------------------------------------------- endings


def test_every_game_lasts_exactly_sixty_lines(game):
    rng = np.random.default_rng(1)
    for _ in range(20):
        assert len(random_game(game, rng)) == LINES + 1


def test_the_last_line_always_closes_a_box(game):
    """Every box the last line borders already has its other three sides."""
    rng = np.random.default_rng(2)
    for _ in range(40):
        *_, before, final = random_game(game, rng)
        assert game.to_play(final) == game.to_play(before)


def test_the_player_to_move_at_the_end_is_often_the_winner(game):
    """+1 is routine here, where Connect 4 and Gomoku can never return it."""
    rng = np.random.default_rng(3)
    outcomes = {game.terminal_value(random_game(game, rng)[-1]) for _ in range(60)}
    assert outcomes == {-1.0, 1.0}, "25 boxes: no draws, and both results occur"


def test_terminal_value_is_the_sign_of_the_score(game):
    rng = np.random.default_rng(4)
    for _ in range(30):
        final = random_game(game, rng)[-1]
        mine, theirs = game.score(final)
        assert mine + theirs == SQUARES
        assert game.terminal_value(final) == (1.0 if mine > theirs else -1.0)


def test_a_drawn_line_cannot_be_drawn_again(game):
    state = play(game, horizontal(2, 2))
    assert not game.legal_actions(state)[horizontal(2, 2)]
    with pytest.raises(ValueError):
        game.apply(state, horizontal(2, 2))


# ------------------------------------------------------------------ encoding


def test_encoding_places_lines_and_boxes_on_the_lattice(game):
    drawn = {horizontal(0, 0), horizontal(1, 0), vertical(0, 0), vertical(0, 1)}
    state = play(game, *drawn)
    encoded = game.encode(state)

    assert encoded.shape == (game.input_planes, LATTICE, LATTICE)
    assert encoded.dtype == np.float32
    assert encoded[0].sum() == 4
    assert {cell_to_line(r, c) for r, c in zip(*np.nonzero(encoded[0]))} == drawn
    # The closed box belongs to the mover, and sits at its centre cell.
    assert encoded[1, 1, 1] == 1.0 and encoded[1].sum() == 1
    assert encoded[2].sum() == 0


def test_orientation_planes_are_constant_and_correct(game):
    encoded = game.encode(game.initial_state())
    assert encoded[3].sum() == HORIZONTAL_LINES
    assert encoded[4].sum() == LINES - HORIZONTAL_LINES
    assert all(r % 2 == 0 and c % 2 == 1 for r, c in zip(*np.nonzero(encoded[3])))
    assert all(r % 2 == 1 and c % 2 == 0 for r, c in zip(*np.nonzero(encoded[4])))


# ---------------------------------------------------------------- symmetries


def asymmetric_state(game):
    """A position with no symmetry of its own, so all eight variants differ."""
    return play(game, horizontal(0, 0), vertical(1, 3), horizontal(4, 2), vertical(3, 0))


def test_eight_distinct_symmetries_with_the_identity_first(game):
    encoded = game.encode(asymmetric_state(game))
    policy = np.arange(LINES, dtype=np.float32)
    variants = game.symmetries(encoded, policy)

    assert len(variants) == 8
    assert len({board.tobytes() for board, _ in variants}) == 8
    np.testing.assert_array_equal(variants[0][0], encoded)
    np.testing.assert_array_equal(variants[0][1], policy)


def test_orientation_planes_survive_every_symmetry(game):
    """A quarter-turn carries horizontal lines onto vertical ones.

    Rotating the whole tensor would leave the two constant planes swapped - an
    augmented sample that never occurs in real play. They must come out of every
    variant exactly as they went in.
    """
    encoded = game.encode(asymmetric_state(game))
    for board, _ in game.symmetries(encoded, np.zeros(LINES, np.float32)):
        np.testing.assert_array_equal(board[3], encoded[3])
        np.testing.assert_array_equal(board[4], encoded[4])
    # The trap is real: naively rotated, the horizontal plane no longer matches.
    assert not np.array_equal(np.rot90(encoded[3]), encoded[3])


def test_symmetries_do_not_write_through_to_the_input(game):
    encoded = game.encode(asymmetric_state(game))
    original = encoded.copy()
    for board, _ in game.symmetries(encoded, np.zeros(LINES, np.float32)):
        board[:] = 7.0
    np.testing.assert_array_equal(encoded, original)


def test_policy_moves_with_the_board_under_every_symmetry(game):
    """The line plane and the policy must be permuted identically.

    Checked without the module's permutation tables: draw a line, and the one new
    cell in each variant's line plane must be exactly where that variant's policy
    put its weight.
    """
    rng = np.random.default_rng(5)
    checked = 0
    for _ in range(12):
        state = game.initial_state()
        for _ in range(int(rng.integers(0, 40))):
            legal = np.flatnonzero(game.legal_actions(state))
            state = game.apply(state, int(rng.choice(legal)))

        for action in rng.choice(np.flatnonzero(game.legal_actions(state)), 4, replace=False):
            action = int(action)
            onehot = np.zeros(LINES, np.float32)
            onehot[action] = 1.0
            before = game.symmetries(game.encode(state), onehot)
            after = game.symmetries(game.encode(game.apply(state, action)), onehot)
            for (board, policy), (moved, _) in zip(before, after):
                new = np.argwhere(moved[0] - board[0] == 1.0)
                assert len(new) == 1
                assert cell_to_line(*new[0]) == int(policy.argmax())
                checked += 1
    assert checked == 12 * 4 * 8


def test_symmetries_preserve_lines_and_score(game):
    state = random_game(game, np.random.default_rng(6))[45]
    encoded = game.encode(state)
    for board, _ in game.symmetries(encoded, np.zeros(LINES, np.float32)):
        for plane in (0, 1, 2):
            assert board[plane].sum() == encoded[plane].sum()


def test_render_draws_the_dot_grid(game):
    lines = game.render(play(game, horizontal(0, 0))).splitlines()
    assert len(lines) == DOTS + BOXES + 1
    assert lines[0].startswith("•───•")
