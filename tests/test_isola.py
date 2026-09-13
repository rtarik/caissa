"""Tests for Isola.

The compound action space is what is new, and it is where the attention goes.
Two halves multiplied into one index means every transform has to act on both,
and the failure mode - a policy describing different moves from the position it
is paired with - is completely silent.
"""

from __future__ import annotations

import numpy as np
import pytest

from caissa.games.base import Game
from caissa.games.isola import (
    ACTIONS,
    DIRECTIONS,
    SIZE,
    SQUARES,
    Isola,
    IsolaState,
    direction_permutation,
    step,
)


@pytest.fixture
def game() -> Isola:
    return Isola()


def square(row: int, col: int) -> int:
    return row * SIZE + col


def action(direction: int, destroy: int) -> int:
    return direction * SQUARES + destroy


def test_satisfies_game_protocol(game):
    assert isinstance(game, Game)


def test_the_action_space_is_a_product(game):
    """Eight directions times forty-nine squares - the chess encoding in
    miniature, where 4672 is 73 move types times 64 squares."""
    assert game.action_size == ACTIONS == len(DIRECTIONS) * SQUARES == 392


def test_opening_position(game):
    state = game.initial_state()
    assert state.usable.all()
    assert state.mover == square(0, SIZE // 2)
    assert state.opponent == square(SIZE - 1, SIZE // 2)
    assert game.terminal_value(state) is None

    # Five of eight directions are on the board from the top edge, and each
    # allows destroying any square but the two occupied ones.
    assert int(game.legal_actions(state).sum()) == 5 * (SQUARES - 2)


# ------------------------------------------------------------------- the move


def test_a_move_and_a_demolition_happen_together(game):
    state = game.initial_state()
    down = DIRECTIONS.index((1, 0))
    target = step(state.mover, down)
    victim = square(3, 3)

    after = game.apply(state, action(down, victim))

    assert not after.usable[victim], "the square should be gone"
    assert int(after.usable.sum()) == SQUARES - 1
    # The mover became the opponent: one swap per apply, as everywhere.
    assert after.opponent == target
    assert after.mover == state.opponent
    assert after.ply == 1


def test_the_vacated_square_may_be_destroyed(game):
    """It is free again the moment the piece leaves it."""
    state = game.initial_state()
    down = DIRECTIONS.index((1, 0))
    assert game.legal_actions(state)[action(down, state.mover)]

    after = game.apply(state, action(down, state.mover))
    assert not after.usable[state.mover]


def test_occupied_squares_cannot_be_destroyed(game):
    state = game.initial_state()
    down = DIRECTIONS.index((1, 0))
    target = step(state.mover, down)

    legal = game.legal_actions(state)
    assert not legal[action(down, target)], "cannot destroy the square moved onto"
    assert not legal[action(down, state.opponent)], "nor the one the opponent holds"


def test_moves_off_the_board_are_illegal(game):
    state = game.initial_state()  # mover on the top edge
    for direction, (d_row, _) in enumerate(DIRECTIONS):
        if d_row >= 0:
            continue
        slice_ = game.legal_actions(state)[direction * SQUARES : (direction + 1) * SQUARES]
        assert not slice_.any(), f"direction {DIRECTIONS[direction]} leaves the board"


def test_cannot_step_onto_the_opponent(game):
    state = IsolaState(
        usable=np.ones(SQUARES, dtype=bool),
        mover=square(3, 3), opponent=square(3, 4), ply=0,
    )
    right = DIRECTIONS.index((0, 1))
    assert not game.legal_actions(state)[right * SQUARES : (right + 1) * SQUARES].any()


def test_cannot_step_onto_a_destroyed_square(game):
    usable = np.ones(SQUARES, dtype=bool)
    usable[square(4, 3)] = False
    state = IsolaState(usable=usable, mover=square(3, 3), opponent=square(0, 0), ply=0)

    down = DIRECTIONS.index((1, 0))
    assert not game.legal_actions(state)[down * SQUARES : (down + 1) * SQUARES].any()


def test_an_illegal_action_is_refused(game):
    state = game.initial_state()
    up = DIRECTIONS.index((-1, 0))
    with pytest.raises(ValueError):
        game.apply(state, action(up, square(3, 3)))


# ---------------------------------------------------------------- the ending


def test_a_stranded_player_has_lost(game):
    """Being unable to move ends the game here - the opposite of Reversi, where
    it means pass. Which one applies is a rule, and the framework is told."""
    usable = np.zeros(SQUARES, dtype=bool)
    usable[square(3, 3)] = True   # the mover's own square, and nothing around it
    usable[square(0, 0)] = True   # the opponent, far away
    state = IsolaState(usable=usable, mover=square(3, 3), opponent=square(0, 0), ply=10)

    assert not game.legal_actions(state).any()
    assert game.terminal_value(state) == -1.0


def test_a_legal_step_always_has_a_legal_demolition(game):
    """The two halves are not independent, and it took a failing test to notice.

    A position where the mover can step but has nothing to destroy looks like it
    should exist - both halves are required, so surely either could run out. It
    cannot: the square just vacated is always standing, is never occupied
    afterwards, and is therefore always available to destroy. So the game ends on
    immobility alone, and ``terminal_value`` needs to consider nothing else.
    """
    rng = np.random.default_rng(11)
    for _ in range(40):
        state = game.initial_state()
        for _ in range(int(rng.integers(0, 40))):
            if game.terminal_value(state) is not None:
                break
            legal = np.flatnonzero(game.legal_actions(state))
            state = game.apply(state, int(rng.choice(legal)))

        can_step = any(
            (target := step(state.mover, d)) is not None
            and state.usable[target]
            and target != state.opponent
            for d in range(len(DIRECTIONS))
        )
        assert can_step == bool(game.legal_actions(state).any())

    # And directly: strip the board to the bare minimum and it still holds.
    usable = np.zeros(SQUARES, dtype=bool)
    usable[square(3, 3)] = True   # the mover, whose square survives the step
    usable[square(3, 4)] = True   # the only step available
    usable[square(0, 0)] = True   # the opponent, out of reach
    state = IsolaState(usable=usable, mover=square(3, 3), opponent=square(0, 0), ply=10)

    right = DIRECTIONS.index((0, 1))
    assert game.legal_actions(state)[action(right, square(3, 3))], (
        "the vacated square is the demolition of last resort"
    )
    assert int(game.legal_actions(state).sum()) == 1


def test_value_is_never_positive_for_the_player_to_move(game):
    """Like Connect 4 and Gomoku: the game ends on the loser's turn. Unlike
    Reversi, where it ends when neither side can move and +1 is normal."""
    rng = np.random.default_rng(0)
    for _ in range(25):
        state = game.initial_state()
        while (value := game.terminal_value(state)) is None:
            legal = np.flatnonzero(game.legal_actions(state))
            state = game.apply(state, int(rng.choice(legal)))
        assert value == -1.0, "Isola has no draws"


def test_games_terminate_and_shrink_the_board(game):
    rng = np.random.default_rng(1)
    for _ in range(15):
        state = game.initial_state()
        standing = int(state.usable.sum())
        for _ in range(SQUARES + 1):
            if game.terminal_value(state) is not None:
                break
            legal = np.flatnonzero(game.legal_actions(state))
            state = game.apply(state, int(rng.choice(legal)))
            assert int(state.usable.sum()) == standing - 1, "one square per turn"
            standing -= 1
        else:
            pytest.fail("game ran longer than there are squares to destroy")


# -------------------------------------------------------------------- encoding


def test_encoding_shows_both_pieces_and_the_standing_squares(game):
    state = game.initial_state()
    encoded = game.encode(state)

    assert encoded.shape == (3, SIZE, SIZE)
    assert encoded[0].sum() == 1 and encoded[1].sum() == 1
    assert encoded[0].ravel()[state.mover] == 1.0
    assert encoded[1].ravel()[state.opponent] == 1.0
    assert encoded[2].sum() == SQUARES


def test_applying_a_move_swaps_the_two_piece_planes(game):
    state = game.initial_state()
    down = DIRECTIONS.index((1, 0))
    before = game.encode(state)
    after = game.encode(game.apply(state, action(down, square(3, 3))))

    # Plane 0 is always the mover's; after the turn that is the other player.
    np.testing.assert_array_equal(after[0], before[1])
    assert after[2].sum() == before[2].sum() - 1


# ------------------------------------------------------------------ symmetries


VARIANTS = [(turns, mirror) for turns in range(4) for mirror in (False, True)]


def transform_square(value: int, turns: int, mirror: bool) -> int:
    labels = np.rot90(np.arange(SQUARES).reshape(SIZE, SIZE), turns)
    if mirror:
        labels = labels[:, ::-1]
    row, col = np.argwhere(labels == value)[0]
    return int(row) * SIZE + int(col)


def transform_action(value: int, turns: int, mirror: bool) -> int:
    direction, destroy = divmod(value, SQUARES)
    return (
        direction_permutation(turns, mirror)[direction] * SQUARES
        + transform_square(destroy, turns, mirror)
    )


def transform_state(state: IsolaState, turns: int, mirror: bool) -> IsolaState:
    usable = np.rot90(state.usable.reshape(SIZE, SIZE), turns)
    if mirror:
        usable = usable[:, ::-1]
    return IsolaState(
        usable=np.ascontiguousarray(usable).ravel(),
        mover=transform_square(state.mover, turns, mirror),
        opponent=transform_square(state.opponent, turns, mirror),
        ply=state.ply,
    )


def test_direction_permutation_is_a_bijection():
    for turns, mirror in VARIANTS:
        permutation = direction_permutation(turns, mirror)
        assert sorted(permutation) == list(range(len(DIRECTIONS))), (turns, mirror)


def test_the_identity_transform_moves_nothing():
    assert direction_permutation(0, False) == list(range(len(DIRECTIONS)))


def test_a_quarter_turn_rotates_the_compass():
    """np.rot90 sends (r, c) to (N-1-c, r), so a step (dr, dc) becomes (-dc, dr)."""
    permutation = direction_permutation(1, False)
    for before, (d_row, d_col) in enumerate(DIRECTIONS):
        assert DIRECTIONS[permutation[before]] == (-d_col, d_row)


def test_transforming_commutes_with_playing(game):
    """The property the whole augmentation rests on, for a compound action.

    Both halves must move together: the destroyed square like any square, the
    direction rotating with it. Transforming one and not the other still yields
    a legal action and a plausible board - which is exactly why it has to be
    checked rather than eyeballed.
    """
    rng = np.random.default_rng(3)
    checked = 0
    for _ in range(20):
        state = game.initial_state()
        for _ in range(int(rng.integers(0, 18))):
            if game.terminal_value(state) is not None:
                break
            legal = np.flatnonzero(game.legal_actions(state))
            state = game.apply(state, int(rng.choice(legal)))
        if game.terminal_value(state) is not None:
            continue

        legal = np.flatnonzero(game.legal_actions(state))
        for value in rng.choice(legal, size=min(8, legal.size), replace=False):
            value = int(value)
            for turns, mirror in VARIANTS:
                played_then_moved = transform_state(
                    game.apply(state, value), turns, mirror
                )
                moved_then_played = game.apply(
                    transform_state(state, turns, mirror),
                    transform_action(value, turns, mirror),
                )
                np.testing.assert_array_equal(
                    moved_then_played.usable, played_then_moved.usable
                )
                assert moved_then_played.mover == played_then_moved.mover
                assert moved_then_played.opponent == played_then_moved.opponent
                checked += 1

    assert checked > 500, "symmetry check did not exercise enough positions"


def test_policy_permutation_matches_the_action_transform(game):
    """What `symmetries` does to a one-hot policy must equal `transform_action`."""
    state = game.initial_state()
    for _ in range(3):
        state = game.apply(state, int(np.flatnonzero(game.legal_actions(state))[0]))
    encoded = game.encode(state)

    rng = np.random.default_rng(5)
    for value in rng.choice(np.flatnonzero(game.legal_actions(state)), size=6):
        value = int(value)
        policy = np.zeros(ACTIONS, dtype=np.float32)
        policy[value] = 1.0
        for index, (turns, mirror) in enumerate(VARIANTS):
            _, permuted = game.symmetries(encoded, policy)[index]
            assert permuted.argmax() == transform_action(value, turns, mirror), (
                f"action {value} under rot{turns} mirror={mirror}"
            )
            assert permuted.sum() == pytest.approx(1.0)


def test_symmetries_include_the_identity(game):
    encoded = game.encode(game.initial_state())
    policy = np.arange(ACTIONS, dtype=np.float32)
    board, permuted = game.symmetries(encoded, policy)[0]
    np.testing.assert_array_equal(board, encoded)
    np.testing.assert_array_equal(permuted, policy)


def test_there_are_eight_of_them(game):
    state = game.initial_state()
    for _ in range(4):
        state = game.apply(state, int(np.flatnonzero(game.legal_actions(state))[0]))
    variants = game.symmetries(game.encode(state), np.zeros(ACTIONS, np.float32))
    assert len(variants) == 8
    assert len({v[0].tobytes() for v in variants}) == 8


def test_render_reports_the_position(game):
    text = game.render(game.initial_state())
    assert "X" in text and "O" in text
    assert "49 squares standing" in text
