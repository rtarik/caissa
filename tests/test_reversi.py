"""Tests for Reversi, and through it for the Game abstraction.

Connect 4 validated the contract; Reversi is the first thing to stress it. Every
test that has an equivalent in `test_connect4.py` is repeated here, plus the
three things this game adds: passing, scoring, and the full dihedral symmetry.
"""

from __future__ import annotations

import numpy as np
import pytest

from caissa.games.base import Game
from caissa.games.reversi import (
    PASS,
    SIZE,
    SQUARES,
    Reversi,
    ReversiState,
    captures,
)


@pytest.fixture
def game() -> Reversi:
    return Reversi()


def square(row: int, col: int) -> int:
    return row * SIZE + col


def play(game: Reversi, *actions: int) -> ReversiState:
    state = game.initial_state()
    for action in actions:
        state = game.apply(state, action)
    return state


def test_satisfies_game_protocol(game):
    assert isinstance(game, Game)


def test_opening_position_is_standard(game):
    state = game.initial_state()
    # Black moves first and is therefore the mover, so black is +1.
    assert state.board[3, 4] == 1 and state.board[4, 3] == 1
    assert state.board[3, 3] == -1 and state.board[4, 4] == -1
    assert int(np.count_nonzero(state.board)) == 4
    assert game.terminal_value(state) is None

    legal = np.flatnonzero(game.legal_actions(state))
    assert set(legal) == {square(2, 3), square(3, 2), square(4, 5), square(5, 4)}


def test_a_move_captures_the_enclosed_run(game):
    state = game.apply(game.initial_state(), square(2, 3))
    # Black now holds (2,3), the captured (3,3), and its two original discs.
    # From white's perspective those are the -1s.
    assert int((state.board == -1).sum()) == 4
    assert int((state.board == 1).sum()) == 1
    assert state.board[3, 3] == -1, "the enclosed disc should have flipped"


def test_capturing_is_the_legality_test(game):
    """One piece of logic, so the two cannot disagree."""
    board = game.initial_state().board
    assert captures(board, 2, 3), "a legal move captures something"
    assert not captures(board, 0, 0), "an isolated square captures nothing"
    assert not captures(board, 3, 3), "an occupied square is never playable"


def test_an_unproductive_square_is_rejected(game):
    with pytest.raises(ValueError):
        game.apply(game.initial_state(), square(0, 0))


# ---------------------------------------------------------------------- passing


def constructed(board_rows: list[str], passes: int = 0) -> ReversiState:
    """Build a position from a picture. `x` is the mover, `o` the opponent."""
    lookup = {"x": 1, "o": -1, ".": 0}
    board = np.array([[lookup[c] for c in row] for row in board_rows], dtype=np.int8)
    return ReversiState(board=board, passes=passes, ply=int(np.count_nonzero(board)))


#: Reached by real play at ply 59. One square is still empty at (0,3); the mover
#: cannot use it, the opponent can. Taken from an actual random game rather than
#: drawn by hand - the first position written for this test looked stuck and was
#: not, because a diagonal quietly closed on one of the mover's own discs.
MUST_PASS = [
    "xxx.xxoo",
    "xxxxxooo",
    "xoxxoooo",
    "xooo xxoo".replace(" ", ""),
    "xxxooxoo",
    "xxoxooxo",
    "xoxxxxoo",
    "oooooooo",
]

#: Also from real play, at ply 60: the board is full, so neither side can move.
FULL = [
    "xooooxxx",
    "xooxxoxx",
    "xoxoxxxx",
    "xoxooooo",
    "xxxxxxox",
    "xxxxoxxx",
    "oooooxxx",
    "oooooooo",
]


def test_passing_is_legal_only_when_nothing_else_is(game):
    stuck = constructed(MUST_PASS)
    legal = game.legal_actions(stuck)
    assert not legal[:SQUARES].any()
    assert legal[PASS]

    opening = game.legal_actions(game.initial_state())
    assert opening[:SQUARES].any()
    assert not opening[PASS], "declining a turn is not a legal option"


def test_passing_while_a_move_exists_is_refused(game):
    with pytest.raises(ValueError, match="cannot pass"):
        game.apply(game.initial_state(), PASS)


def test_a_pass_hands_over_the_turn_without_changing_the_discs(game):
    stuck = constructed(MUST_PASS)
    after = game.apply(stuck, PASS)

    assert after.passes == 1
    assert after.ply == stuck.ply + 1
    # The board is unchanged apart from the perspective flip.
    np.testing.assert_array_equal(after.board, -stuck.board)


def test_no_legal_move_does_not_end_the_game(game):
    """The assumption Connect 4 never challenged.

    The mover here has nothing to play, yet the game is far from over: after the
    pass the opponent has a move and play continues. A framework that equated an
    empty move list with a finished game would score this as a result.
    """
    stuck = constructed(MUST_PASS)
    assert not game.legal_actions(stuck)[:SQUARES].any()
    assert game.terminal_value(stuck) is None

    after = game.apply(stuck, PASS)
    assert game.legal_actions(after)[:SQUARES].any(), "the opponent can still play"
    assert game.terminal_value(after) is None


def test_two_consecutive_passes_end_the_game(game):
    """A full board reaches the end the same way any deadlock does.

    One termination rule rather than two - "the board is full" and "nobody can
    move" cannot disagree if only the second one exists.
    """
    full = constructed(FULL, passes=1)
    assert not game.legal_actions(full)[:SQUARES].any()
    assert game.terminal_value(full) is None, "one pass is not enough"
    assert game.terminal_value(game.apply(full, PASS)) is not None


def test_a_pass_resets_when_a_move_is_played(game):
    state = game.apply(game.initial_state(), square(2, 3))
    assert state.passes == 0


# --------------------------------------------------------------------- scoring


def test_the_result_is_the_disc_count(game):
    more = constructed(["xxxxxxxx"] + ["o" * 8] + ["x" * 8] + ["." * 8] * 5, passes=2)
    assert game.terminal_value(more) == 1.0

    fewer = constructed(["oooooooo"] + ["x" * 8] + ["o" * 8] + ["." * 8] * 5, passes=2)
    assert game.terminal_value(fewer) == -1.0

    level = constructed(["xxxxxxxx", "oooooooo"] + ["." * 8] * 6, passes=2)
    assert game.terminal_value(level) == 0.0


def test_values_are_mover_relative(game):
    """The same finished board is +1 to one side and -1 to the other."""
    rows = ["xxxxxxxx"] + ["o" * 8] + ["x" * 8] + ["." * 8] * 5
    mine = constructed(rows, passes=2)
    theirs = ReversiState(board=-mine.board, passes=2, ply=mine.ply)
    assert game.terminal_value(mine) == -game.terminal_value(theirs)


def test_random_games_terminate_and_score(game):
    rng = np.random.default_rng(0)
    for _ in range(20):
        state = game.initial_state()
        for _ in range(200):
            outcome = game.terminal_value(state)
            if outcome is not None:
                assert outcome in (-1.0, 0.0, 1.0)
                break
            legal = np.flatnonzero(game.legal_actions(state))
            assert legal.size > 0, "a position with no action at all is impossible"
            state = game.apply(state, int(rng.choice(legal)))
        else:
            pytest.fail("game did not finish")


# -------------------------------------------------------------------- encoding


def test_encoding_shape_and_planes(game):
    encoded = game.encode(game.initial_state())
    assert encoded.shape == (game.input_planes, SIZE, SIZE)
    assert encoded.dtype == np.float32
    assert encoded[0].sum() == 2 and encoded[1].sum() == 2


def test_applying_a_move_swaps_the_two_planes(game):
    """The canonical flip, as in Connect 4 - but captures move discs between the
    planes as well as adding one, so the count is not simply +1."""
    before = game.encode(game.initial_state())
    after = game.encode(game.apply(game.initial_state(), square(2, 3)))

    # Plane 0 is now the opponent's: what was the mover's, minus what was taken.
    assert after[0].sum() == 1
    assert after[1].sum() == 4
    assert before[0].sum() + before[1].sum() + 1 == after[0].sum() + after[1].sum()


# ------------------------------------------------------------------ symmetries


def transform_board(board: np.ndarray, turns: int, mirror: bool) -> np.ndarray:
    grid = np.rot90(board, turns)
    if mirror:
        grid = grid[:, ::-1]
    return np.ascontiguousarray(grid)


def transform_action(action: int, turns: int, mirror: bool) -> int:
    if action == PASS:
        return PASS
    labels = transform_board(np.arange(SQUARES).reshape(SIZE, SIZE), turns, mirror)
    row, col = np.argwhere(labels == action)[0]
    return int(row) * SIZE + int(col)


VARIANTS = [(turns, mirror) for turns in range(4) for mirror in (False, True)]


def test_a_square_board_has_eight_symmetries(game):
    state = play(game, square(2, 3))
    variants = game.symmetries(game.encode(state), np.zeros(game.action_size, np.float32))
    assert len(variants) == 8

    boards = {v[0].tobytes() for v in variants}
    assert len(boards) == 8, "the eight variants should be genuinely distinct"


def test_symmetries_include_the_identity(game):
    encoded = game.encode(play(game, square(2, 3)))
    policy = np.arange(game.action_size, dtype=np.float32)
    board, permuted = game.symmetries(encoded, policy)[0]
    np.testing.assert_array_equal(board, encoded)
    np.testing.assert_array_equal(permuted, policy)


def test_the_pass_action_is_never_permuted(game):
    """Squares move under a rotation; passing means the same thing either way.

    Rotating it along with the squares would pair every augmented position with
    a policy whose final entry belongs to a different action - poisoning eight
    training samples for every one produced.
    """
    encoded = game.encode(play(game, square(2, 3)))
    policy = np.zeros(game.action_size, dtype=np.float32)
    policy[PASS] = 1.0

    for board, permuted in game.symmetries(encoded, policy):
        assert permuted[PASS] == 1.0
        assert permuted[:SQUARES].sum() == 0.0
        assert board.shape == encoded.shape


def test_policy_permutation_matches_the_board_permutation(game):
    """One-hot on a square must land where that square lands."""
    encoded = game.encode(play(game, square(2, 3)))
    for index, (turns, mirror) in enumerate(VARIANTS):
        for action in (square(0, 0), square(2, 5), square(7, 3)):
            policy = np.zeros(game.action_size, dtype=np.float32)
            policy[action] = 1.0
            _, permuted = game.symmetries(encoded, policy)[index]
            assert permuted.argmax() == transform_action(action, turns, mirror), (
                f"action {action} under rot{turns} mirror={mirror}"
            )


def test_transforming_commutes_with_playing(game):
    """Transform-then-play must equal play-then-transform, for all eight.

    The property that makes augmentation sound. With eight variants per position
    rather than Connect 4's two, a mismatch here would corrupt training data four
    times faster.
    """
    rng = np.random.default_rng(3)
    checked = 0
    for _ in range(40):
        state = game.initial_state()
        for _ in range(int(rng.integers(0, 25))):
            if game.terminal_value(state) is not None:
                break
            legal = np.flatnonzero(game.legal_actions(state))
            state = game.apply(state, int(rng.choice(legal)))
        if game.terminal_value(state) is not None:
            continue

        for action in np.flatnonzero(game.legal_actions(state)):
            action = int(action)
            for turns, mirror in VARIANTS:
                rotated = ReversiState(
                    board=transform_board(state.board, turns, mirror),
                    passes=state.passes, ply=state.ply,
                )
                play_then_transform = transform_board(
                    game.apply(state, action).board, turns, mirror
                )
                transform_then_play = game.apply(
                    rotated, transform_action(action, turns, mirror)
                ).board
                np.testing.assert_array_equal(transform_then_play, play_then_transform)
                checked += 1

    assert checked > 500, "symmetry check did not exercise enough positions"


def test_legal_actions_transform_with_the_board(game):
    rng = np.random.default_rng(5)
    for _ in range(20):
        state = game.initial_state()
        for _ in range(int(rng.integers(0, 30))):
            if game.terminal_value(state) is not None:
                break
            legal = np.flatnonzero(game.legal_actions(state))
            state = game.apply(state, int(rng.choice(legal)))

        for turns, mirror in VARIANTS:
            rotated = ReversiState(
                board=transform_board(state.board, turns, mirror),
                passes=state.passes, ply=state.ply,
            )
            expected = {transform_action(int(a), turns, mirror)
                        for a in np.flatnonzero(game.legal_actions(state))}
            actual = set(np.flatnonzero(game.legal_actions(rotated)).tolist())
            assert actual == expected


def test_render_shows_the_score(game):
    text = game.render(game.initial_state())
    assert "x 2  o 2" in text
    assert len(text.splitlines()) == SIZE + 2


def test_the_mover_can_be_the_winner(game):
    """Unlike Connect 4, +1 is reachable here - and that is the game, not a bug.

    Connect 4 ends the instant someone connects four, so the game always finishes
    on the loser's turn and a mover-relative value is only ever -1 or 0. Reversi
    ends when *neither* side can move, and the player to move at that point may
    well be holding more discs.

    Recorded as a test because the Connect 4 behaviour was written into this
    project's conventions as though it were universal, and the TypeScript port
    asserted it for both games before this position proved otherwise.
    """
    rng = np.random.default_rng(0)
    seen = set()
    for _ in range(60):
        state = game.initial_state()
        while (outcome := game.terminal_value(state)) is None:
            legal = np.flatnonzero(game.legal_actions(state))
            state = game.apply(state, int(rng.choice(legal)))
        seen.add(outcome)

    assert 1.0 in seen, "the player to move should sometimes be the winner"
    assert seen <= {-1.0, 0.0, 1.0}
