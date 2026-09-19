"""Tests for chess.

The rules themselves belong to python-chess and are not re-tested here. What is
ours - and where silent bugs would live - is the translation: 4,672 action
indices that must name each legal move exactly once, a board that must look the
same to Black as its mirror image looks to White, and the draws the framework
has to recognise as draws.
"""

from __future__ import annotations

import chess
import numpy as np
import pytest

from caissa.games.base import Game
from caissa.games.chess import (
    ACTIONS,
    KNIGHT_TYPES,
    MOVE_TYPES,
    MOVES,
    QUEEN_TYPES,
    SQUARES,
    Chess,
    action_of,
    move_of,
    position,
)


@pytest.fixture
def game() -> Chess:
    return Chess()


def play(game, state, *moves):
    """Apply moves written in UCI notation, through the action encoding."""
    for uci in moves:
        state = game.apply(state, action_of(state.board, chess.Move.from_uci(uci)))
    return state


def random_positions(game, games, seed, max_plies=120):
    """Every position of some random games, both colours to move."""
    rng = np.random.default_rng(seed)
    for _ in range(games):
        state = game.initial_state()
        while game.terminal_value(state) is None and state.board.ply() < max_plies:
            yield state
            state = game.apply(state, int(rng.choice(np.flatnonzero(game.legal_actions(state)))))


def test_satisfies_game_protocol(game):
    assert isinstance(game, Game)


# ------------------------------------------------------------------- actions


def test_seventy_three_move_types_from_each_of_sixty_four_squares(game):
    assert MOVE_TYPES == 73
    assert game.action_size == ACTIONS == MOVE_TYPES * SQUARES == 4672


def test_exactly_1858_actions_stay_on_the_board():
    """Leela Chess Zero's policy map arrives at the same number: 1,456 queen-like
    moves, 336 knight jumps and 66 underpromotions over the empty board. A slip
    anywhere in the table's geometry would change one of them."""
    def on_board(types):
        return sum(MOVES[t * SQUARES + s] is not None for t in types for s in range(SQUARES))

    assert on_board(range(QUEEN_TYPES)) == 1456
    assert on_board(range(QUEEN_TYPES, QUEEN_TYPES + KNIGHT_TYPES)) == 336
    assert on_board(range(QUEEN_TYPES + KNIGHT_TYPES, MOVE_TYPES)) == 66
    assert sum(move is not None for move in MOVES) == 1858


def test_every_legal_move_has_its_own_index_and_back(game):
    """Two moves sharing an index would make the network unable to tell them
    apart; an index decoding to a different move would make search play a move
    it never evaluated. Checked across both colours in thousands of positions."""
    positions = 0
    for state in random_positions(game, 60, seed=1):
        board = state.board
        indices = [action_of(board, move) for move in state.legal_moves]
        assert len(set(indices)) == len(indices)
        assert [move_of(board, index) for index in indices] == state.legal_moves
        mask = game.legal_actions(state)
        assert mask.sum() == len(indices) and mask[indices].all()
        positions += 1
    assert positions > 3000


def test_black_sees_the_board_as_white_would(game):
    """Canonical perspective: a position must encode exactly like its
    colour-mirrored twin with the other side to move, and offer the same
    actions. This is what lets one network play both colours."""
    for state in random_positions(game, 30, seed=2):
        here = position(state.board.fen())
        twin = position(state.board.mirror().fen())
        assert here.board.turn != twin.board.turn
        np.testing.assert_array_equal(game.encode(here), game.encode(twin))
        np.testing.assert_array_equal(game.legal_actions(here), game.legal_actions(twin))


def test_castling_is_the_king_moving_two_squares(game):
    state = position("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1")
    east, west = 2, 6  # indexes into QUEEN_DIRECTIONS
    kingside = action_of(state.board, chess.Move.from_uci("e1g1"))
    queenside = action_of(state.board, chess.Move.from_uci("e1c1"))
    assert kingside == (east * 7 + 1) * SQUARES + chess.E1
    assert queenside == (west * 7 + 1) * SQUARES + chess.E1

    # Black castles with the same actions: from its side, too, the king starts
    # on e1 and moves two squares.
    black = position("r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 0 1")
    assert action_of(black.board, chess.Move.from_uci("e8g8")) == kingside
    assert action_of(black.board, chess.Move.from_uci("e8c8")) == queenside

    after = game.apply(state, kingside)
    assert after.board.piece_at(chess.G1) == chess.Piece(chess.KING, chess.WHITE)
    assert after.board.piece_at(chess.F1) == chess.Piece(chess.ROOK, chess.WHITE)


def test_castling_out_of_or_through_check_is_never_offered(game):
    kingside = action_of(chess.Board(), chess.Move.from_uci("e1g1"))
    queenside = action_of(chess.Board(), chess.Move.from_uci("e1c1"))

    through = game.legal_actions(position("r3k2r/8/8/8/8/8/5r2/R3K2R w KQkq - 0 1"))
    assert not through[kingside]  # f1 is attacked
    assert through[queenside]

    out_of = game.legal_actions(position("r3k2r/8/8/8/8/8/4r3/R3K2R w KQkq - 0 1"))
    assert not out_of[kingside] and not out_of[queenside]


def test_every_promotion_has_its_own_action(game):
    """Straight on or capturing either way, to any of four pieces: twelve moves.
    Promotion to a queen is the queen-like move itself; the other three pieces
    have planes of their own."""
    state = position("r1r1k3/1P6/8/8/8/8/8/4K3 w - - 0 1")
    promotions = [move for move in state.legal_moves if move.promotion]
    assert len(promotions) == 12
    indices = {move.uci(): action_of(state.board, move) for move in promotions}
    assert len(set(indices.values())) == 12

    north, north_east, north_west = 0, 1, 7
    assert indices["b7b8q"] == (north * 7) * SQUARES + chess.B7
    assert indices["b7c8q"] == (north_east * 7) * SQUARES + chess.B7
    assert indices["b7a8q"] == (north_west * 7) * SQUARES + chess.B7
    underpromotion = QUEEN_TYPES + KNIGHT_TYPES
    assert indices["b7a8n"] == (underpromotion + 0) * SQUARES + chess.B7
    assert indices["b7b8r"] == (underpromotion + 5) * SQUARES + chess.B7
    assert indices["b7c8b"] == (underpromotion + 7) * SQUARES + chess.B7

    for uci, index in indices.items():
        assert move_of(state.board, index).uci() == uci
    knight = game.apply(state, indices["b7a8n"])
    assert knight.board.piece_at(chess.A8) == chess.Piece(chess.KNIGHT, chess.WHITE)

    # The same promotions for Black, one mirror away, use the same twelve actions.
    black = position("4k3/8/8/8/8/8/1p6/R1R1K3 b - - 0 1")
    assert {action_of(black.board, move) for move in black.legal_moves if move.promotion} \
        == set(indices.values())


def test_illegal_and_off_board_actions_are_rejected(game):
    state = game.initial_state()
    with pytest.raises(ValueError, match="not legal"):
        game.apply(state, action_of(state.board, chess.Move.from_uci("e2e5")))
    with pytest.raises(ValueError, match="no move"):
        game.apply(state, 0 * SQUARES + chess.H8)  # one step north from the top edge
    with pytest.raises(ValueError, match="no move"):
        game.apply(state, ACTIONS)


def test_apply_leaves_the_original_state_untouched(game):
    """Search keeps every state it creates, so a state must never change."""
    state = game.initial_state()
    fen, history = state.board.fen(), state.history
    play(game, state, "e2e4")
    assert state.board.fen() == fen
    assert state.history == history
    assert state.board.move_stack == []


def test_white_moves_first_and_turns_alternate(game):
    state = game.initial_state()
    assert game.to_play(state) == 0
    assert game.to_play(play(game, state, "e2e4")) == 1
    assert game.to_play(play(game, state, "e2e4", "e7e5")) == 0


def test_symmetries_are_the_identity_alone(game):
    encoded = game.encode(game.initial_state())
    policy = np.full(ACTIONS, 1 / ACTIONS)
    (only,) = game.symmetries(encoded, policy)
    assert only[0] is encoded and only[1] is policy


# ------------------------------------------------------------------ encoding


def test_the_opening_position_planes(game):
    planes = game.encode(game.initial_state())
    assert planes.shape == (game.input_planes, 8, 8) == (19, 8, 8)
    assert planes.dtype == np.float32
    assert planes[0, 1].all() and planes[0].sum() == 8    # mover's pawns: second rank
    assert planes[6, 6].all() and planes[6].sum() == 8    # opponent's: seventh
    assert planes[5, 0, 4] == 1 and planes[5].sum() == 1  # mover's king on e1
    assert planes[11, 7, 4] == 1                          # opponent's king on e8
    assert planes[12:16].all()                            # four castling rights
    assert not planes[16:].any()                          # no en passant, clock or repeat


def test_black_to_move_sees_its_own_pieces_at_the_bottom(game):
    planes = game.encode(play(game, game.initial_state(), "e2e4"))
    assert planes[0, 1].all() and planes[0].sum() == 8    # Black's pawns, mirrored down
    assert planes[5, 0, 4] == 1                           # Black's king on "e1"
    assert planes[6, 6].sum() == 7 and planes[6, 6, 4] == 0
    assert planes[6, 4, 4] == 1                           # White's e4 pawn, seen as e5
    assert not planes[16].any()                           # e3 is no capture: not marked


def test_en_passant_is_marked_only_when_the_capture_is_possible(game):
    white = play(game, game.initial_state(), "e2e4", "a7a6", "e4e5", "d7d5")
    assert game.encode(white)[16, 5, 3] == 1 and game.encode(white)[16].sum() == 1  # d6

    # Black's capture on d3 is marked on d6 too: in Black's own frame, that is
    # where it is.
    black = play(game, game.initial_state(), "a2a3", "e7e5", "a3a4", "e5e4", "d2d4")
    assert game.encode(black)[16, 5, 3] == 1 and game.encode(black)[16].sum() == 1

    captured = play(game, white, "e5d6")
    assert captured.board.piece_at(chess.D5) is None


def test_castling_rights_are_the_movers_first(game):
    state = play(game, game.initial_state(), "e2e4", "e7e5", "e1e2")  # White's king walks
    planes = game.encode(state)  # Black to move: its own rights first, both intact
    assert planes[12].all() and planes[13].all()
    assert not planes[14].any() and not planes[15].any()


def test_the_fifty_move_counter_is_a_plane(game):
    planes = game.encode(position("4k3/8/8/8/8/8/8/R3K3 w - - 98 70"))
    assert np.allclose(planes[17], 0.98)


# --------------------------------------------------------------- the results


def test_checkmate_is_a_loss_for_the_player_to_move(game):
    mated = play(game, game.initial_state(), "f2f3", "e7e5", "g2g4", "d8h4")
    assert game.terminal_value(mated) == -1.0
    assert not game.legal_actions(mated).any()


def test_stalemate_is_a_draw(game):
    stalemated = position("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1")
    assert game.terminal_value(stalemated) == 0.0
    assert not game.legal_actions(stalemated).any()


def test_insufficient_material_is_a_draw(game):
    assert game.terminal_value(position("8/8/8/4k3/8/8/8/4K3 w - - 0 1")) == 0.0
    assert game.terminal_value(position("8/8/8/4k3/8/8/8/4KB2 w - - 0 1")) == 0.0
    assert game.terminal_value(position("8/8/8/4k3/8/8/8/R3K3 w - - 0 1")) is None


def test_the_fifty_move_rule_draws_at_the_hundredth_half_move(game):
    state = position("4k3/8/8/8/8/8/8/R3K3 w - - 98 70")
    state = play(game, state, "a1a2")
    assert game.terminal_value(state) is None
    state = play(game, state, "e8d8")
    assert game.terminal_value(state) == 0.0


def test_mate_on_the_hundredth_half_move_still_wins(game):
    mated = play(game, position("k7/8/1K6/8/8/8/8/7R w - - 99 70"), "h1h8")
    assert mated.board.halfmove_clock == 100
    assert game.terminal_value(mated) == -1.0


def test_threefold_repetition_is_a_draw(game):
    shuffle = ("g1f3", "g8f6", "f3g1", "f6g8")
    twice = play(game, game.initial_state(), *shuffle)
    assert twice.repetitions == 2
    assert game.terminal_value(twice) is None
    assert game.encode(twice)[18].all()  # the network is told it has been here before

    thrice = play(game, twice, *shuffle)
    assert thrice.repetitions == 3
    assert game.terminal_value(thrice) == 0.0


def test_irreversible_moves_restart_the_repetition_count(game):
    state = play(game, game.initial_state(), "g1f3", "g8f6", "f3g1", "f6g8", "e2e4")
    assert len(state.history) == 1
    assert state.repetitions == 1
    assert not game.encode(state)[18].any()


def test_games_end_on_the_losers_turn(game):
    """As in Four in a Row, a win ends the game on the loser's move, so the result
    for the player to move is only ever a loss or a draw."""
    rng = np.random.default_rng(3)
    results = set()
    for _ in range(40):
        state = game.initial_state()
        while (result := game.terminal_value(state)) is None:
            state = game.apply(state, int(rng.choice(np.flatnonzero(game.legal_actions(state)))))
        results.add(result)
    assert results == {-1.0, 0.0}


def test_render_shows_the_board_and_whose_move_it_is(game):
    text = game.render(game.initial_state())
    assert "r n b q k b n r" in text
    assert "White to move" in text
