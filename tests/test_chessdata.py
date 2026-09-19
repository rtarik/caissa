"""Tests for stored chess positions.

The stored form exists so training can encode positions in bulk, which puts a
second encoder in the project. It has to agree with the first exactly: a
network trained on one and played with the other is fed positions it never saw,
and nothing about that fails loudly.
"""

from __future__ import annotations

import chess
import numpy as np
import pytest

from caissa.data.chess import GAME, POSITION, board_of, encode, move_of, record, values
from caissa.games.chess import Chess, action_of, position


@pytest.fixture
def game() -> Chess:
    return Chess()


def store(pairs) -> np.ndarray:
    """Stored positions for (state, move) pairs, all from one game."""
    return np.array([record(state, move, 0, ply) for ply, (state, move) in enumerate(pairs)],
                    dtype=POSITION)


def random_play(game, games, seed, max_plies=160):
    """(state, move) pairs from random games: both colours, captures, castling,
    promotions and the odd en passant, as random games produce them."""
    rng = np.random.default_rng(seed)
    pairs = []
    for _ in range(games):
        state = game.initial_state()
        while game.terminal_value(state) is None and state.board.ply() < max_plies:
            move = state.legal_moves[int(rng.integers(len(state.legal_moves)))]
            pairs.append((state, move))
            state = game.apply(state, action_of(state.board, move))
    return pairs


def played(game, fen, *moves):
    """(state, move) pairs along a line of UCI moves from a position."""
    state = position(fen) if fen else game.initial_state()
    pairs = []
    for uci in moves:
        move = chess.Move.from_uci(uci)
        pairs.append((state, move))
        state = game.apply(state, action_of(state.board, move))
    return pairs


def assert_encoded_alike(game, pairs):
    planes, actions = encode(store(pairs))
    for index, (state, move) in enumerate(pairs):
        np.testing.assert_array_equal(planes[index], game.encode(state),
                                      err_msg=f"planes differ in {state.board.fen()}")
        assert actions[index] == action_of(state.board, move), state.board.fen()


def test_stored_positions_encode_exactly_as_live_ones(game):
    pairs = random_play(game, 25, seed=0)
    assert len(pairs) > 2000
    assert_encoded_alike(game, pairs)


def test_en_passant_both_ways(game):
    assert_encoded_alike(game, played(game, None, "e2e4", "a7a6", "e4e5", "d7d5", "e5d6"))
    assert_encoded_alike(game, played(game, None, "a2a3", "e7e5", "a3a4", "e5e4", "d2d4", "e4d3"))


def test_castling_both_ways_for_both_colours(game):
    fen = "r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w KQkq - 0 1"
    assert_encoded_alike(game, played(game, fen, "e1g1", "e8c8"))
    assert_encoded_alike(game, played(game, fen, "e1c1", "e8g8"))


def test_every_promotion_for_both_colours(game):
    for piece in "qrbn":
        assert_encoded_alike(game, played(game, "r1r1k3/1P6/8/8/8/8/8/4K3 w - - 0 1", f"b7a8{piece}"))
        assert_encoded_alike(game, played(game, "4k3/8/8/8/8/8/1p6/R1R1K3 b - - 0 1", f"b2b1{piece}"))


def test_repetition_and_the_fifty_move_counter(game):
    shuffle = ("g1f3", "g8f6", "f3g1", "f6g8")
    pairs = played(game, None, *shuffle, *shuffle)
    assert store(pairs)["repetitions"].max() == 2
    assert_encoded_alike(game, pairs)
    assert_encoded_alike(game, played(game, "4k3/8/8/8/8/8/8/R3K3 w - - 98 70", "a1a2", "e8d8"))


def test_a_stored_position_reads_back_as_the_same_board(game):
    """So a converted month can be spot-checked with python-chess."""
    pairs = random_play(game, 5, seed=1)
    for stored, (state, move) in zip(store(pairs), pairs):
        board = board_of(stored)
        assert board.fen().split()[:4] == state.board.fen().split()[:4]
        assert move_of(stored) == move
        assert move in board.legal_moves


def test_values_are_the_result_for_the_player_to_move(game):
    games = np.zeros(2, dtype=GAME)
    games["result"] = [1, -1]  # White won the first, Black the second
    positions = store(played(game, None, "e2e4", "e7e5", "g1f3", "b8c6"))
    positions["game"] = [0, 0, 1, 1]
    # Plies alternate White, Black: the winner's positions are +1, the loser's -1.
    np.testing.assert_array_equal(values(positions, games), [1, -1, -1, 1])
