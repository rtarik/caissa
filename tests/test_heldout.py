"""Tests for the held-out human exam.

This is the measurement self-play is judged against for forgetting, and it is the
kind of measurement that fails quietly: an exam that grades the wrong positions,
or grades an argmax over moves the rules forbid, still prints a plausible
percentage that moves in plausible ways. So the parts that could be plausibly
wrong - which games it draws from, what counts as a correct move, how the phases
are cut, whose result the value is - are pinned here against hand-built data
whose right answers are known in advance.
"""

from __future__ import annotations

import chess
import numpy as np
import pytest
import torch

from caissa.data.chess import (GAME, POSITION, board_of, human_samples,
                               load_months, move_of, record)
from caissa.data.heldout import PHASES, Validation
from caissa.games.chess import ACTIONS, Chess, action_of

#: Four plies of one opening. Every made-up game here plays it, so the game a
#: position came from is the only thing that tells two of them apart - which is
#: exactly what the validation split works on.
OPENING = ("e2e4", "e7e5", "g1f3", "b8c6")


@pytest.fixture
def game() -> Chess:
    return Chess()


def write_month(root, name, game, results):
    """Write a month of identical short games with the given results."""
    positions, games = [], np.zeros(len(results), dtype=GAME)
    for number, result in enumerate(results):
        games[number] = (result, 2400, 2400, 300, 0, len(OPENING),
                         number * len(OPENING), 0)
        state = game.initial_state()
        for ply, uci in enumerate(OPENING):
            move = chess.Move.from_uci(uci)
            positions.append(record(state, move, number, ply))
            state = game.apply(state, action_of(state.board, move))
    path = root / name
    path.mkdir(parents=True)
    np.save(path / "positions.npy", np.array(positions, dtype=POSITION))
    np.save(path / "games.npy", games)


def one_month(root, game, results):
    """A month written and loaded back, with every game held out."""
    write_month(root, "2020-01", game, results)
    positions, games = load_months(root, ["2020-01"])
    games["validation"] = 1
    return positions, games


def exam_of(root, game, results, count=None):
    positions, games = one_month(root, game, results)
    count = count if count is not None else len(positions)
    return Validation(positions, games, count, np.random.default_rng(0))


def one_hot(exam, rows=None):
    """Logits whose highest value is the move the human played."""
    rows = rows if rows is not None else len(exam.actions)
    logits = torch.zeros(rows, ACTIONS)
    logits[np.arange(rows), exam.actions[:rows]] = 10.0
    return logits


def another_legal_move(exam, row):
    legal = exam.legal[exam.offsets[row]:exam.offsets[row + 1]]
    return next(action for action in legal.tolist() if action != exam.actions[row])


class Fixed:
    """A network whose logits are handed to it, so its score is known."""

    def __init__(self, logits, value=0.0):
        self.logits = torch.as_tensor(logits, dtype=torch.float32)
        self.value = value

    def __call__(self, planes):
        rows = len(planes)
        return (self.logits[:rows].clone(),
                torch.full((rows,), self.value, dtype=torch.float32))

    def eval(self):
        pass

    def train(self):
        pass


def grade(exam, net):
    return exam.measure(net, torch.device("cpu"))


# ------------------------------------------------------------- joining months


def test_months_are_renumbered_into_one_table(tmp_path, game):
    """Each month is stored on its own, numbered from zero.

    Concatenated without renumbering, the second month's positions point at the
    first month's games - so a position would carry the result of a different
    game entirely, and the exam would run on happily.
    """
    write_month(tmp_path, "2020-01", game, [1, 1])
    write_month(tmp_path, "2020-02", game, [-1, -1])

    positions, games = load_months(tmp_path, ["2020-01", "2020-02"])

    assert len(games) == 4
    assert list(games["result"]) == [1, 1, -1, -1]
    assert sorted(set(positions["game"].tolist())) == [0, 1, 2, 3]
    # Each game's recorded first row points back at its own positions.
    for number, first in enumerate(games["first"]):
        assert positions["game"][int(first)] == number


def test_one_month_is_left_as_it_is(tmp_path, game):
    write_month(tmp_path, "2020-01", game, [1, 0, -1])
    positions, games = load_months(tmp_path, ["2020-01"])
    assert list(positions["game"]) == [0] * 4 + [1] * 4 + [2] * 4
    assert list(games["first"]) == [0, 4, 8]


# --------------------------------------------------------- drawing the sample


def test_the_exam_only_ever_draws_held_out_games(tmp_path, game):
    """Grading on training positions measures memorisation, not knowledge."""
    write_month(tmp_path, "2020-01", game, [1, 0, 1, 0, 1, 1])
    positions, games = load_months(tmp_path, ["2020-01"])
    # The two held-out games are the drawn ones, so a sample that strayed into
    # the training games would carry a value of +/-1 rather than 0.
    games["validation"] = [0, 1, 0, 1, 0, 0]

    exam = Validation(positions, games, 100, np.random.default_rng(0))

    assert len(exam.actions) == 8, "every held-out position, and nothing else"
    assert set(exam.values.tolist()) == {0.0}


def test_the_sample_is_capped_and_reproducible(tmp_path, game):
    """The same seed must give the same exam, or two stages are not comparable."""
    positions, games = one_month(tmp_path, game, [1] * 10)

    first = Validation(positions, games, 12, np.random.default_rng(3))
    again = Validation(positions, games, 12, np.random.default_rng(3))
    other = Validation(positions, games, 12, np.random.default_rng(4))

    assert len(first.actions) == 12
    np.testing.assert_array_equal(first.plies, again.plies)
    assert not np.array_equal(first.plies, other.plies)


# -------------------------------------------------------------------- grading


def test_a_network_that_plays_the_human_move_scores_everything(tmp_path, game):
    exam = exam_of(tmp_path, game, [1])
    assert grade(exam, Fixed(one_hot(exam)))["accuracy"] == 1.0


def test_a_network_that_prefers_another_legal_move_scores_nothing(tmp_path, game):
    """The alternative is legal, so this is a real disagreement, not a mask."""
    exam = exam_of(tmp_path, game, [1])
    logits = torch.zeros(len(exam.actions), ACTIONS)
    for row in range(len(exam.actions)):
        logits[row, another_legal_move(exam, row)] = 10.0

    assert grade(exam, Fixed(logits))["accuracy"] == 0.0


def test_illegal_moves_are_masked_out_before_the_argmax(tmp_path, game):
    """Play never lets the network choose an illegal move, so grading must not.

    Here the highest logit everywhere is an illegal move and the best *legal* one
    is what the human played. An unmasked argmax scores zero; the exam must score
    one, because that is the move the network would actually make.
    """
    exam = exam_of(tmp_path, game, [1])
    logits = one_hot(exam)
    for row in range(len(exam.actions)):
        legal = set(exam.legal[exam.offsets[row]:exam.offsets[row + 1]].tolist())
        logits[row, next(a for a in range(ACTIONS) if a not in legal)] = 100.0

    assert grade(exam, Fixed(logits))["accuracy"] == 1.0


def test_phases_are_cut_by_ply(tmp_path, game):
    """A move in an endgame is a different exam from one in the opening.

    The breakdown is how a stage shows *where* it changed - self-play tends to
    keep the endgames it can search and lose the opening theory it was told.
    """
    positions, games = one_month(tmp_path, game, [1])
    positions["ply"] = [0, 19, 20, 70]  # opening, opening, middlegame, endgame
    exam = Validation(positions, games, 4, np.random.default_rng(0))

    logits = one_hot(exam)
    logits[0] = 0.0  # wrong on the first opening position, right on the rest
    logits[0, another_legal_move(exam, 0)] = 10.0

    result = grade(exam, Fixed(logits))
    assert result["accuracy"] == 0.75
    assert result["accuracy_opening"] == 0.5
    assert result["accuracy_middlegame"] == 1.0
    assert result["accuracy_endgame"] == 1.0
    assert {name for name, _, _ in PHASES} == {"opening", "middlegame", "endgame"}


def test_the_value_head_is_graded_from_the_mover_s_side(tmp_path, game):
    """White won, so White's positions are +1 and Black's -1.

    A network answering +1 everywhere is right on exactly half of them. A grader
    that forgot to flip for the side to move would call it perfect.
    """
    exam = exam_of(tmp_path, game, [1])

    result = grade(exam, Fixed(torch.zeros(len(exam.actions), ACTIONS), value=1.0))

    assert result["value_sign"] == 0.5
    assert result["value_loss"] == pytest.approx(2.0)  # 0, 4, 0, 4 - halved


def test_drawn_games_are_left_out_of_the_sign_score(tmp_path, game):
    """A draw has no winner to get right; counting it would dilute the number."""
    positions, games = one_month(tmp_path, game, [1, 0])
    exam = Validation(positions, games, len(positions), np.random.default_rng(0))

    result = grade(exam, Fixed(torch.zeros(len(exam.actions), ACTIONS), value=1.0))

    assert result["value_sign"] == 0.5, "half of the decisive game, not a quarter of all"
    assert result["value_loss"] == pytest.approx(1.5)  # (0+4+0+4 + 1+1+1+1) / 8


# ------------------------------------------- the other side of the split: rehearsal


def test_rehearsal_never_draws_from_the_held_out_games(tmp_path, game):
    """The exam is what says whether rehearsal worked.

    Training on its positions would make the measurement report memorisation
    instead - and report it as success, since the numbers would go up.
    """
    write_month(tmp_path, "2020-01", game, [1, 0, -1, 0, 1, -1])
    positions, games = load_months(tmp_path, ["2020-01"])
    # The held-out games are the drawn ones, so anything with value 0 leaked.
    games["validation"] = [0, 1, 0, 1, 0, 0]

    samples = human_samples(positions, games, 100, np.random.default_rng(0))

    assert len(samples) == 16, "every training position, and no other"
    assert all(sample.value != 0.0 for sample in samples)


def test_rehearsal_policy_targets_are_the_move_the_human_played(tmp_path, game):
    """A search that visited one move and nothing else - which is the claim being
    made: this move was played, by someone strong, in this position."""
    positions, games = one_month(tmp_path, game, [1])
    games["validation"] = 0

    samples = human_samples(positions, games, 4, np.random.default_rng(0))

    for stored, sample in zip(positions, samples):
        assert sample.policy.sum() == pytest.approx(1.0)
        assert int(sample.policy.argmax()) == action_of(board_of(stored), move_of(stored))


def test_rehearsal_values_are_the_result_for_whoever_is_to_move(tmp_path, game):
    """White won this game, and the plies alternate, so the labels must too."""
    positions, games = one_month(tmp_path, game, [1])
    games["validation"] = 0

    samples = human_samples(positions, games, 4, np.random.default_rng(0))

    assert [sample.value for sample in samples] == [1.0, -1.0, 1.0, -1.0]


def test_rehearsal_asks_for_more_than_exists(tmp_path, game):
    positions, games = one_month(tmp_path, game, [1, -1])
    games["validation"] = 0
    assert len(human_samples(positions, games, 1_000, np.random.default_rng(0))) == 8
