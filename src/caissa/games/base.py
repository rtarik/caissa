"""The contract every game must satisfy for AlphaZero to play it.

The search and training code never imports a concrete game. It only ever sees
this interface, which is what lets the same agent learn Connect 4, Reversi or
chess without changing a line of the algorithm.

Two conventions run through everything here:

*Canonical perspective* - a state is always described from the point of view of
the player about to move. Their pieces are +1, the opponent's are -1. Applying a
move flips the board's sign, so the next player sees the world the same way. The
network therefore learns one function rather than one per side, and every
training sample teaches both sides at once.

*Mover-relative values* - every value in this codebase answers "how good is this
for the player to move?". +1 is a win for them, -1 a loss, 0 a draw. Mixing this
up with an absolute "good for player one" convention is the classic way to build
an agent that trains hard toward losing.
"""

from __future__ import annotations

from typing import Protocol, TypeVar, runtime_checkable

import numpy as np

# Each game defines its own state type; the algorithm only passes them around.
State = TypeVar("State")


@runtime_checkable
class Game(Protocol[State]):
    """Rules of a two-player, perfect-information, zero-sum game."""

    #: Short identifier, used in file paths and the web UI.
    name: str
    #: Number of distinct actions. Network policy heads emit this many logits.
    action_size: int
    #: (height, width) of the board, for shaping convolutions.
    board_shape: tuple[int, int]
    #: Number of feature planes produced by :meth:`encode`.
    input_planes: int

    def initial_state(self) -> State:
        """The position the game starts from."""

    def legal_actions(self, state: State) -> np.ndarray:
        """Boolean mask of shape ``(action_size,)``, True where the move is legal.

        A mask rather than a list: the network emits a logit per action, and we
        set illegal ones to -inf before the softmax. That keeps probability mass
        off moves that do not exist without the network having to learn to avoid
        them.
        """

    def apply(self, state: State, action: int) -> State:
        """Return the state after ``action``, seen by the *next* player.

        States are treated as immutable so search can hold on to them without
        defensive copying.
        """

    def terminal_value(self, state: State) -> float | None:
        """``None`` if the game continues, otherwise the result for the mover.

        Note that in most games this is never +1: if the player to move had just
        won, the game would already have ended on the previous ply. Seeing only
        -1 and 0 here is expected, not a bug.
        """

    def encode(self, state: State) -> np.ndarray:
        """Feature planes of shape ``(input_planes, *board_shape)``, float32.

        This is the network's input, and it is always in canonical perspective.
        """

    def symmetries(
        self, encoded: np.ndarray, policy: np.ndarray
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        """Equivalent (position, policy) pairs under the board's symmetries.

        Free training data. A Connect 4 position mirrored left-to-right is a
        different array but the same game, so one self-play position can teach
        the network twice. Square boards usually admit eight variants, which is
        a large multiplier on sample efficiency when games are expensive.

        Must include the identity. The policy has to be permuted the same way as
        the board, which is where the subtle bugs live.
        """

    def render(self, state: State) -> str:
        """Human-readable board, for debugging and playing in a terminal."""
