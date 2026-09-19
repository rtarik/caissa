"""Game implementations satisfying the :class:`caissa.games.base.Game` protocol."""

from caissa.games.base import Game
from caissa.games.chess import Chess
from caissa.games.connect4 import Connect4
from caissa.games.dotsandboxes import DotsAndBoxes
from caissa.games.gomoku import Gomoku
from caissa.games.isola import Isola
from caissa.games.reversi import Reversi

#: Registry used by the training scripts and the web export. Ordered as the
#: ladder in PLAN.md, each game adding one new difficulty to the framework.
GAMES: dict[str, type] = {
    g.name: g for g in (Connect4, Reversi, Gomoku, Isola, DotsAndBoxes, Chess)
}

__all__ = ["GAMES", "Chess", "Connect4", "DotsAndBoxes", "Game", "Gomoku", "Isola", "Reversi"]
