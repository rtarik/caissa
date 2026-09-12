"""Game implementations satisfying the :class:`caissa.games.base.Game` protocol."""

from caissa.games.base import Game
from caissa.games.connect4 import Connect4
from caissa.games.reversi import Reversi

#: Registry used by the training scripts and the web export.
GAMES: dict[str, type] = {g.name: g for g in (Connect4, Reversi)}

__all__ = ["GAMES", "Connect4", "Game", "Reversi"]
