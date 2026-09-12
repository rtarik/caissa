"""Game implementations satisfying the :class:`caissa.games.base.Game` protocol."""

from caissa.games.base import Game
from caissa.games.connect4 import Connect4

#: Registry used by the training scripts and, later, the web export.
GAMES: dict[str, type] = {Connect4.name: Connect4}

__all__ = ["GAMES", "Connect4", "Game"]
