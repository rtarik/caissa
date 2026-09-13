import { Connect4 } from "./connect4";
import { Gomoku } from "./gomoku";
import { Isola } from "./isola";
import { Reversi } from "./reversi";
import type { Game } from "./types";

/** Mirrors `GAMES` in `src/caissa/games/__init__.py`. */
export const GAMES: Record<string, () => Game<unknown>> = {
  connect4: () => new Connect4() as Game<unknown>,
  reversi: () => new Reversi() as Game<unknown>,
  gomoku: () => new Gomoku() as Game<unknown>,
  isola: () => new Isola() as Game<unknown>,
};

export function createGame(name: string): Game<unknown> {
  const make = GAMES[name];
  if (!make) throw new Error(`unknown game: ${name}`);
  return make();
}
