import { Connect4 } from "./connect4";
import { Reversi } from "./reversi";
import type { Game } from "./types";

/** Mirrors `GAMES` in `src/caissa/games/__init__.py`. */
export const GAMES: Record<string, () => Game<unknown>> = {
  connect4: () => new Connect4() as Game<unknown>,
  reversi: () => new Reversi() as Game<unknown>,
};

export const TITLES: Record<string, string> = {
  connect4: "Connect 4",
  reversi: "Reversi",
};

export function createGame(name: string): Game<unknown> {
  const make = GAMES[name];
  if (!make) throw new Error(`unknown game: ${name}`);
  return make();
}
