/**
 * The opening book and the opening names, as the chess page uses them.
 *
 * Both are keyed by position - the first four fields of a FEN, which is also the
 * chess state's repetition key - so a position is found however it was reached.
 * Built offline by `scripts/book.py` from our own 2200+ games and by
 * `scripts/openings.py` from Lichess's public-domain list of named openings.
 */
import type { ChessState } from "./games/chess";

/** Position key -> the moves strong players chose there, as [uci, games]. */
export type Book = Map<string, readonly (readonly [string, number])[]>;
/** Position key -> [ECO code, name]. */
export type Names = Map<string, readonly [string, string]>;

export interface BookMove {
  uci: string;
  /** The share of strong players' games in this position that went this way. */
  share: number;
  /** How many games reached this position. */
  games: number;
}

/** What identifies a position for the book and the names: the FEN's first four fields. */
export function positionKey(state: ChessState): string {
  return state.fen.split(" ").slice(0, 4).join(" ");
}

/**
 * A move from the book, chosen as strong players chose it, or null out of book.
 *
 * Sampled in proportion to how often each move was played: the main line most
 * of the time, every respectable alternative some of the time. That is where the
 * variety comes from, and it costs nothing in strength - every move in the book
 * was played there by strong players many times over. `random` is injectable so
 * a test can pin the draw.
 */
export function bookMove(state: ChessState, book: Book, random: () => number = Math.random): BookMove | null {
  const entries = book.get(positionKey(state));
  if (!entries?.length) return null;
  const games = entries.reduce((sum, [, count]) => sum + count, 0);
  let draw = random() * games;
  for (const [uci, count] of entries) {
    draw -= count;
    if (draw < 0) return { uci, share: count / games, games };
  }
  const [uci, count] = entries[entries.length - 1];
  return { uci, share: count / games, games };
}

/**
 * The opening a game is in: the name of the last named position it passed
 * through, or null before it has reached one.
 *
 * The last, not the first: names grow more specific as a line goes deeper, so
 * the deepest named position is the most precise description - and a game that
 * wanders off the named paths keeps the last name it had, as players describe it.
 */
export function openingOf(states: readonly ChessState[], names: Names): readonly [string, string] | null {
  let found: readonly [string, string] | null = null;
  for (const state of states) found = names.get(positionKey(state)) ?? found;
  return found;
}

/** Read the files the build scripts write. */
export function parseBook(data: { positions: Record<string, [string, number][]> }): Book {
  return new Map(Object.entries(data.positions));
}

export function parseNames(data: { positions: Record<string, [string, string]> }): Names {
  return new Map(Object.entries(data.positions));
}
