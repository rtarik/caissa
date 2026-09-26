/**
 * The games you have played, kept in this browser.
 *
 * Saved as they are played, not when they end, so a reload never loses one and
 * an unfinished game can still be exported. Stored as the engine's own move
 * numbers plus what the game was - who played which side, against which level
 * - rather than as PGN: the record is the fact, and PGN is one way of writing it
 * out, regenerated whenever it is asked for.
 *
 * Browser storage can be missing (a private window), full, blocked, or edited
 * by hand, and none of that is worth breaking a game over. Every access is
 * guarded, anything unreadable is skipped, and a failed save leaves play alone.
 */

export type Result = "1-0" | "0-1" | "1/2-1/2" | "*";

export interface GameRecord {
  id: string;
  /** The game's key, "chess" for now. */
  game: string;
  /** When the first move was made, as an ISO timestamp. */
  started: string;
  /** When this record was last written. */
  updated: string;
  /** The moves, as the engine numbers them. */
  moves: number[];
  /** A FEN when the game began from a position other than the usual one. */
  start?: string;
  humanWhite: boolean;
  /** The level's name, "Master", and how hard it searched. */
  level: string;
  simulations: number;
  /** The network played, by file, and as the page names it. */
  network: string;
  networkLabel: string;
  /** The level's measured rating when the game was played, if it had one. */
  rating?: number;
  result: Result;
  /** Why the game ended, in words: "checkmate", "resignation", ... */
  termination?: string;
}

/** The part of the Web Storage API this needs, so tests can hand in their own. */
export interface Store {
  getItem(key: string): string | null;
  setItem(key: string, value: string): void;
  removeItem(key: string): void;
}

const KEY = "caissa:games";

/**
 * How many games to keep. A chess record is a kilobyte or two, so two hundred
 * is well inside what browsers allow and far more than a list is useful for.
 */
export const LIMIT = 200;

/** The page's storage, or null where there is none to be had. */
export function browserStore(): Store | null {
  try {
    return typeof localStorage === "undefined" ? null : localStorage;
  } catch {
    return null;
  }
}

function looksLikeARecord(value: unknown): value is GameRecord {
  const record = value as GameRecord;
  return typeof record === "object" && record !== null
    && typeof record.id === "string" && typeof record.game === "string"
    && Array.isArray(record.moves) && record.moves.every((m) => Number.isInteger(m))
    && typeof record.humanWhite === "boolean" && typeof record.started === "string";
}

/** Every saved game, newest first; nothing at all if storage cannot be read. */
export function loadGames(store: Store | null): GameRecord[] {
  if (!store) return [];
  try {
    const parsed: unknown = JSON.parse(store.getItem(KEY) ?? "[]");
    return Array.isArray(parsed) ? parsed.filter(looksLikeARecord) : [];
  } catch {
    return [];
  }
}

function write(store: Store, games: GameRecord[]): boolean {
  try {
    store.setItem(KEY, JSON.stringify(games));
    return true;
  } catch {
    return false;
  }
}

/**
 * Save a game, replacing any earlier save of the same game, newest first.
 *
 * Replacing is what makes saving on every move safe: an undo or a finished game
 * rewrites the one record rather than adding another. Returns whether the save
 * landed; a full or blocked store is reported, never thrown.
 */
export function saveGame(store: Store | null, record: GameRecord): boolean {
  if (!store) return false;
  const others = loadGames(store).filter((game) => game.id !== record.id);
  return write(store, [record, ...others].slice(0, LIMIT));
}

export function deleteGame(store: Store | null, id: string): void {
  if (!store) return;
  write(store, loadGames(store).filter((game) => game.id !== id));
}

export function clearGames(store: Store | null): void {
  if (!store) return;
  try {
    store.removeItem(KEY);
  } catch {
    // Nothing to do: a store that cannot be written to has nothing of ours in it.
  }
}

/** A fresh id for a new game, unique enough for one browser's history. */
export function newGameId(random: () => number = Math.random, now: () => number = Date.now): string {
  return `${now().toString(36)}-${Math.floor(random() * 36 ** 6).toString(36).padStart(6, "0")}`;
}
