/**
 * The saved-games history and the PGN it exports.
 *
 * Two kinds of failure matter here. Storage: browser storage can be missing,
 * full or edited by hand, and none of that may break a game or lose the rest of
 * the history. PGN: a file an analysis tool rejects - or worse, reads as a
 * different game - is the whole feature failing, so every export is read back
 * by chess.js's own PGN reader and checked move for move.
 */
import { describe, expect, it } from "vitest";
import { Chess as Board } from "chess.js";
import {
  LIMIT, clearGames, deleteGame, loadGames, newGameId, saveGame,
  type GameRecord, type Store,
} from "../src/archive";
import { Chess, ChessState, actionOf } from "../src/games/chess";
import { chessOutcome, collectionPgn, movesText, pgnFileName, recordPgn } from "../src/pgn";

/** A Map with the Storage interface, and a switch to make it misbehave. */
function memoryStore(fail: "none" | "writes" | "all" = "none"): Store & { data: Map<string, string> } {
  const data = new Map<string, string>();
  return {
    data,
    getItem: (key) => {
      if (fail === "all") throw new Error("storage is off");
      return data.get(key) ?? null;
    },
    setItem: (key, value) => {
      if (fail !== "none") throw new Error("quota exceeded");
      data.set(key, value);
    },
    removeItem: (key) => {
      if (fail !== "none") throw new Error("storage is off");
      data.delete(key);
    },
  };
}

/** Moves in SAN, as the engine's action numbers. */
function actions(sans: string[], start?: string): number[] {
  const rules = new Chess();
  const board = new Board(start);
  let state = start ? ChessState.create(start) : rules.initialState();
  const out: number[] = [];
  for (const san of sans) {
    const move = board.move(san);
    const action = actionOf(state, { from: move.from, to: move.to, promotion: move.promotion ?? null });
    out.push(action);
    state = rules.apply(state, action);
  }
  return out;
}

function record(overrides: Partial<GameRecord> = {}): GameRecord {
  return {
    id: "game-1",
    game: "chess",
    started: "2026-09-26T14:03:00.000Z",
    updated: "2026-09-26T14:20:00.000Z",
    moves: actions(["e4", "c5", "Nf3", "d6"]),
    humanWhite: true,
    level: "Master",
    simulations: 600,
    network: "chess-imitation1",
    networkLabel: "Imitation 1",
    result: "*",
    ...overrides,
  };
}

describe("the history", () => {
  it("keeps one record per game, rewriting it as the game goes on", () => {
    // Saved after every move: the same game must stay one entry, not forty.
    const store = memoryStore();
    saveGame(store, record({ moves: actions(["e4"]) }));
    saveGame(store, record({ moves: actions(["e4", "c5"]) }));
    const saved = loadGames(store);
    expect(saved).toHaveLength(1);
    expect(saved[0].moves).toHaveLength(2);
  });

  it("lists the most recently played game first", () => {
    const store = memoryStore();
    saveGame(store, record({ id: "a" }));
    saveGame(store, record({ id: "b" }));
    saveGame(store, record({ id: "a", result: "1-0" }));
    expect(loadGames(store).map((game) => game.id)).toEqual(["a", "b"]);
  });

  it("forgets the oldest games beyond its limit", () => {
    const store = memoryStore();
    for (let index = 0; index < LIMIT + 5; index++) saveGame(store, record({ id: `g${index}` }));
    const saved = loadGames(store);
    expect(saved).toHaveLength(LIMIT);
    expect(saved[0].id).toBe(`g${LIMIT + 4}`);
    expect(saved.some((game) => game.id === "g0")).toBe(false);
  });

  it("deletes one game, or all of them", () => {
    const store = memoryStore();
    saveGame(store, record({ id: "a" }));
    saveGame(store, record({ id: "b" }));
    deleteGame(store, "a");
    expect(loadGames(store).map((game) => game.id)).toEqual(["b"]);
    clearGames(store);
    expect(loadGames(store)).toEqual([]);
  });

  it("skips entries it cannot read instead of losing the rest", () => {
    const store = memoryStore();
    store.data.set("caissa:games", JSON.stringify([
      { id: "broken" },
      record({ id: "fine" }),
      { ...record({ id: "bad-moves" }), moves: ["e4"] },
    ]));
    expect(loadGames(store).map((game) => game.id)).toEqual(["fine"]);
  });

  it("treats garbage as an empty history", () => {
    const store = memoryStore();
    store.data.set("caissa:games", "{not json");
    expect(loadGames(store)).toEqual([]);
  });

  it("never throws when storage refuses, and says the save did not land", () => {
    // A private window, a full disk, a blocked site: play must carry on.
    expect(saveGame(memoryStore("writes"), record())).toBe(false);
    expect(loadGames(memoryStore("all"))).toEqual([]);
    expect(() => deleteGame(memoryStore("all"), "x")).not.toThrow();
    expect(() => clearGames(memoryStore("all"))).not.toThrow();
    expect(saveGame(null, record())).toBe(false);
  });

  it("gives new games ids that differ", () => {
    const ids = new Set(Array.from({ length: 50 }, () => newGameId()));
    expect(ids.size).toBe(50);
  });
});

describe("how a game ended", () => {
  const rules = new Chess();
  const after = (sans: string[], start?: string) => {
    let state = start ? ChessState.create(start) : rules.initialState();
    for (const action of actions(sans, start)) state = rules.apply(state, action);
    return state;
  };

  it("calls checkmate for the side that delivered it", () => {
    const mate = after(["f3", "e5", "g4", "Qh4#"]);
    expect(chessOutcome(mate, true, false)).toEqual({ result: "0-1", termination: "checkmate" });
  });

  it("calls stalemate a draw", () => {
    const stalemate = ChessState.create("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1");
    expect(chessOutcome(stalemate, true, false)).toEqual({ result: "1/2-1/2", termination: "stalemate" });
  });

  it("gives a resigned game to the other side, whatever the position", () => {
    const start = rules.initialState();
    expect(chessOutcome(start, true, true)).toEqual({ result: "0-1", termination: "resignation" });
    expect(chessOutcome(start, false, true)).toEqual({ result: "1-0", termination: "resignation" });
  });

  it("leaves a game in progress unresolved", () => {
    expect(chessOutcome(after(["e4", "e5"]), true, false)).toEqual({ result: "*" });
  });
});

describe("the exported PGN", () => {
  it("reads back as the same game, player names and all", () => {
    const pgn = recordPgn(record({ result: "*" }));
    const board = new Board();
    board.loadPgn(pgn);
    expect(board.history()).toEqual(["e4", "c5", "Nf3", "d6"]);
    const headers = board.getHeaders();
    expect(headers.White).toBe("You");
    expect(headers.Black).toBe("Caissa (Master, Imitation 1)");
    expect(headers.Date).toBe("2026.09.26");
    expect(headers.Result).toBe("*");
  });

  it("puts the players the right way round when you had Black", () => {
    const headers = new Board();
    headers.loadPgn(recordPgn(record({ humanWhite: false })));
    expect(headers.getHeaders().White).toBe("Caissa (Master, Imitation 1)");
    expect(headers.getHeaders().Black).toBe("You");
  });

  it("gives the engine's side its rating and no one else's", () => {
    const board = new Board();
    board.loadPgn(recordPgn(record({ rating: 2300 })));
    expect(board.getHeaders().BlackElo).toBe("2300");
    expect(board.getHeaders().WhiteElo).toBeUndefined();
  });

  it("records a finished game's result and why it ended", () => {
    const pgn = recordPgn(record({
      moves: actions(["f3", "e5", "g4", "Qh4#"]),
      result: "0-1",
      termination: "checkmate",
    }));
    expect(pgn).toContain('[Result "0-1"]');
    expect(pgn).toContain("{Checkmate.}");
    expect(pgn.trim().endsWith("0-1")).toBe(true);
  });

  it("says who resigned", () => {
    const pgn = recordPgn(record({ result: "0-1", termination: "resignation" }));
    expect(pgn).toContain("{White resigns.}");
  });

  it("carries a set-up position in the FEN tag, so it can be replayed", () => {
    const start = "4k3/8/8/8/8/8/4P3/4K3 b - - 0 1";
    const board = new Board();
    board.loadPgn(recordPgn(record({ start, moves: actions(["Kd7", "e4"], start) })));
    expect(board.getHeaders().FEN).toBe(start);
    expect(board.history()).toEqual(["Kd7", "e4"]);
  });

  it("writes a collection as games a reader can split apart", () => {
    const text = collectionPgn([record({ id: "a" }), record({ id: "b", humanWhite: false })]);
    expect(text.match(/\[Event /g)).toHaveLength(2);
  });

  it("names files by date, side and opponent", () => {
    expect(pgnFileName(record())).toBe("caissa-2026-09-26-white-vs-master.pgn");
  });

  it("writes the moves the way a score sheet does", () => {
    expect(movesText(record())).toBe("1. e4 c5 2. Nf3 d6");
    const start = "4k3/8/8/8/8/8/4P3/4K3 b - - 0 1";
    expect(movesText(record({ start, moves: actions(["Kd7"], start) }))).toBe("1. ... Kd7");
  });
});
