/**
 * The opening book and the opening names.
 *
 * Both files are written in Python and read here, keyed by a position string
 * each side computes for itself - so the tests that matter most are the ones
 * that load the shipped files and check them against this side's chess rules:
 * every position the book names must be one these rules can set up, and every
 * move it offers must be legal there. A key computed differently on one side
 * would make the book silently empty; a move from the wrong position would
 * throw in the middle of a game.
 */
import { readFileSync } from "node:fs";
import { Chess as Board } from "chess.js";
import { describe, expect, it } from "vitest";
import { Chess, ChessState, actionOf } from "../src/games/chess";
import { bookMove, openingOf, parseBook, parseNames, positionKey, type Book } from "../src/openings";

const load = (name: string) =>
  JSON.parse(readFileSync(new URL(`../public/${name}`, import.meta.url), "utf8"));
const book = parseBook(load("book.json"));
const names = parseNames(load("openings.json"));
const rules = new Chess();

/** Every position along a line of moves in SAN, the start included. */
function line(sans: string[]): ChessState[] {
  const board = new Board();
  let state = rules.initialState();
  const states = [state];
  for (const san of sans) {
    const move = board.move(san);
    state = rules.apply(state, actionOf(state, { from: move.from, to: move.to, promotion: move.promotion ?? null }));
    states.push(state);
  }
  return states;
}

describe("the position key", () => {
  it("is the FEN's first four fields, which is also the repetition key", () => {
    expect(positionKey(rules.initialState())).toBe("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq -");
  });

  it("leaves out an en passant square no pawn can use, as the Python side does", () => {
    // After 1. e4 no black pawn can take on e3, so the square is not written.
    expect(positionKey(line(["e4"])[1])).toBe("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq -");
  });
});

describe("sampling from the book", () => {
  const tiny: Book = new Map([[positionKey(rules.initialState()), [["e2e4", 60], ["d2d4", 30], ["g1f3", 10]]]]);

  it("chooses in proportion to how often strong players did", () => {
    const pick = (r: number) => bookMove(rules.initialState(), tiny, () => r)!.uci;
    expect(pick(0)).toBe("e2e4");
    expect(pick(0.599)).toBe("e2e4");
    expect(pick(0.6)).toBe("d2d4");
    expect(pick(0.899)).toBe("d2d4");
    expect(pick(0.9)).toBe("g1f3");
    expect(pick(0.9999)).toBe("g1f3");
  });

  it("says how common the choice was", () => {
    const choice = bookMove(rules.initialState(), tiny, () => 0.95)!;
    expect(choice.share).toBeCloseTo(0.1);
    expect(choice.games).toBe(100);
  });

  it("has nothing to say out of book", () => {
    expect(bookMove(line(["a3"])[1], tiny)).toBeNull();
  });
});

describe("the shipped book", () => {
  it("starts where every game starts, with the moves strong players open with", () => {
    const start = book.get(positionKey(rules.initialState()))!;
    expect(start.map(([uci]) => uci)).toEqual(expect.arrayContaining(["e2e4", "d2d4"]));
  });

  it("offers more than one reply to 1. e4: the whole point", () => {
    const replies = book.get(positionKey(line(["e4"])[1]))!;
    expect(replies.length).toBeGreaterThanOrEqual(4);
    expect(replies.map(([uci]) => uci)).toEqual(expect.arrayContaining(["c7c5", "e7e5", "e7e6"]));
  });

  it("names only positions these rules can set up, and only moves legal in them", () => {
    let moves = 0;
    for (const [key, entries] of book) {
      const state = ChessState.create(`${key} 0 1`);
      expect(positionKey(state), key).toBe(key);
      const legal = new Set(state.moves().map((m) => `${m.from}${m.to}${m.promotion ?? ""}`));
      for (const [uci] of entries) {
        expect(legal.has(uci), `${uci} in ${key}`).toBe(true);
        moves += 1;
      }
    }
    expect(moves).toBeGreaterThan(5000);
  });
});

describe("naming openings", () => {
  it("names a line by the deepest named position it has reached", () => {
    const najdorf = line(["e4", "c5", "Nf3", "d6", "d4", "cxd4", "Nxd4", "Nf6", "Nc3", "a6"]);
    expect(openingOf(najdorf, names)).toEqual(["B90", "Sicilian Defense: Najdorf Variation"]);
  });

  it("names a position however it was reached", () => {
    // The Queen's Gambit Declined, through the front door and through the English.
    const direct = openingOf(line(["d4", "d5", "c4", "e6"]), names);
    const english = openingOf(line(["c4", "e6", "d4", "d5"]), names);
    expect(direct).toEqual(english);
    expect(direct?.[1]).toBe("Queen's Gambit Declined");
  });

  it("keeps the last name a game had once it leaves the named paths", () => {
    const wandering = line(["e4", "e6", "a3", "a6", "h3"]);
    expect(openingOf(wandering, names)?.[1]).toMatch(/^French Defense/);
  });

  it("has no name before the first named position", () => {
    expect(openingOf([rules.initialState()], names)).toBeNull();
  });

  it("only names positions these rules can set up", () => {
    for (const key of names.keys()) {
      expect(positionKey(ChessState.create(`${key} 0 1`))).toBe(key);
    }
  });
});
