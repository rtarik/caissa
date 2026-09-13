/**
 * Both games' TypeScript rules must agree with their Python counterparts.
 *
 * Parameterised over every game deliberately: the suite is the browser-side
 * equivalent of Phase 5's claim that the abstraction is real. Adding a game
 * should mean adding a row here, not writing a new test file.
 */
import { describe, expect, it } from "vitest";
import { Connect4 } from "../src/games/connect4";
import { Gomoku } from "../src/games/gomoku";
import { Isola } from "../src/games/isola";
import { Reversi, PASS, SQUARES } from "../src/games/reversi";
import type { Game } from "../src/games/types";
import connect4Vectors from "./connect4-vectors.json";
import gomokuVectors from "./gomoku-vectors.json";
import isolaVectors from "./isola-vectors.json";
import reversiVectors from "./reversi-vectors.json";

interface Vectors {
  game: string;
  boardShape: number[];
  inputPlanes: number;
  actionSize: number;
  cases: { moves: number[]; legal: number[]; terminal: number | null; encoded: number[] }[];
}

const SUBJECTS: { game: Game<unknown>; vectors: Vectors }[] = [
  { game: new Connect4() as Game<unknown>, vectors: connect4Vectors as Vectors },
  { game: new Reversi() as Game<unknown>, vectors: reversiVectors as Vectors },
  { game: new Gomoku() as Game<unknown>, vectors: gomokuVectors as Vectors },
  { game: new Isola() as Game<unknown>, vectors: isolaVectors as Vectors },
];

for (const { game, vectors } of SUBJECTS) {
  const replay = (moves: number[]) => {
    let state = game.initialState();
    for (const move of moves) state = game.apply(state, move);
    return state;
  };

  describe(`${game.name} agrees with Python`, () => {
    it("is checking the right game", () => {
      expect(game.name).toBe(vectors.game);
      expect([...game.boardShape]).toEqual(vectors.boardShape);
      expect(game.actionSize).toBe(vectors.actionSize);
      expect(game.inputPlanes).toBe(vectors.inputPlanes);
      expect(vectors.cases.length).toBeGreaterThan(100);
    });

    it("agrees on legal moves", () => {
      for (const testCase of vectors.cases) {
        expect(
          game.legalActions(replay(testCase.moves)).map(Number),
          `moves ${testCase.moves}`,
        ).toEqual(testCase.legal);
      }
    });

    it("agrees on terminal values", () => {
      for (const testCase of vectors.cases) {
        expect(game.terminalValue(replay(testCase.moves)), `moves ${testCase.moves}`)
          .toBe(testCase.terminal);
      }
    });

    it("agrees on encodings, which is what the network consumes", () => {
      for (const testCase of vectors.cases) {
        expect(
          Array.from(game.encode(replay(testCase.moves))),
          `moves ${testCase.moves}`,
        ).toEqual(testCase.encoded);
      }
    });

    it("only ever reports an outcome of -1, 0 or +1", () => {
      // Not "never +1". That holds for Connect 4, which ends the instant someone
      // wins and therefore always ends on the loser's turn - but not for Reversi,
      // which ends when neither side can move and where the player to move may be
      // the one ahead. Asserting the Connect 4 behaviour for both is what this
      // test did first, and Reversi rejected it.
      for (const testCase of vectors.cases) {
        const value = game.terminalValue(replay(testCase.moves));
        expect(value === null || [-1, 0, 1].includes(value)).toBe(true);
      }
    });

    it("always offers at least one action while the game continues", () => {
      for (const testCase of vectors.cases) {
        const state = replay(testCase.moves);
        if (game.terminalValue(state) !== null) continue;
        expect(game.legalActions(state).some(Boolean)).toBe(true);
      }
    });
  });
}

describe("reversi's own complications", () => {
  const game = new Reversi();

  it("starts from the standard opening", () => {
    const state = game.initialState();
    expect(state.board[3 * 8 + 4]).toBe(1);
    expect(state.board[3 * 8 + 3]).toBe(-1);
    expect(game.legalActions(state).filter(Boolean).length).toBe(4);
  });

  it("offers a pass only when nothing else is available", () => {
    // A forced pass exists among the generated positions; find it.
    const forced = (reversiVectors as Vectors).cases.find((c) => {
      let state = game.initialState();
      for (const move of c.moves) state = game.apply(state, move);
      return game.legalActions(state)[PASS];
    });
    expect(forced, "no forced-pass position in the vectors").toBeDefined();

    let state = game.initialState();
    for (const move of forced!.moves) state = game.apply(state, move);
    expect(game.legalActions(state).slice(0, SQUARES).some(Boolean)).toBe(false);
    // And having no move does not end the game.
    expect(game.terminalValue(state)).toBeNull();
  });

  it("refuses a pass while a move exists", () => {
    expect(() => game.apply(game.initialState(), PASS)).toThrow();
  });
});
