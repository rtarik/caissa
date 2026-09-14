/**
 * Both games' TypeScript rules must agree with their Python counterparts.
 *
 * Parameterised over every game deliberately: the suite is the browser-side
 * equivalent of Phase 5's claim that the abstraction is real. Adding a game
 * should mean adding a row here, not writing a new test file.
 */
import { describe, expect, it } from "vitest";
import { Connect4 } from "../src/games/connect4";
import { DotsAndBoxes, horizontal, vertical } from "../src/games/dotsandboxes";
import { Gomoku } from "../src/games/gomoku";
import { Isola } from "../src/games/isola";
import { Reversi, PASS, SQUARES } from "../src/games/reversi";
import type { Game } from "../src/games/types";
import connect4Vectors from "./connect4-vectors.json";
import dotsVectors from "./dotsandboxes-vectors.json";
import gomokuVectors from "./gomoku-vectors.json";
import isolaVectors from "./isola-vectors.json";
import reversiVectors from "./reversi-vectors.json";

interface Vectors {
  game: string;
  boardShape: number[];
  inputPlanes: number;
  actionSize: number;
  cases: {
    moves: number[];
    legal: number[];
    toPlay: number;
    terminal: number | null;
    encoded: number[];
  }[];
}

const SUBJECTS: { game: Game<unknown>; vectors: Vectors }[] = [
  { game: new Connect4() as Game<unknown>, vectors: connect4Vectors as Vectors },
  { game: new Reversi() as Game<unknown>, vectors: reversiVectors as Vectors },
  { game: new Gomoku() as Game<unknown>, vectors: gomokuVectors as Vectors },
  { game: new Isola() as Game<unknown>, vectors: isolaVectors as Vectors },
  { game: new DotsAndBoxes() as Game<unknown>, vectors: dotsVectors as Vectors },
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

    it("agrees on whose turn it is", () => {
      // Stated by the game rather than counted: in Dots & Boxes the two differ
      // after every closed box, and a UI that counted would give the bonus move
      // to the wrong player.
      for (const testCase of vectors.cases) {
        expect(game.toPlay(replay(testCase.moves)), `moves ${testCase.moves}`)
          .toBe(testCase.toPlay);
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

describe("dots and boxes' own complications", () => {
  const game = new DotsAndBoxes();

  it("keeps the turn when a line closes a box", () => {
    let state = game.initialState();
    for (const line of [horizontal(0, 0), horizontal(1, 0), vertical(0, 0)]) {
      state = game.apply(state, line);
    }
    expect(game.toPlay(state)).toBe(1);

    const closed = game.apply(state, vertical(0, 1));
    expect(game.toPlay(closed)).toBe(1);
    expect(game.score(closed)).toEqual([1, 0]);
  });

  it("exercises bonus moves in the vectors", () => {
    const cases = (dotsVectors as Vectors).cases;
    const bonuses = cases.filter(
      (c, i) => i > 0 && c.moves.length === cases[i - 1].moves.length + 1
        && c.toPlay === cases[i - 1].toPlay,
    );
    expect(bonuses.length).toBeGreaterThan(20);
  });
});

describe("pass actions", () => {
  it("are declared only by the game that has one", () => {
    // The page used to infer the pass from the last action index, which in Four
    // in a Row is simply the seventh column.
    expect(new Reversi().passAction).toBe(PASS);
    for (const game of [new Connect4(), new Gomoku(), new Isola(), new DotsAndBoxes()]) {
      expect((game as Game<unknown>).passAction).toBeUndefined();
    }
  });
});
