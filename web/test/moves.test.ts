/**
 * The move list's layout, Undo's reach, and each game's names for its moves.
 *
 * Turns are where this project has been wrong before, so the cases that matter
 * are the ones that do not alternate: a Dots & Boxes chain, a Reversi pass, a
 * record that starts with the second player.
 */
import { describe, expect, it } from "vitest";
import { Chess, ChessState, actionOf } from "../src/games/chess";
import { Connect4 } from "../src/games/connect4";
import { DotsAndBoxes, HORIZONTAL_LINES } from "../src/games/dotsandboxes";
import { Gomoku, SIZE as G_SIZE } from "../src/games/gomoku";
import { DIRECTIONS, Isola, SQUARES as I_SQUARES } from "../src/games/isola";
import { PASS, Reversi } from "../src/games/reversi";
import { rowsOf, undoTarget, type Ply } from "../src/moves";
import { moveLabel } from "../src/ui/notation";

const ply = (index: number, seat: number, label = `m${index}`): Ply => ({ index, seat, label });
const labels = (cells: Ply[]) => cells.map((p) => p.label);

describe("laying out the move list", () => {
  it("pairs alternating moves into numbered rows", () => {
    const rows = rowsOf([ply(0, 0, "e4"), ply(1, 1, "e5"), ply(2, 0, "Nf3")]);
    expect(rows.map((r) => r.number)).toEqual([1, 2]);
    expect(labels(rows[0].cells[0])).toEqual(["e4"]);
    expect(labels(rows[0].cells[1])).toEqual(["e5"]);
    expect(labels(rows[1].cells[0])).toEqual(["Nf3"]);
    expect(rows[1].cells[1]).toEqual([]);
  });

  it("keeps a chain of moves by one side in one cell", () => {
    // Dots & Boxes: the first player closes two boxes and moves three times.
    const rows = rowsOf([ply(0, 0), ply(1, 1), ply(2, 0), ply(3, 0), ply(4, 0), ply(5, 1)]);
    expect(rows).toHaveLength(2);
    expect(labels(rows[1].cells[0])).toEqual(["m2", "m3", "m4"]);
    expect(labels(rows[1].cells[1])).toEqual(["m5"]);
  });

  it("starts with an empty first cell when the second side moves first", () => {
    const rows = rowsOf([ply(0, 1, "e5"), ply(1, 0, "Nf3")], 12);
    expect(rows[0].number).toBe(12);
    expect(rows[0].cells[0]).toEqual([]);
    expect(labels(rows[0].cells[1])).toEqual(["e5"]);
    expect(labels(rows[1].cells[0])).toEqual(["Nf3"]);
  });

  it("is empty before anyone moves", () => {
    expect(rowsOf([])).toEqual([]);
  });
});

describe("what Undo takes back", () => {
  it("your last move and the reply to it", () => {
    // you, engine, you, engine: back to before your second move.
    expect(undoTarget([0, 1, 0, 1], [10, 11, 12, 13], 0)).toBe(2);
  });

  it("just your move, when the engine has not answered yet", () => {
    expect(undoTarget([0, 1, 0], [10, 11, 12], 0)).toBe(2);
  });

  it("one move of a chain at a time, and it is still your turn", () => {
    // You closed boxes and moved three times in a row.
    expect(undoTarget([0, 1, 0, 0, 0], [1, 2, 3, 4, 5], 0)).toBe(4);
  });

  it("walks past forced passes to a move you chose", () => {
    // Your real move, the engine's reply, then a pass the page made for you.
    expect(undoTarget([0, 1, 0], [19, 26, PASS], 0, PASS)).toBe(0);
  });

  it("works from either side of the board", () => {
    expect(undoTarget([0, 1, 0, 1], [1, 2, 3, 4], 1)).toBe(3);
  });

  it("has nothing to do before you have moved", () => {
    expect(undoTarget([], [], 0)).toBeNull();
    expect(undoTarget([0], [3], 1)).toBeNull();
  });
});

describe("naming moves", () => {
  it("Four in a Row by column, counted from one", () => {
    const game = new Connect4();
    expect(moveLabel("connect4", game.initialState(), 0)).toBe("1");
    expect(moveLabel("connect4", game.initialState(), 6)).toBe("7");
  });

  it("Reversi in Othello notation, a1 top left, and passes by name", () => {
    const game = new Reversi();
    const start = game.initialState();
    expect(moveLabel("reversi", start, 0)).toBe("a1");
    expect(moveLabel("reversi", start, 19)).toBe("d3");
    expect(moveLabel("reversi", start, 63)).toBe("h8");
    expect(moveLabel("reversi", start, PASS)).toBe("pass");
  });

  it("Gomoku with rows counted from the bottom edge", () => {
    const game = new Gomoku();
    const centre = Math.floor(G_SIZE / 2) * G_SIZE + Math.floor(G_SIZE / 2);
    expect(moveLabel("gomoku", game.initialState(), centre)).toBe("e5");
    expect(moveLabel("gomoku", game.initialState(), 0)).toBe("a9");
  });

  it("Isolation by where the piece landed and which square went", () => {
    const game = new Isola();
    const state = game.initialState();
    const direction = DIRECTIONS.findIndex(([dr, dc]) => dr === 1 && dc === 0);
    // The first player's piece stands in the top row's middle; a step down lands
    // on d6, and the square destroyed here is a1, the bottom-left corner.
    const action = direction * I_SQUARES + 42;
    expect(game.legalActions(state)[action]).toBe(true);
    expect(moveLabel("isola", state, action)).toBe("d6 ×a1");
  });

  it("Dots and Boxes by the two dots a line joins", () => {
    const game = new DotsAndBoxes();
    expect(moveLabel("dotsandboxes", game.initialState(), 0)).toBe("a1-b1");
    expect(moveLabel("dotsandboxes", game.initialState(), HORIZONTAL_LINES)).toBe("a1-a2");
  });

  it("chess in standard algebraic notation, checks and castling included", () => {
    const game = new Chess();
    const start = game.initialState();
    const e4 = actionOf(start, { from: "e2", to: "e4", promotion: null });
    expect(moveLabel("chess", start, e4)).toBe("e4");

    const castle = ChessState.create("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1");
    expect(moveLabel("chess", castle, actionOf(castle, { from: "e1", to: "g1", promotion: null })))
      .toBe("O-O");

    const promote = ChessState.create("7k/P7/8/8/8/8/8/K7 w - - 0 1");
    expect(moveLabel("chess", promote, actionOf(promote, { from: "a7", to: "a8", promotion: "q" })))
      .toBe("a8=Q+");
  });
});
