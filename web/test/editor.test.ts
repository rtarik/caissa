/**
 * The board editor's rules: FEN in and out, rights that follow the pieces, and
 * what stops a setup being played.
 *
 * chess.js turns away a missing king, a doubled king and pawns on the edge
 * ranks. What it accepts, the editor has to catch: a side in check that is not
 * to move, castling with no rook, too many pawns or pieces, and a game that is
 * over before it starts. Each message is tested against a position that should
 * produce it, and a legal position is tested to produce none.
 */
import { describe, expect, it } from "vitest";
import {
  STANDARD_FEN, emptySetup, fenOf, parseFen, possibleCastling, problems,
  withCastling, withPiece, withSideToMove,
} from "../src/editor";
import { Chess } from "../src/games/chess";

const setup = (fen: string) => {
  const parsed = parseFen(fen);
  if (!parsed) throw new Error(`test FEN did not parse: ${fen}`);
  return parsed;
};

describe("reading and writing FEN", () => {
  it("round-trips the starting position and a middlegame", () => {
    for (const fen of [
      STANDARD_FEN,
      "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",
      "4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 12",
    ]) {
      expect(fenOf(setup(fen))).toBe(fen);
    }
  });

  it("puts a1 at index 0 and h8 at 63, as the rest of the page numbers them", () => {
    const start = setup(STANDARD_FEN);
    expect(start.squares[0]).toBe("R"); // a1
    expect(start.squares[4]).toBe("K"); // e1
    expect(start.squares[60]).toBe("k"); // e8
    expect(start.squares[63]).toBe("r"); // h8
  });

  it("accepts a FEN with only the board and the side to move", () => {
    expect(fenOf(setup("4k3/8/8/8/8/8/8/4K3 b"))).toBe("4k3/8/8/8/8/8/8/4K3 b - - 0 1");
  });

  it("refuses what is not a FEN", () => {
    for (const bad of ["", "hello", "8/8/8/8/8/8/8 w", "9/8/8/8/8/8/8/8 w", "4k3/8/8/8/8/8/8/4K3 x",
      "4k3/8/8/8/8/8/8/4KX2 w"]) {
      expect(parseFen(bad), bad).toBeNull();
    }
  });
});

describe("editing", () => {
  it("places and removes pieces", () => {
    let board = emptySetup();
    board = withPiece(board, 4, "K");
    expect(board.squares[4]).toBe("K");
    board = withPiece(board, 4, null);
    expect(board.squares[4]).toBeNull();
  });

  it("only allows castling with the king and that rook at home", () => {
    const start = setup(STANDARD_FEN);
    expect(possibleCastling(start.squares)).toEqual({ K: true, Q: true, k: true, q: true });
    // Take the h1 rook away: White may no longer castle king side.
    const noRook = withPiece(start, 7, null);
    expect(noRook.castling.K).toBe(false);
    expect(noRook.castling.Q).toBe(true);
    // And it cannot be switched back on without the rook.
    expect(withCastling(noRook, "K", true).castling.K).toBe(false);
  });

  it("forgets a pasted en passant square as soon as anything changes", () => {
    const passant = setup("4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 12");
    expect(withPiece(passant, 0, "R").enPassant).toBe("-");
    expect(withSideToMove(passant, false).enPassant).toBe("-");
  });
});

describe("what stops a setup being played", () => {
  it("names both missing kings on an empty board", () => {
    expect(problems(emptySetup())).toEqual(["White needs a king.", "Black needs a king."]);
  });

  it("refuses a second king", () => {
    expect(problems(setup("4k3/8/8/8/8/8/8/3KK3 w - - 0 1"))).toContain("White can only have one king.");
  });

  it("refuses pawns on the edge ranks", () => {
    expect(problems(setup("4k3/8/8/8/8/8/8/P3K3 w - - 0 1")))
      .toContain("Pawns can't stand on the first or last rank.");
  });

  it("refuses more than eight pawns", () => {
    expect(problems(setup("4k3/8/8/8/8/P7/PPPPPPPP/4K3 w - - 0 1")))
      .toContain("White can't have more than eight pawns.");
  });

  it("refuses a side in check that is not to move", () => {
    // White to move with Black in check: White could simply take the king.
    expect(problems(setup("4k3/8/8/8/8/8/4R3/4K3 w - - 0 1")))
      .toEqual(["Black is in check, so it has to be Black to move."]);
    expect(problems(setup("4k3/8/8/8/8/8/4R3/4K3 b - - 0 1"))).toEqual([]);
  });

  it("refuses a game that is already over", () => {
    expect(problems(setup("7k/6Q1/6K1/8/8/8/8/8 b - - 0 1")))
      .toEqual(["That's checkmate already, so there's nothing to play."]);
    expect(problems(setup("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1")))
      .toEqual(["That's stalemate already, so there's nothing to play."]);
    expect(problems(setup("4k3/8/8/8/8/8/8/4K3 w - - 0 1")))
      .toEqual(["Neither side has enough pieces left to win."]);
  });

  it("has nothing to say about a legal position, and the rules can start from it", () => {
    for (const fen of [STANDARD_FEN, "4k3/8/8/8/8/8/4R3/4K3 b - - 0 1", "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1"]) {
      expect(problems(setup(fen)), fen).toEqual([]);
      // What the editor approves, the engine's rules must accept.
      const rules = new Chess();
      const state = rules.positionFrom(fenOf(setup(fen)));
      expect(rules.terminalValue(state)).toBeNull();
      // And it must be *this* position, not a fresh game: the engine replays the
      // moves from wherever this puts it.
      expect(state.fen).toBe(fen);
    }
  });
});
