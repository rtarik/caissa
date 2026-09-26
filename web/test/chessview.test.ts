/**
 * The chess board's clicks: what each one means.
 *
 * A move is two clicks and a promotion three, because four different actions
 * share a promotion's two squares. Reaching a promotion by playing through the
 * page takes dozens of moves, so the view is checked directly here instead.
 */
import { describe, expect, it } from "vitest";
import { Chess, ChessState, actionOf } from "../src/games/chess";
import { chessView } from "../src/ui/chessview";
import type { ViewContext } from "../src/ui/views";

const game = new Chess();

/** A stand-in for the clicked element: all the view reads is its data attributes. */
const square = (index: number) => ({ dataset: { square: String(index) } }) as unknown as HTMLElement;
const choice = (action: number) => ({ dataset: { action: String(action) } }) as unknown as HTMLElement;

const context = (state: ChessState, pending: number | null): ViewContext => ({
  game: game as never,
  state,
  moves: [],
  humanFirst: true,
  locked: false,
  pending,
});

describe("the chess board's clicks", () => {
  it("moves in two clicks, switches pieces, and lets go of an unreachable square", () => {
    const state = game.initialState();
    const e2 = 12;
    expect(chessView.select(square(e2), context(state, null))).toEqual({ kind: "pending", pending: e2 });
    expect(chessView.select(square(28), context(state, e2))).toEqual({
      kind: "action",
      action: actionOf(state, { from: "e2", to: "e4", promotion: null }),
    });
    expect(chessView.select(square(6), context(state, e2))).toEqual({ kind: "pending", pending: 6 });
    expect(chessView.select(square(36), context(state, e2))).toEqual({ kind: "pending", pending: null });
    // An empty square or an opponent's piece starts nothing.
    expect(chessView.select(square(36), context(state, null))).toBeNull();
    expect(chessView.select(square(52), context(state, null))).toBeNull();
  });

  it("takes a promotion in three clicks: the pawn, the square, the piece", () => {
    const state = ChessState.create("4k3/1P6/8/8/8/8/8/4K3 w - - 0 1");
    const [b7, b8] = [49, 57];
    expect(chessView.select(square(b7), context(state, null))).toEqual({ kind: "pending", pending: b7 });

    const asked = chessView.select(square(b8), context(state, b7));
    expect(asked?.kind).toBe("pending");
    const pending = (asked as { pending: number }).pending;

    const offered = [...chessView.board(context(state, pending)).matchAll(/data-action="(\d+)"/g)]
      .map((match) => Number(match[1]));
    const expected = ["q", "r", "b", "n"].map((piece) =>
      actionOf(state, { from: "b7", to: "b8", promotion: piece }));
    expect(offered).toEqual(expected);
    expect(new Set(offered).size).toBe(4);

    const knight = expected[3];
    expect(chessView.select(choice(knight), context(state, pending)))
      .toEqual({ kind: "action", action: knight });
    expect(game.apply(state, knight).board().get("b8")).toMatchObject({ type: "n", color: "w" });

    // Clicking the board instead of a piece cancels the promotion.
    expect(chessView.select(square(b7), context(state, pending))).toEqual({ kind: "pending", pending: null });
  });
});

describe("a game that began from a set-up position", () => {
  /** The markup of one square of a rendered board. */
  const cell = (html: string, square: number) =>
    html.split("<button").find((chunk) => chunk.includes(`data-square="${square}"`)) ?? "";

  it("marks the last move, played from the set-up position rather than the usual one", () => {
    // Black to move first. The view used to replay the game from the usual
    // starting position, where this action decodes as a white king's move from
    // e1 - which is not legal there, so rendering threw and the page froze.
    const start = ChessState.create("4k3/8/8/8/8/8/4R3/4K3 b - - 0 1");
    const action = actionOf(start, { from: "e8", to: "d7", promotion: null });
    const after = game.apply(start, action);

    const html = chessView.board({
      game: game as never,
      state: after,
      previous: start,
      moves: [action],
      humanFirst: true,
      locked: true,
      pending: null,
    });
    expect(cell(html, 60)).toContain("last"); // e8
    expect(cell(html, 51)).toContain("last"); // d7
    expect(cell(html, 4)).not.toContain("last"); // e1, where the old replay put it
  });

  it("marks nothing before the first move", () => {
    const start = ChessState.create("4k3/8/8/8/8/8/4R3/4K3 b - - 0 1");
    const html = chessView.board({
      game: game as never, state: start, previous: null, moves: [], humanFirst: true, locked: true, pending: null,
    });
    expect(html).not.toContain(" last");
  });
});
