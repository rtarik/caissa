/**
 * The Isola board's clicks: a step, then a demolition.
 *
 * The square a piece has just left is the one players reach for first - it is
 * the obvious square to remove, and it is free the moment the piece steps off
 * it. The board offers it, and this checks that the offer is real: that the
 * action the click produces is one the rules accept.
 */
import { describe, expect, it } from "vitest";
import { Isola, SQUARES } from "../src/games/isola";
import { isolaView } from "../src/ui/views";
import type { ViewContext } from "../src/ui/views";

const game = new Isola();

const square = (index: number) =>
  ({ dataset: { square: String(index) } }) as unknown as HTMLElement;

const context = (state: ReturnType<Isola["initialState"]>, pending: number | null,
                 humanFirst = true): ViewContext => ({
  game: game as never,
  state: state as never,
  moves: [],
  humanFirst,
  locked: false,
  pending,
});

describe("the Isola board's clicks", () => {
  it("destroys the square the piece just came from", () => {
    const state = game.initialState();
    const from = state.mover;
    const to = game.steps(state)[0];

    expect(isolaView.select(square(to), context(state, null))).toEqual({
      kind: "pending",
      pending: to,
    });

    const selection = isolaView.select(square(from), context(state, to));
    expect(selection).not.toBeNull();
    expect(selection!.kind).toBe("action");

    const action = (selection as { kind: "action"; action: number }).action;
    expect(game.legalActions(state)[action]).toBe(true);
    // And it really is that square that goes: the rules read the action the
    // same way the click meant it.
    expect(action % SQUARES).toBe(from);
    expect(game.apply(state, action).usable[from]).toBe(0);
  });

  it("offers that square on the board, enabled", () => {
    const state = game.initialState();
    const to = game.steps(state)[0];
    const html = isolaView.board(context(state, to));

    const cell = html
      .split("<button")
      .find((chunk) => chunk.includes(`data-square="${state.mover}"`))!;
    expect(cell).toContain("doomed");
    expect(cell).not.toContain("disabled");
  });

  it("takes the step back when the landing square is clicked again", () => {
    const state = game.initialState();
    const to = game.steps(state)[0];
    expect(isolaView.select(square(to), context(state, to))).toEqual({
      kind: "pending",
      pending: null,
    });
  });
});
