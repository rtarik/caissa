import { COLS as C4_COLS, ROWS as C4_ROWS } from "../games/connect4";
import { PASS, SIZE as R_SIZE, SQUARES as R_SQUARES } from "../games/reversi";
import type { Game } from "../games/types";
import { absoluteGrid, connect4LastMove, winningLine, type Grid } from "./board";

export interface ViewContext<S = unknown> {
  game: Game<S>;
  state: S;
  moves: number[];
  humanFirst: boolean;
  locked: boolean;
}

export interface View {
  /** CSS class controlling the board's grid shape. */
  readonly layout: string;
  board(ctx: ViewContext): string;
  /** The action a clicked element stands for, or null if it is not playable. */
  actionFor(element: HTMLElement): number | null;
  /** Extra line under the board, such as a running score. */
  detail(ctx: ViewContext): string;
}

const who = (owner: number, humanFirst: boolean) =>
  owner === 0 ? "" : (owner === 1) === humanFirst ? "you" : "engine";

export const connect4View: View = {
  layout: "grid-7",

  board(ctx) {
    const grid: Grid = absoluteGrid(ctx.game, ctx.state, ctx.moves.length);
    const last = connect4LastMove(ctx.moves, C4_COLS, C4_ROWS);
    const line = new Set(winningLine(grid, C4_COLS, C4_ROWS, last) ?? []);
    const legal = ctx.game.legalActions(ctx.state);

    return Array.from({ length: C4_ROWS * C4_COLS }, (_, i) => {
      const col = i % C4_COLS;
      const classes = ["cell", who(grid[i], ctx.humanFirst), line.has(i) ? "win" : "",
                       i === last ? "last" : "",
                       !ctx.locked && legal[col] ? "playable" : ""];
      return `<button class="${classes.filter(Boolean).join(" ")}" data-action="${col}"
        ${ctx.locked || !legal[col] ? "disabled" : ""} aria-label="Column ${col + 1}"></button>`;
    }).join("");
  },

  actionFor(element) {
    const value = element.dataset.action;
    return value === undefined ? null : Number(value);
  },

  detail: () => "",
};

export const reversiView: View = {
  layout: "grid-8 squares",

  board(ctx) {
    const grid: Grid = absoluteGrid(ctx.game, ctx.state, ctx.moves.length);
    const legal = ctx.game.legalActions(ctx.state);
    const last = ctx.moves.length && ctx.moves[ctx.moves.length - 1] !== PASS
      ? ctx.moves[ctx.moves.length - 1]
      : null;

    return Array.from({ length: R_SQUARES }, (_, i) => {
      const owner = who(grid[i], ctx.humanFirst);
      // Legal squares are hinted, which is standard in Reversi interfaces and
      // removes a lot of squinting without giving anything away.
      const classes = ["cell", owner, i === last ? "last" : "",
                       !ctx.locked && legal[i] ? "playable hint" : ""];
      const row = Math.floor(i / R_SIZE);
      return `<button class="${classes.filter(Boolean).join(" ")}" data-action="${i}"
        ${ctx.locked || !legal[i] ? "disabled" : ""}
        aria-label="Row ${row + 1}, column ${(i % R_SIZE) + 1}"></button>`;
    }).join("");
  },

  actionFor(element) {
    const value = element.dataset.action;
    return value === undefined ? null : Number(value);
  },

  detail(ctx) {
    const grid = absoluteGrid(ctx.game, ctx.state, ctx.moves.length);
    let first = 0;
    let second = 0;
    for (const owner of grid) {
      if (owner === 1) first++;
      else if (owner === 2) second++;
    }
    const [yours, theirs] = ctx.humanFirst ? [first, second] : [second, first];
    return `<span class="dot you"></span> ${yours}
            <span class="dot engine" style="margin-left:10px"></span> ${theirs}`;
  },
};

export const VIEWS: Record<string, View> = {
  connect4: connect4View,
  reversi: reversiView,
};
