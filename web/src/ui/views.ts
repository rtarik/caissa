import { COLS as C4_COLS, ROWS as C4_ROWS } from "../games/connect4";
import { Gomoku, SIZE as G_SIZE, SQUARES as G_SQUARES } from "../games/gomoku";
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
  /**
   * How search's visit counts should be drawn.
   *
   * A bar per action only reads when there are few of them. Connect 4 has seven,
   * so bars work. Gomoku has eighty-one and Reversi sixty-five, and a row of
   * eighty-one slivers says nothing - but those actions *are* board squares, so
   * the same numbers laid out as a heatmap show exactly where the search looked.
   */
  visits(counts: number[]): string;
}

/** A bar per action, for small action spaces. */
function bars(counts: number[], columns: number): string {
  const best = Math.max(...counts);
  const cells = counts
    .map((v) => `<div class="visit ${v === best && v > 0 ? "best" : ""}"
       style="height:${best > 0 ? Math.max(2, (v / best) * 100) : 2}%"
       title="${v} visits"></div>`)
    .join("");
  return `<div class="visits" style="grid-template-columns: repeat(${columns}, 1fr)">${cells}</div>`;
}

/** The same numbers laid over the board, for action spaces that are squares. */
function heatmap(counts: number[], squares: number, columns: number): string {
  const best = Math.max(...counts.slice(0, squares));
  const cells = counts
    .slice(0, squares)
    .map((v) => `<div class="heat" style="opacity:${
      best > 0 ? Math.max(0.04, v / best) : 0.04
    }" title="${v} visits"></div>`)
    .join("");
  return `<div class="heatmap" style="grid-template-columns: repeat(${columns}, 1fr)">${cells}</div>`;
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
  visits: (counts) => bars(counts, C4_COLS),
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
      const classes = ["cell", i === last ? "last" : "",
                       !ctx.locked && legal[i] ? "playable hint" : ""];
      const row = Math.floor(i / R_SIZE);
      // The disc is a child rather than the cell's own background, so it can be
      // drawn smaller than its square and neighbouring discs never touch.
      const disc = owner ? `<span class="stone ${owner}"></span>` : "";
      return `<button class="${classes.filter(Boolean).join(" ")}" data-action="${i}"
        ${ctx.locked || !legal[i] ? "disabled" : ""}
        aria-label="Row ${row + 1}, column ${(i % R_SIZE) + 1}">${disc}</button>`;
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
    return `<span class="spacer score">
              <span class="dot you"></span>${yours}
              <span class="dot engine" style="margin-left:8px"></span>${theirs}
            </span>`;
  },

  // The 65th action is the pass, which is not a square; the heatmap shows the
  // board and leaves it out rather than pretending it has a location.
  visits: (counts) => heatmap(counts, R_SQUARES, R_SIZE),
};

export const gomokuView: View = {
  layout: "grid-9 intersections",

  board(ctx) {
    const grid: Grid = absoluteGrid(ctx.game, ctx.state, ctx.moves.length);
    const legal = ctx.game.legalActions(ctx.state);
    const last = ctx.moves.length ? ctx.moves[ctx.moves.length - 1] : null;
    // Highlighting the five is worth the detour: on an open board a finished
    // line is genuinely hard to spot among thirty other stones.
    const line = new Set(
      (ctx.game as unknown as Gomoku).winningLine?.(ctx.state as never) ?? [],
    );

    return Array.from({ length: G_SQUARES }, (_, i) => {
      const row = Math.floor(i / G_SIZE);
      const col = i % G_SIZE;
      // Edge classes trim the grid lines so they stop at the outer
      // intersections, as they do on a real board, rather than running on into
      // the margin.
      const classes = ["cell", i === last ? "last" : "",
                       !ctx.locked && legal[i] ? "playable" : "",
                       row === 0 ? "top" : "", row === G_SIZE - 1 ? "bottom" : "",
                       col === 0 ? "left" : "", col === G_SIZE - 1 ? "right" : ""];
      const owner = who(grid[i], ctx.humanFirst);
      const stone = owner
        ? `<span class="stone ${owner}${line.has(i) ? " win" : ""}"></span>`
        : "";
      return `<button class="${classes.filter(Boolean).join(" ")}" data-action="${i}"
        ${ctx.locked || !legal[i] ? "disabled" : ""}
        aria-label="Row ${row + 1}, column ${col + 1}">${stone}</button>`;
    }).join("");
  },

  actionFor(element) {
    const value = element.dataset.action;
    return value === undefined ? null : Number(value);
  },

  detail: () => "",
  visits: (counts) => heatmap(counts, G_SQUARES, G_SIZE),
};

export const VIEWS: Record<string, View> = {
  connect4: connect4View,
  reversi: reversiView,
  gomoku: gomokuView,
};
