import { COLS as C4_COLS, ROWS as C4_ROWS } from "../games/connect4";
import {
  HORIZONTAL_LINES as D_HORIZONTAL,
  LATTICE as D_LATTICE,
  LINES as D_LINES,
  SQUARES as D_SQUARES,
  boxCell,
  lineCell,
  type DotsAndBoxesState,
} from "../games/dotsandboxes";
import { Gomoku, SIZE as G_SIZE, SQUARES as G_SQUARES } from "../games/gomoku";
import { Isola, SIZE as I_SIZE, SQUARES as I_SQUARES } from "../games/isola";
import { PASS, SIZE as R_SIZE, SQUARES as R_SQUARES } from "../games/reversi";
import type { Game } from "../games/types";
import { absoluteGrid, connect4LastMove, winningLine, type Grid } from "./board";
import { chessView } from "./chessview";
import { heatmap } from "./heatmap";

export interface ViewContext<S = unknown> {
  game: Game<S>;
  state: S;
  moves: number[];
  humanFirst: boolean;
  locked: boolean;
  /**
   * A half-finished move, for games whose turn is more than one decision.
   *
   * Isola asks for a step and then a demolition. Rather than let the page
   * invent a second kind of turn, the view reports what a click *meant* and the
   * page holds the fragment until it becomes a whole action.
   */
  pending: number | null;
}

/** What a click on the board amounted to. */
export type Selection =
  | { kind: "action"; action: number }
  | { kind: "pending"; pending: number | null };

export interface View {
  /** CSS class controlling the board's grid shape. */
  readonly layout: string;
  board(ctx: ViewContext): string;
  /** What clicking this element means: a whole action, or part of one. */
  select(element: HTMLElement, ctx: ViewContext): Selection | null;
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
  visits(counts: number[], ctx: ViewContext): string;
  /** An optional side panel, such as chess's move list. */
  notes?(ctx: ViewContext): string;
  /** What to ask for while a move is half made, for games whose moves take more than one click. */
  prompt?(ctx: ViewContext): string;
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

/** For games where one click is one turn. */
const wholeAction = (element: HTMLElement): Selection | null => {
  const value = element.dataset.action;
  return value === undefined ? null : { kind: "action", action: Number(value) };
};

/** Whether the Isola board should be drawn upside down to face the human. */
function isolaFlipped(ctx: ViewContext): boolean {
  return ctx.humanFirst;
}

const who = (owner: number, humanFirst: boolean) =>
  owner === 0 ? "" : (owner === 1) === humanFirst ? "you" : "engine";

export const connect4View: View = {
  layout: "grid-7",

  board(ctx) {
    const grid: Grid = absoluteGrid(ctx.game, ctx.state);
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

  select: wholeAction,
  detail: () => "",
  visits: (counts) => bars(counts, C4_COLS),
};

export const reversiView: View = {
  layout: "grid-8 squares",

  board(ctx) {
    const grid: Grid = absoluteGrid(ctx.game, ctx.state);
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

  select: wholeAction,

  detail(ctx) {
    const grid = absoluteGrid(ctx.game, ctx.state);
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
    const grid: Grid = absoluteGrid(ctx.game, ctx.state);
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

  select: wholeAction,
  detail: () => "",
  visits: (counts) => heatmap(counts, G_SQUARES, G_SIZE),
};

export const isolaView: View = {
  layout: "grid-7 tiles",
  prompt: () => "Now choose a square to destroy.",

  board(ctx) {
    const game = ctx.game as unknown as Isola;
    const state = ctx.state as never as ReturnType<Isola["initialState"]>;
    const yourTurn = ctx.game.toPlay(ctx.state) === (ctx.humanFirst ? 0 : 1);

    // Isola is the first game here with sides, so it is the first that can be
    // the wrong way up. Flip the rows when the human starts on the far edge, so
    // their piece is always the near one - the convention every board game with
    // a facing uses.
    const flipped = isolaFlipped(ctx);
    const toBoard = (display: number) =>
      flipped ? (I_SIZE - 1 - Math.floor(display / I_SIZE)) * I_SIZE + (display % I_SIZE)
              : display;

    // Two phases. Before a step is chosen the reachable squares are offered;
    // afterwards, the squares that may be destroyed. Showing both at once would
    // be ambiguous, since most squares qualify for one or the other.
    const stepping = ctx.pending === null;
    const targets = new Set(stepping ? game.steps(state) : []);
    const landing = ctx.pending ?? state.mover;

    const destroyable = new Set<number>();
    if (!stepping) {
      for (let square = 0; square < I_SQUARES; square++) {
        if (!state.usable[square]) continue;
        if (square === landing || square === state.opponent) continue;
        destroyable.add(square);
      }
    }

    return Array.from({ length: I_SQUARES }, (_, display) => {
      const i = toBoard(display);
      const classes = ["cell"];
      if (!state.usable[i]) classes.push("gone");

      // The mover is whoever is to move now, which is you on your turn.
      if (i === state.mover && stepping) classes.push("piece", yourTurn ? "you" : "engine");
      else if (i === state.mover && !stepping) classes.push("piece", "ghost");
      else if (i === landing && !stepping) classes.push("piece", yourTurn ? "you" : "engine");
      else if (i === state.opponent) classes.push("piece", yourTurn ? "engine" : "you");

      const offered = !ctx.locked && (targets.has(i) || destroyable.has(i));
      if (offered) classes.push("playable", stepping ? "reach" : "doomed");

      const row = Math.floor(i / I_SIZE);
      return `<button class="${classes.join(" ")}" data-square="${i}"
        ${offered || (i === landing && !stepping) ? "" : "disabled"}
        aria-label="Row ${row + 1}, column ${(i % I_SIZE) + 1}"></button>`;
    }).join("");
  },

  select(element, ctx) {
    const value = element.dataset.square;
    if (value === undefined) return null;
    // `data-square` already holds the board index, not the displayed one, so the
    // flip never reaches the action encoding.
    const square = Number(value);
    const game = ctx.game as unknown as Isola;
    const state = ctx.state as never as ReturnType<Isola["initialState"]>;

    if (ctx.pending === null) {
      return game.steps(state).includes(square)
        ? { kind: "pending", pending: square }
        : null;
    }
    // Clicking the chosen square again takes the step back.
    if (square === ctx.pending) return { kind: "pending", pending: null };

    const direction = game.directionBetween(state.mover, ctx.pending);
    if (direction < 0) return null;
    return { kind: "action", action: direction * I_SQUARES + square };
  },

  detail(ctx) {
    const state = ctx.state as never as ReturnType<Isola["initialState"]>;
    let standing = 0;
    for (const value of state.usable) standing += value;
    return `<span class="spacer score">${standing} squares left</span>`;
  },

  // Each of the eight direction planes covers the whole board, so the visits are
  // summed per destroyed square - which is the half of the action that has a
  // place on the board.
  // Summed per destroyed square - the half of the action that has a place on
  // the board - and flipped to match, or it would describe a board the viewer
  // is not looking at.
  visits(counts, ctx) {
    const perSquare = new Array<number>(I_SQUARES).fill(0);
    counts.forEach((visits, action) => {
      perSquare[action % I_SQUARES] += visits;
    });
    if (!isolaFlipped(ctx)) return heatmap(perSquare, I_SQUARES, I_SIZE);

    const flipped = perSquare.map((_, display) => {
      const row = I_SIZE - 1 - Math.floor(display / I_SIZE);
      return perSquare[row * I_SIZE + (display % I_SIZE)];
    });
    return heatmap(flipped, I_SQUARES, I_SIZE);
  },
};

/** Lattice cell to line, and lattice cell to box, for drawing Dots & Boxes. */
const D_CELL_LINE = new Int16Array(D_LATTICE * D_LATTICE).fill(-1);
const D_CELL_BOX = new Int16Array(D_LATTICE * D_LATTICE).fill(-1);
for (let line = 0; line < D_LINES; line++) {
  const [row, col] = lineCell(line);
  D_CELL_LINE[row * D_LATTICE + col] = line;
}
for (let box = 0; box < D_SQUARES; box++) {
  const [row, col] = boxCell(box);
  D_CELL_BOX[row * D_LATTICE + col] = box;
}

/**
 * Who holds a box, in fixed colours, from its canonical owner.
 *
 * +1 means "the player to move now" - which after a bonus move is the same
 * player who just moved - so the seat, not the move count, says whose colour it is.
 */
function boxHolder(state: DotsAndBoxesState, box: number, humanFirst: boolean): string {
  const held = state.owner[box];
  if (held === 0) return "";
  const seat = held === 1 ? state.seat : 1 - state.seat;
  return seat === (humanFirst ? 0 : 1) ? "you" : "engine";
}

export const dotsAndBoxesView: View = {
  // The grid draws the board itself: dots and lines get narrow tracks and boxes
  // wide ones, so there is no background image to keep aligned.
  layout: "lattice",

  board(ctx) {
    const state = ctx.state as DotsAndBoxesState;
    const last = ctx.moves.length ? ctx.moves[ctx.moves.length - 1] : null;

    return Array.from({ length: D_LATTICE * D_LATTICE }, (_, cell) => {
      const line = D_CELL_LINE[cell];
      if (line >= 0) {
        const orientation = line < D_HORIZONTAL ? "h" : "v";
        const drawn = state.lines[line] === 1;
        const classes = ["line", orientation, drawn ? "drawn" : "",
                         line === last ? "last" : "",
                         !drawn && !ctx.locked ? "playable" : ""];
        return `<button class="${classes.filter(Boolean).join(" ")}" data-action="${line}"
          ${drawn || ctx.locked ? "disabled" : ""}
          aria-label="${orientation === "h" ? "Horizontal" : "Vertical"} line ${line + 1}"></button>`;
      }
      const box = D_CELL_BOX[cell];
      if (box >= 0) return `<span class="box ${boxHolder(state, box, ctx.humanFirst)}"></span>`;
      return `<span class="point"></span>`;
    }).join("");
  },

  select: wholeAction,

  detail(ctx) {
    const state = ctx.state as DotsAndBoxesState;
    let yours = 0;
    let theirs = 0;
    for (let box = 0; box < D_SQUARES; box++) {
      const holder = boxHolder(state, box, ctx.humanFirst);
      if (holder === "you") yours++;
      else if (holder === "engine") theirs++;
    }
    return `<span class="spacer score">
              <span class="dot you"></span>${yours}
              <span class="dot engine" style="margin-left:8px"></span>${theirs}
            </span>`;
  },

  // Line visits laid out on the same lattice, so the heatmap reads as the board.
  visits(counts) {
    const lattice = new Array<number>(D_LATTICE * D_LATTICE).fill(0);
    counts.slice(0, D_LINES).forEach((visits, line) => {
      const [row, col] = lineCell(line);
      lattice[row * D_LATTICE + col] = visits;
    });
    return heatmap(lattice, D_LATTICE * D_LATTICE, D_LATTICE);
  },
};

export const VIEWS: Record<string, View> = {
  connect4: connect4View,
  reversi: reversiView,
  gomoku: gomokuView,
  isola: isolaView,
  dotsandboxes: dotsAndBoxesView,
  chess: chessView,
};
