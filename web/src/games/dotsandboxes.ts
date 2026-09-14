import type { Game } from "./types";

/**
 * Dots and Boxes, mirroring `src/caissa/games/dotsandboxes.py`.
 *
 * Typed arrays here where Python uses bitmasks - sixty lines do not fit
 * JavaScript's 32-bit bitwise operators - but everything that crosses the
 * boundary must agree exactly: which lines are legal, whose turn it is, how the
 * game ends and what the network sees. `test/rules.test.ts` checks all four
 * against vectors generated from Python.
 */

export const BOXES = 5;
export const DOTS = BOXES + 1;
export const HORIZONTAL_LINES = DOTS * BOXES;
export const LINES = HORIZONTAL_LINES + BOXES * DOTS;
export const SQUARES = BOXES * BOXES;
export const LATTICE = 2 * BOXES + 1;

/** The line along the top of box (row, col); `row` runs up to BOXES. */
export const horizontal = (row: number, col: number): number => row * BOXES + col;
/** The line down the left of box (row, col); `col` runs up to BOXES. */
export const vertical = (row: number, col: number): number =>
  HORIZONTAL_LINES + row * DOTS + col;

/** The four lines around each box: top, bottom, left, right. */
export const BOX_LINES: ReadonlyArray<readonly number[]> = Array.from(
  { length: SQUARES },
  (_, box) => {
    const row = Math.floor(box / BOXES);
    const col = box % BOXES;
    return [horizontal(row, col), horizontal(row + 1, col), vertical(row, col), vertical(row, col + 1)];
  },
);

/** The one or two boxes each line borders. */
export const LINE_BOXES: ReadonlyArray<readonly number[]> = (() => {
  const adjacent: number[][] = Array.from({ length: LINES }, () => []);
  BOX_LINES.forEach((lines, box) => {
    for (const line of lines) adjacent[line].push(box);
  });
  return adjacent;
})();

/** A line's lattice cell: horizontal lines at (even, odd), vertical at (odd, even). */
export function lineCell(line: number): [number, number] {
  if (line < HORIZONTAL_LINES) {
    return [2 * Math.floor(line / BOXES), 2 * (line % BOXES) + 1];
  }
  const offset = line - HORIZONTAL_LINES;
  return [2 * Math.floor(offset / DOTS) + 1, 2 * (offset % DOTS)];
}

/** A box's centre cell on the lattice, at (odd, odd). */
export function boxCell(box: number): [number, number] {
  return [2 * Math.floor(box / BOXES) + 1, 2 * (box % BOXES) + 1];
}

export interface DotsAndBoxesState {
  /** 1 where a line is drawn. */
  readonly lines: Uint8Array;
  /** +1 the mover's box, -1 the opponent's, 0 open. Negated only when the turn passes. */
  readonly owner: Int8Array;
  /** The seat to move. The ply's parity no longer says this. */
  readonly seat: number;
  /** Lines drawn so far; the game ends at sixty. */
  readonly ply: number;
}

export class DotsAndBoxes implements Game<DotsAndBoxesState> {
  readonly name = "dotsandboxes";
  readonly actionSize = LINES;
  readonly boardShape = [LATTICE, LATTICE] as const;
  /** Lines drawn, the mover's boxes, the opponent's boxes, and two orientation masks. */
  readonly inputPlanes = 5;

  initialState(): DotsAndBoxesState {
    return { lines: new Uint8Array(LINES), owner: new Int8Array(SQUARES), seat: 0, ply: 0 };
  }

  toPlay(state: DotsAndBoxesState): number {
    return state.seat;
  }

  legalActions(state: DotsAndBoxesState): boolean[] {
    return Array.from(state.lines, (drawn) => drawn === 0);
  }

  apply(state: DotsAndBoxesState, action: number): DotsAndBoxesState {
    if (state.lines[action]) throw new Error(`line ${action} is already drawn`);

    const lines = new Uint8Array(state.lines);
    lines[action] = 1;
    const owner = new Int8Array(state.owner);

    let closed = false;
    for (const box of LINE_BOXES[action]) {
      if (BOX_LINES[box].every((line) => lines[line] === 1)) {
        owner[box] = 1; // the mover's, in canonical perspective
        closed = true;
      }
    }

    // A closed box is a bonus move: same seat, same perspective, nothing flips.
    if (closed) return { lines, owner, seat: state.seat, ply: state.ply + 1 };

    // Otherwise the turn passes, and the perspective passes with it.
    for (let box = 0; box < SQUARES; box++) owner[box] = -owner[box];
    return { lines, owner, seat: 1 - state.seat, ply: state.ply + 1 };
  }

  terminalValue(state: DotsAndBoxesState): number | null {
    if (state.ply < LINES) return null;
    let margin = 0;
    for (const held of state.owner) margin += held;
    // The last line always closes a box, so the game ends on the turn of whoever
    // drew it - which makes +1 routine here rather than impossible.
    return Math.sign(margin);
  }

  encode(state: DotsAndBoxesState): Float32Array {
    const cells = LATTICE * LATTICE;
    const planes = new Float32Array(this.inputPlanes * cells);
    for (let line = 0; line < LINES; line++) {
      const [row, col] = lineCell(line);
      const cell = row * LATTICE + col;
      if (state.lines[line]) planes[cell] = 1;
      // The constant orientation masks: plane 3 horizontal, plane 4 vertical.
      planes[(line < HORIZONTAL_LINES ? 3 : 4) * cells + cell] = 1;
    }
    for (let box = 0; box < SQUARES; box++) {
      const [row, col] = boxCell(box);
      const cell = row * LATTICE + col;
      if (state.owner[box] === 1) planes[cells + cell] = 1;
      else if (state.owner[box] === -1) planes[2 * cells + cell] = 1;
    }
    return planes;
  }

  /** Boxes held, as [player to move, opponent]. */
  score(state: DotsAndBoxesState): [number, number] {
    let mine = 0;
    let theirs = 0;
    for (const held of state.owner) {
      if (held === 1) mine++;
      else if (held === -1) theirs++;
    }
    return [mine, theirs];
  }
}
