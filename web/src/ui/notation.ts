/**
 * What each game calls a move, for the move list.
 *
 * Chess has a notation everyone uses, and its moves are named in it. The other
 * games get the plainest thing that still says where the move went: a column, a
 * square, a line between two dots. Every name is absolute - board coordinates,
 * not the flipped view a player might be looking at - so a move keeps its name
 * whichever side of the board you sat on.
 */
import { san } from "./chessview";
import { moveOf, type ChessState } from "../games/chess";
import { COLS as C4_COLS } from "../games/connect4";
import { BOXES, DOTS, HORIZONTAL_LINES } from "../games/dotsandboxes";
import { SIZE as G_SIZE } from "../games/gomoku";
import { SIZE as I_SIZE, SQUARES as I_SQUARES, step } from "../games/isola";
import { PASS as R_PASS, SIZE as R_SIZE } from "../games/reversi";

const letter = (index: number) => String.fromCharCode(97 + index);

/** A square named by column letter and row number, row 1 at the top. */
function fromTop(square: number, size: number): string {
  return `${letter(square % size)}${Math.floor(square / size) + 1}`;
}

/** A square named by column letter and row number, row 1 at the bottom. */
function fromBottom(square: number, size: number): string {
  return `${letter(square % size)}${size - Math.floor(square / size)}`;
}

type Labeller = (before: unknown, action: number) => string;

const LABELS: Record<string, Labeller> = {
  // Columns, numbered as players count them.
  connect4: (_before, action) => `${Math.min(action, C4_COLS - 1) + 1}`,

  // Othello notation: a1 in the top-left corner.
  reversi: (_before, action) => (action === R_PASS ? "pass" : fromTop(action, R_SIZE)),

  // Go-style: row numbers count up from the bottom edge.
  gomoku: (_before, action) => fromBottom(action, G_SIZE),

  // Where the piece landed, and the square taken away.
  isola: (before, action) => {
    const { mover } = before as { mover: number };
    const direction = Math.floor(action / I_SQUARES);
    const landing = step(mover, direction);
    return `${fromBottom(landing, I_SIZE)} ×${fromBottom(action % I_SQUARES, I_SIZE)}`;
  },

  // A line, by the two dots it joins: dots lettered by column, numbered from the top.
  dotsandboxes: (_before, action) => {
    const dot = (row: number, col: number) => `${letter(col)}${row + 1}`;
    if (action < HORIZONTAL_LINES) {
      const row = Math.floor(action / BOXES);
      const col = action % BOXES;
      return `${dot(row, col)}-${dot(row, col + 1)}`;
    }
    const offset = action - HORIZONTAL_LINES;
    const row = Math.floor(offset / DOTS);
    const col = offset % DOTS;
    return `${dot(row, col)}-${dot(row + 1, col)}`;
  },

  chess: (before, action) => {
    const state = before as ChessState;
    return san(state, moveOf(state, action));
  },
};

/** A move's name in its game's notation, given the position it was played from. */
export function moveLabel(game: string, before: unknown, action: number): string {
  return (LABELS[game] ?? ((_b, a) => `${a}`))(before, action);
}
