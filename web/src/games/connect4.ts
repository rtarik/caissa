import type { Game } from "./types";

export const ROWS = 6;
export const COLS = 7;
const CONNECT = 4;

/** Row 0 is the top of the grid, matching the Python implementation. */
const index = (row: number, col: number) => row * COLS + col;

/** Horizontal, vertical, and the two diagonals, as [dRow, dCol]. */
const DIRECTIONS: ReadonlyArray<readonly [number, number]> = [
  [0, 1],
  [1, 0],
  [1, 1],
  [1, -1],
];

export interface Connect4State {
  /** +1 for the player about to move, -1 for the opponent, 0 empty. */
  readonly board: Int8Array;
  /** Flat index of the previous move, or null at the start. */
  readonly lastMove: number | null;
  readonly ply: number;
}

export class Connect4 implements Game<Connect4State> {
  readonly name = "connect4";
  readonly actionSize = COLS;
  readonly boardShape = [ROWS, COLS] as const;
  readonly inputPlanes = 2;

  initialState(): Connect4State {
    return { board: new Int8Array(ROWS * COLS), lastMove: null, ply: 0 };
  }

  toPlay(state: Connect4State): number {
    // Turns strictly alternate, so the ply's parity is the seat.
    return state.ply % 2;
  }

  legalActions(state: Connect4State): boolean[] {
    // A column accepts a piece exactly when its top cell is still empty.
    return Array.from({ length: COLS }, (_, col) => state.board[index(0, col)] === 0);
  }

  apply(state: Connect4State, action: number): Connect4State {
    let row = -1;
    for (let r = ROWS - 1; r >= 0; r--) {
      if (state.board[index(r, action)] === 0) {
        row = r;
        break;
      }
    }
    if (row < 0) throw new Error(`column ${action} is full`);

    const board = new Int8Array(state.board);
    board[index(row, action)] = 1;
    // Flip, so the next player also sees their own pieces as +1.
    for (let i = 0; i < board.length; i++) board[i] = -board[i] as number;

    return { board, lastMove: index(row, action), ply: state.ply + 1 };
  }

  terminalValue(state: Connect4State): number | null {
    if (state.lastMove === null) return null;

    // The previous mover's pieces are -1 now, because apply() flipped the board.
    if (this.winsThrough(state.board, state.lastMove, -1)) return -1;

    for (let col = 0; col < COLS; col++) {
      if (state.board[index(0, col)] === 0) return null;
    }
    return 0; // grid full, nobody connected four
  }

  private winsThrough(board: Int8Array, square: number, player: number): boolean {
    const row = Math.floor(square / COLS);
    const col = square % COLS;

    for (const [dRow, dCol] of DIRECTIONS) {
      let count = 1;
      for (const sign of [1, -1]) {
        let r = row + dRow * sign;
        let c = col + dCol * sign;
        while (r >= 0 && r < ROWS && c >= 0 && c < COLS && board[index(r, c)] === player) {
          count++;
          if (count >= CONNECT) return true;
          r += dRow * sign;
          c += dCol * sign;
        }
      }
    }
    return false;
  }

  encode(state: Connect4State): Float32Array {
    // One plane for the mover's pieces, one for the opponent's. No plane says
    // whose turn it is: canonically, the answer is always "the first one".
    const planes = new Float32Array(this.inputPlanes * ROWS * COLS);
    const squares = ROWS * COLS;
    for (let i = 0; i < squares; i++) {
      if (state.board[i] === 1) planes[i] = 1;
      else if (state.board[i] === -1) planes[squares + i] = 1;
    }
    return planes;
  }

  /** Row a piece would land in, or -1 if the column is full. Used by the UI. */
  landingRow(state: Connect4State, col: number): number {
    for (let r = ROWS - 1; r >= 0; r--) {
      if (state.board[index(r, col)] === 0) return r;
    }
    return -1;
  }
}
