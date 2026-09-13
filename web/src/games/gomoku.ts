import type { Game } from "./types";

export const SIZE = 9;
export const SQUARES = SIZE * SIZE;
export const CONNECT = 5;

/** Horizontal, vertical, and the two diagonals. */
const DIRECTIONS: ReadonlyArray<readonly [number, number]> = [
  [0, 1], [1, 0], [1, 1], [1, -1],
];

export interface GomokuState {
  /** +1 for the player about to move, -1 for the opponent, 0 empty. */
  readonly board: Int8Array;
  /** Flat index of the previous move, so terminal checks stay local. */
  readonly lastMove: number | null;
  readonly ply: number;
}

export class Gomoku implements Game<GomokuState> {
  readonly name = "gomoku";
  readonly actionSize = SQUARES;
  readonly boardShape = [SIZE, SIZE] as const;
  readonly inputPlanes = 2;

  initialState(): GomokuState {
    return { board: new Int8Array(SQUARES), lastMove: null, ply: 0 };
  }

  legalActions(state: GomokuState): boolean[] {
    // Every empty intersection: free-style Gomoku, with none of the opening
    // handicaps some rulesets add to curb the first player's advantage.
    return Array.from(state.board, (value) => value === 0);
  }

  apply(state: GomokuState, action: number): GomokuState {
    if (state.board[action] !== 0) {
      throw new Error(`square ${action} is already occupied`);
    }
    const board = new Int8Array(state.board);
    board[action] = 1;
    // Flip, so the next player also sees their own stones as +1.
    for (let i = 0; i < board.length; i++) board[i] = -board[i] as number;
    return { board, lastMove: action, ply: state.ply + 1 };
  }

  terminalValue(state: GomokuState): number | null {
    if (state.lastMove === null) return null;

    // The previous mover's stones are -1 now, because apply() flipped the board.
    if (this.winsThrough(state.board, state.lastMove, -1)) return -1;
    for (const value of state.board) if (value === 0) return null;
    return 0; // board full, nobody made five
  }

  private winsThrough(board: Int8Array, square: number, player: number): boolean {
    const row = Math.floor(square / SIZE);
    const col = square % SIZE;

    for (const [dRow, dCol] of DIRECTIONS) {
      let count = 1;
      for (const sign of [1, -1]) {
        let r = row + dRow * sign;
        let c = col + dCol * sign;
        while (r >= 0 && r < SIZE && c >= 0 && c < SIZE && board[r * SIZE + c] === player) {
          count++;
          if (count >= CONNECT) return true;
          r += dRow * sign;
          c += dCol * sign;
        }
      }
    }
    return false;
  }

  encode(state: GomokuState): Float32Array {
    const planes = new Float32Array(this.inputPlanes * SQUARES);
    for (let i = 0; i < SQUARES; i++) {
      if (state.board[i] === 1) planes[i] = 1;
      else if (state.board[i] === -1) planes[SQUARES + i] = 1;
    }
    return planes;
  }

  /** The five cells of the winning line, for highlighting. */
  winningLine(state: GomokuState): number[] | null {
    if (state.lastMove === null) return null;
    const row = Math.floor(state.lastMove / SIZE);
    const col = state.lastMove % SIZE;

    for (const [dRow, dCol] of DIRECTIONS) {
      const cells = [state.lastMove];
      for (const sign of [1, -1]) {
        let r = row + dRow * sign;
        let c = col + dCol * sign;
        while (r >= 0 && r < SIZE && c >= 0 && c < SIZE && state.board[r * SIZE + c] === -1) {
          cells.push(r * SIZE + c);
          r += dRow * sign;
          c += dCol * sign;
        }
      }
      if (cells.length >= CONNECT) return cells;
    }
    return null;
  }
}
