import type { Game } from "./types";

export const SIZE = 8;
export const SQUARES = SIZE * SIZE;
/** Squares, then one more for the pass. */
export const ACTIONS = SQUARES + 1;
export const PASS = SQUARES;

const DIRECTIONS: ReadonlyArray<readonly [number, number]> = [
  [-1, -1], [-1, 0], [-1, 1],
  [0, -1], [0, 1],
  [1, -1], [1, 0], [1, 1],
];

const index = (row: number, col: number) => row * SIZE + col;

export interface ReversiState {
  /** +1 for the player about to move, -1 for the opponent, 0 empty. */
  readonly board: Int8Array;
  /** Consecutive passes; two means neither side can move and the game is over. */
  readonly passes: number;
  readonly ply: number;
}

/**
 * Discs the mover would capture by playing here - empty when the move is illegal.
 *
 * Both the legality test and the move itself, so the two cannot disagree.
 */
export function captures(board: Int8Array, row: number, col: number): number[] {
  if (board[index(row, col)] !== 0) return [];

  const captured: number[] = [];
  for (const [dRow, dCol] of DIRECTIONS) {
    const run: number[] = [];
    let r = row + dRow;
    let c = col + dCol;
    while (r >= 0 && r < SIZE && c >= 0 && c < SIZE && board[index(r, c)] === -1) {
      run.push(index(r, c));
      r += dRow;
      c += dCol;
    }
    // A run only counts when it is closed by one of the mover's own discs.
    if (run.length > 0 && r >= 0 && r < SIZE && c >= 0 && c < SIZE && board[index(r, c)] === 1) {
      captured.push(...run);
    }
  }
  return captured;
}

export class Reversi implements Game<ReversiState> {
  readonly name = "reversi";
  readonly actionSize = ACTIONS;
  readonly boardShape = [SIZE, SIZE] as const;
  readonly inputPlanes = 2;
  readonly passAction = PASS;

  initialState(): ReversiState {
    const board = new Int8Array(SQUARES);
    // Black moves first and is therefore the mover, so black is +1.
    board[index(3, 4)] = 1;
    board[index(4, 3)] = 1;
    board[index(3, 3)] = -1;
    board[index(4, 4)] = -1;
    return { board, passes: 0, ply: 0 };
  }

  toPlay(state: ReversiState): number {
    // A pass is a real action that hands the move over, so parity holds.
    return state.ply % 2;
  }

  legalActions(state: ReversiState): boolean[] {
    const legal = new Array<boolean>(ACTIONS).fill(false);
    let any = false;
    for (let square = 0; square < SQUARES; square++) {
      const row = Math.floor(square / SIZE);
      const col = square % SIZE;
      if (captures(state.board, row, col).length > 0) {
        legal[square] = true;
        any = true;
      }
    }
    // Passing is legal only when nothing else is; declining a turn is not the game.
    if (!any) legal[PASS] = true;
    return legal;
  }

  apply(state: ReversiState, action: number): ReversiState {
    if (action === PASS) {
      if (this.legalActions(state).slice(0, SQUARES).some(Boolean)) {
        throw new Error("cannot pass while a move is available");
      }
      const board = new Int8Array(state.board);
      for (let i = 0; i < board.length; i++) board[i] = -board[i] as number;
      return { board, passes: state.passes + 1, ply: state.ply + 1 };
    }

    const row = Math.floor(action / SIZE);
    const col = action % SIZE;
    const flipped = captures(state.board, row, col);
    if (flipped.length === 0) {
      throw new Error(`square ${action} captures nothing, so it is illegal`);
    }

    const board = new Int8Array(state.board);
    board[action] = 1;
    for (const square of flipped) board[square] = 1;
    // Flip, so the next player also sees their own discs as +1.
    for (let i = 0; i < board.length; i++) board[i] = -board[i] as number;

    return { board, passes: 0, ply: state.ply + 1 };
  }

  terminalValue(state: ReversiState): number | null {
    if (state.passes < 2) return null;
    let mine = 0;
    let theirs = 0;
    for (const value of state.board) {
      if (value === 1) mine++;
      else if (value === -1) theirs++;
    }
    // Mover-relative, as everywhere.
    return Math.sign(mine - theirs);
  }

  encode(state: ReversiState): Float32Array {
    const planes = new Float32Array(this.inputPlanes * SQUARES);
    for (let i = 0; i < SQUARES; i++) {
      if (state.board[i] === 1) planes[i] = 1;
      else if (state.board[i] === -1) planes[SQUARES + i] = 1;
    }
    return planes;
  }

  /** Disc counts as [mover, opponent], for the UI. */
  score(state: ReversiState): [number, number] {
    let mine = 0;
    let theirs = 0;
    for (const value of state.board) {
      if (value === 1) mine++;
      else if (value === -1) theirs++;
    }
    return [mine, theirs];
  }
}
