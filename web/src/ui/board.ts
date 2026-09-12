import { COLS, ROWS } from "../games/connect4";

/** 0 empty, 1 the player who moved first, 2 the player who moved second. */
export type Grid = Uint8Array;

/**
 * Rebuild the visible board from the move list.
 *
 * Kept separate from the canonical board the engine uses. That one flips sign
 * every ply so the network always sees itself as +1, which is exactly what a
 * display must *not* do - the colours have to stay where the player put them.
 * Deriving the view from the move list keeps the two representations from being
 * confused for one another.
 */
export function gridFromMoves(moves: number[]): Grid {
  const grid = new Uint8Array(ROWS * COLS);
  const heights = new Array<number>(COLS).fill(0);
  moves.forEach((col, i) => {
    const row = ROWS - 1 - heights[col];
    heights[col] += 1;
    grid[row * COLS + col] = ((i % 2) + 1) as 1 | 2;
  });
  return grid;
}

/** The four cells of the winning line, if the last move made one. */
export function winningLine(grid: Grid, lastMove: number | null): number[] | null {
  if (lastMove === null) return null;
  const player = grid[lastMove];
  if (player === 0) return null;

  const row = Math.floor(lastMove / COLS);
  const col = lastMove % COLS;
  for (const [dRow, dCol] of [[0, 1], [1, 0], [1, 1], [1, -1]] as const) {
    const line = [lastMove];
    for (const sign of [1, -1]) {
      let r = row + dRow * sign;
      let c = col + dCol * sign;
      while (r >= 0 && r < ROWS && c >= 0 && c < COLS && grid[r * COLS + c] === player) {
        line.push(r * COLS + c);
        r += dRow * sign;
        c += dCol * sign;
      }
    }
    if (line.length >= 4) return line;
  }
  return null;
}

export function lastMoveIndex(moves: number[]): number | null {
  if (moves.length === 0) return null;
  const heights = new Array<number>(COLS).fill(0);
  let index = 0;
  moves.forEach((col) => {
    index = (ROWS - 1 - heights[col]) * COLS + col;
    heights[col] += 1;
  });
  return index;
}
