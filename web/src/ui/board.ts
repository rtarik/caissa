import type { Game } from "../games/types";

/** 0 empty, 1 the player who moved first, 2 the player who moved second. */
export type Grid = Uint8Array;

/**
 * The board as a viewer sees it, with colours fixed.
 *
 * The engine's board flips sign whenever the turn passes, so the network always
 * sees itself as +1 - which is exactly what a display must not do. The seat to
 * move says who that +1 currently is, and turns the canonical board back into
 * stable colours. (This used ply parity until Dots & Boxes made the two differ.)
 *
 * Derived rather than tracked, which matters in Reversi: discs change owner, so
 * a view built by remembering where each piece was placed would be wrong from
 * the first capture.
 */
export function absoluteGrid<S>(game: Game<S>, state: S): Grid {
  const [height, width] = game.boardShape;
  const encoded = game.encode(state);
  const squares = height * width;

  const mover = game.toPlay(state) === 0 ? 1 : 2;
  const opponent = 3 - mover;

  const grid = new Uint8Array(squares);
  for (let i = 0; i < squares; i++) {
    if (encoded[i] === 1) grid[i] = mover;
    else if (encoded[squares + i] === 1) grid[i] = opponent;
  }
  return grid;
}

/** The four-in-a-row through the last move, for highlighting. Connect 4 only. */
export function winningLine(grid: Grid, width: number, height: number,
                            lastMove: number | null): number[] | null {
  if (lastMove === null) return null;
  const player = grid[lastMove];
  if (player === 0) return null;

  const row = Math.floor(lastMove / width);
  const col = lastMove % width;
  for (const [dRow, dCol] of [[0, 1], [1, 0], [1, 1], [1, -1]] as const) {
    const line = [lastMove];
    for (const sign of [1, -1]) {
      let r = row + dRow * sign;
      let c = col + dCol * sign;
      while (r >= 0 && r < height && c >= 0 && c < width && grid[r * width + c] === player) {
        line.push(r * width + c);
        r += dRow * sign;
        c += dCol * sign;
      }
    }
    if (line.length >= 4) return line;
  }
  return null;
}

/** Where the last Connect 4 drop landed, for the move marker. */
export function connect4LastMove(moves: number[], width: number, height: number): number | null {
  if (moves.length === 0) return null;
  const heights = new Array<number>(width).fill(0);
  let index = 0;
  for (const col of moves) {
    index = (height - 1 - heights[col]) * width + col;
    heights[col] += 1;
  }
  return index;
}
