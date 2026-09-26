/**
 * The move list's two pieces of logic: how plies are laid out, and what Undo
 * takes back. Both are about turns, and turns are where this project has been
 * wrong before - Dots & Boxes gives a player several plies in a row - so both
 * work from who actually moved, never from counting.
 */

/** One ply as the move list shows it. */
export interface Ply {
  /** Position in the game: ply 0 is the first move made. */
  index: number;
  /** The seat that made it: 0 moves first. */
  seat: number;
  label: string;
}

/** A numbered line of the move list: what each side did in one turn. */
export interface Row {
  number: number;
  /** The first side's plies, then the second side's. Either can be empty. */
  cells: [Ply[], Ply[]];
}

/**
 * Lay plies out as numbered rows of (first side, second side).
 *
 * Consecutive plies by the same side form one turn and share a cell, so a Dots
 * & Boxes chain of three boxes reads as one entry rather than breaking the two
 * columns. A game whose record begins with the second side to move - a position
 * set up with Black to play - starts with an empty first cell, as a chess score
 * sheet does with "1. ... e5".
 */
export function rowsOf(plies: Ply[], firstNumber = 1): Row[] {
  const turns: { seat: number; plies: Ply[] }[] = [];
  for (const ply of plies) {
    const current = turns[turns.length - 1];
    if (current && current.seat === ply.seat) current.plies.push(ply);
    else turns.push({ seat: ply.seat, plies: [ply] });
  }

  // Turns alternate by construction - a side's consecutive plies were merged
  // above - so a second-side turn always follows a first-side one, whose row
  // still has its second cell free. Only a record that opens with the second
  // side has no row to join.
  const rows: Row[] = [];
  for (const turn of turns) {
    const last = rows[rows.length - 1];
    if (turn.seat === 1 && last) {
      last.cells[1] = turn.plies;
      continue;
    }
    const cells: [Ply[], Ply[]] = turn.seat === 0 ? [turn.plies, []] : [[], turn.plies];
    rows.push({ number: firstNumber + rows.length, cells });
  }
  return rows;
}

/**
 * How long the move list should be after an undo, or null if there is nothing
 * of the player's to undo.
 *
 * Undo means "let me have my last move back": everything from the player's most
 * recent move onwards goes, the engine's reply with it, and it is their turn in
 * the position they chose from. Forced passes do not count as moves to take
 * back - undoing one would only have the page pass again at once - so the walk
 * carries on past them to the last move the player actually chose.
 */
export function undoTarget(
  seats: readonly number[],
  moves: readonly number[],
  human: number,
  pass?: number,
): number | null {
  for (let index = moves.length - 1; index >= 0; index--) {
    if (seats[index] === human && moves[index] !== pass) return index;
  }
  return null;
}
