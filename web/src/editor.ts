/**
 * Setting up a chess position: the board as the editor holds it, FEN in and
 * out, and what is wrong with a setup before anyone tries to play it.
 *
 * The editor works on a looser object than a game position. A game position has
 * to be legal - chess.js will not even construct one without both kings - but a
 * board being set up is half-built most of the time, so it is plain squares and
 * flags here, checked on demand, and only turned into a real position once it
 * passes.
 *
 * chess.js already refuses a missing or doubled king and pawns on the edge
 * ranks. It accepts, and so this checks, the rest: the side not to move standing
 * in check (the side to move could take the king), castling with no rook to
 * castle with, more pawns or pieces than a side can have, and a game that is
 * over before it starts.
 */
import { Chess as Board, DEFAULT_POSITION, validateFen } from "chess.js";

export type Piece = "K" | "Q" | "R" | "B" | "N" | "P" | "k" | "q" | "r" | "b" | "n" | "p";
export const WHITE_PIECES: Piece[] = ["K", "Q", "R", "B", "N", "P"];
export const BLACK_PIECES: Piece[] = ["k", "q", "r", "b", "n", "p"];

export interface Castling {
  /** White king side, white queen side, black king side, black queen side. */
  K: boolean;
  Q: boolean;
  k: boolean;
  q: boolean;
}

export interface Setup {
  /** Sixty-four squares, a1 first and h8 last, as the rest of the page numbers them. */
  squares: (Piece | null)[];
  whiteToMove: boolean;
  castling: Castling;
  /** Kept from a pasted FEN, "-" otherwise; any edit clears it. */
  enPassant: string;
  halfmove: number;
  fullmove: number;
}

export const STANDARD_FEN = DEFAULT_POSITION;

const isPiece = (c: string): c is Piece => "KQRBNPkqrbnp".includes(c) && c.length === 1;
const NONE: Castling = { K: false, Q: false, k: false, q: false };

export function emptySetup(): Setup {
  return {
    squares: new Array(64).fill(null),
    whiteToMove: true,
    castling: { ...NONE },
    enPassant: "-",
    halfmove: 0,
    fullmove: 1,
  };
}

/** Read a FEN; null when it is not one. Needs the board and the side to move at least. */
export function parseFen(fen: string): Setup | null {
  const fields = fen.trim().split(/\s+/);
  if (fields.length < 2) return null;
  const [placement, side, rights = "-", passant = "-", half = "0", full = "1"] = fields;

  const ranks = placement.split("/");
  if (ranks.length !== 8) return null;
  const squares: (Piece | null)[] = new Array(64).fill(null);
  for (let r = 0; r < 8; r++) {
    const rank = 7 - r; // the FEN lists rank 8 first
    let file = 0;
    for (const c of ranks[r]) {
      if (/[1-8]/.test(c)) file += Number(c);
      else if (isPiece(c) && file < 8) squares[rank * 8 + file++] = c;
      else return null;
    }
    if (file !== 8) return null;
  }
  if (side !== "w" && side !== "b") return null;
  if (!/^(-|[KQkq]{1,4})$/.test(rights)) return null;
  if (!/^(-|[a-h][36])$/.test(passant)) return null;

  const castling = { ...NONE };
  for (const c of rights) if (c in castling) castling[c as keyof Castling] = true;
  return {
    squares,
    whiteToMove: side === "w",
    castling,
    enPassant: passant,
    halfmove: Math.max(0, Number.parseInt(half, 10) || 0),
    fullmove: Math.max(1, Number.parseInt(full, 10) || 1),
  };
}

export function fenOf(setup: Setup): string {
  const ranks: string[] = [];
  for (let rank = 7; rank >= 0; rank--) {
    let row = "";
    let empty = 0;
    for (let file = 0; file < 8; file++) {
      const piece = setup.squares[rank * 8 + file];
      if (!piece) {
        empty++;
        continue;
      }
      if (empty) row += empty;
      empty = 0;
      row += piece;
    }
    ranks.push(row + (empty ? empty : ""));
  }
  const rights = (["K", "Q", "k", "q"] as const).filter((c) => setup.castling[c]).join("") || "-";
  return `${ranks.join("/")} ${setup.whiteToMove ? "w" : "b"} ${rights} ${setup.enPassant} ${setup.halfmove} ${setup.fullmove}`;
}

/**
 * Which castling rights the pieces allow: the king and that rook still on their
 * starting squares. A right without them is not a right, only an illegal FEN.
 */
export function possibleCastling(squares: (Piece | null)[]): Castling {
  return {
    K: squares[4] === "K" && squares[7] === "R",
    Q: squares[4] === "K" && squares[0] === "R",
    k: squares[60] === "k" && squares[63] === "r",
    q: squares[60] === "k" && squares[56] === "r",
  };
}

/** Keep only the rights the pieces still allow. */
function allowed(castling: Castling, squares: (Piece | null)[]): Castling {
  const possible = possibleCastling(squares);
  return { K: castling.K && possible.K, Q: castling.Q && possible.Q, k: castling.k && possible.k, q: castling.q && possible.q };
}

/**
 * Put a piece on a square, or take one off with null.
 *
 * Any edit forgets a pasted en passant square, which described the move before
 * a position that no longer exists, and drops castling rights the pieces no
 * longer allow.
 */
export function withPiece(setup: Setup, square: number, piece: Piece | null): Setup {
  const squares = setup.squares.slice();
  squares[square] = piece;
  return { ...setup, squares, enPassant: "-", castling: allowed(setup.castling, squares) };
}

export function withCastling(setup: Setup, right: keyof Castling, on: boolean): Setup {
  const castling = { ...setup.castling, [right]: on };
  return { ...setup, castling: allowed(castling, setup.squares) };
}

export function withSideToMove(setup: Setup, whiteToMove: boolean): Setup {
  return { ...setup, whiteToMove, enPassant: "-" };
}

/**
 * What stops this setup being played, in words, or nothing when it can be.
 *
 * Checked in the order a person would fix them: pieces first, then who is to
 * move, then whether there would be a game at all.
 */
export function problems(setup: Setup): string[] {
  const found: string[] = [];
  const count = (p: Piece) => setup.squares.filter((s) => s === p).length;
  const sides = [["White", "K", "P", WHITE_PIECES], ["Black", "k", "p", BLACK_PIECES]] as const;
  for (const [name, king, pawn, pieces] of sides) {
    const kings = count(king);
    if (kings === 0) found.push(`${name} needs a king.`);
    if (kings > 1) found.push(`${name} can only have one king.`);
    if (count(pawn) > 8) found.push(`${name} can't have more than eight pawns.`);
    const total = setup.squares.filter((s) => s && (pieces as readonly string[]).includes(s)).length;
    if (total > 16) found.push(`${name} can't have more than sixteen pieces.`);
  }
  const edge = [...setup.squares.slice(0, 8), ...setup.squares.slice(56)];
  if (edge.some((s) => s === "P" || s === "p")) found.push("Pawns can't stand on the first or last rank.");
  if (found.length) return found;

  const fen = fenOf(setup);
  const check = validateFen(fen);
  if (!check.ok) return [(check.error ?? "That position isn't valid.").replace(/^Invalid FEN: /, "")];

  // The side not to move must not be in check: whoever is to move could simply
  // take the king. Asked by handing the move to the other side and looking.
  const other = fenOf({ ...setup, whiteToMove: !setup.whiteToMove, enPassant: "-" });
  try {
    if (new Board(other).inCheck()) {
      const checked = setup.whiteToMove ? "Black" : "White";
      return [`${checked} is in check, so it has to be ${checked} to move.`];
    }
    const board = new Board(fen);
    if (board.isCheckmate()) return ["That's checkmate already, so there's nothing to play."];
    if (board.isStalemate()) return ["That's stalemate already, so there's nothing to play."];
    if (board.isInsufficientMaterial()) return ["Neither side has enough pieces left to win."];
  } catch {
    return ["That position isn't valid."];
  }
  return [];
}
