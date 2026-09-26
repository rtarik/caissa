/**
 * Chess, ported from `src/caissa/games/chess.py`.
 *
 * The rules come from chess.js here as they come from python-chess there. What
 * is ported by hand - and checked against Python-generated vectors - is the
 * translation between chess and the network: the mirrored view for Black, the
 * 73 x 64 action table and the 19 input planes. A difference in any of them and
 * the network is fed positions it was never trained on, while every move it
 * plays is still perfectly legal.
 *
 * Repetition is tracked here rather than left to chess.js, because each state
 * must stand on its own: the search builds a position from its parent's FEN, not
 * by replaying the game that led to it.
 *
 * **Speed, and one piece of private API.** chess.js's public `moves({verbose:
 * true})` builds every move's notation and the position before and after it -
 * 0.8 ms a call, more than a network evaluation, and nothing the search needs.
 * The generator underneath, `_moves`, returns plain moves in a fraction of that.
 * It is private, so it is reached in exactly one place (`internals` below),
 * chess.js is pinned to an exact version, and the cross-language vectors fail
 * loudly if an upgrade changes it. The page uses the public API for notation.
 */
import { Chess as Board, DEFAULT_POSITION, type Square } from "chess.js";
import type { Game } from "./types";

export const SQUARES = 64;

/** (file step, rank step) in the mover's frame: N, NE, E, SE, S, SW, W, NW. */
const QUEEN_DIRECTIONS: readonly (readonly [number, number])[] = [
  [0, 1], [1, 1], [1, 0], [1, -1], [0, -1], [-1, -1], [-1, 0], [-1, 1],
];
const KNIGHT_JUMPS: readonly (readonly [number, number])[] = [
  [1, 2], [2, 1], [2, -1], [1, -2], [-1, -2], [-2, -1], [-2, 1], [-1, 2],
];
export type Underpromotion = "n" | "b" | "r";
const UNDERPROMOTIONS: readonly Underpromotion[] = ["n", "b", "r"];
const PROMOTION_FILE_STEPS = [-1, 0, 1];

export const QUEEN_TYPES = QUEEN_DIRECTIONS.length * 7;
export const KNIGHT_TYPES = KNIGHT_JUMPS.length;
export const MOVE_TYPES =
  QUEEN_TYPES + KNIGHT_TYPES + PROMOTION_FILE_STEPS.length * UNDERPROMOTIONS.length;
export const ACTIONS = MOVE_TYPES * SQUARES;
export const PLANES = 19;

const PIECES = ["p", "n", "b", "r", "q", "k"];

/** A move in the mover's frame, squares numbered rank * 8 + file. */
export interface FrameMove {
  from: number;
  to: number;
  promotion: Underpromotion | null;
}

/** A move as chess.js and the page name it: squares like "e2", promotion like "q". */
export interface NamedMove {
  from: string;
  to: string;
  promotion: string | null;
}

function buildMoves(): (FrameMove | null)[] {
  const moves: (FrameMove | null)[] = new Array(ACTIONS).fill(null);
  const target = (square: number, fileStep: number, rankStep: number): number | null => {
    const file = (square % 8) + fileStep;
    const rank = Math.floor(square / 8) + rankStep;
    return file >= 0 && file < 8 && rank >= 0 && rank < 8 ? rank * 8 + file : null;
  };
  for (let square = 0; square < SQUARES; square++) {
    QUEEN_DIRECTIONS.forEach(([fileStep, rankStep], direction) => {
      for (let distance = 1; distance < 8; distance++) {
        const to = target(square, fileStep * distance, rankStep * distance);
        if (to !== null) {
          moves[(direction * 7 + distance - 1) * SQUARES + square] = { from: square, to, promotion: null };
        }
      }
    });
    KNIGHT_JUMPS.forEach(([fileStep, rankStep], jump) => {
      const to = target(square, fileStep, rankStep);
      if (to !== null) moves[(QUEEN_TYPES + jump) * SQUARES + square] = { from: square, to, promotion: null };
    });
    if (Math.floor(square / 8) === 6) {
      PROMOTION_FILE_STEPS.forEach((fileStep, step) => {
        const to = target(square, fileStep, 1);
        if (to === null) return;
        UNDERPROMOTIONS.forEach((piece, index) => {
          const type = QUEEN_TYPES + KNIGHT_TYPES + step * UNDERPROMOTIONS.length + index;
          moves[type * SQUARES + square] = { from: square, to, promotion: piece };
        });
      });
    }
  }
  return moves;
}

/** Action index -> move in the mover's frame, or null where it would leave the board. */
export const MOVES = buildMoves();
const frameKey = (from: number, to: number, promotion: string | null) =>
  `${from}-${to}-${promotion ?? ""}`;
const INDEX = new Map<string, number>();
MOVES.forEach((move, action) => {
  if (move) INDEX.set(frameKey(move.from, move.to, move.promotion), action);
});

export const squareIndex = (name: string): number =>
  name.charCodeAt(0) - 97 + 8 * (name.charCodeAt(1) - 49);
export const squareName = (index: number): string =>
  String.fromCharCode(97 + (index % 8)) + String(Math.floor(index / 8) + 1);
/** The same square seen from the other side of the board. */
const mirror = (square: number): number => square ^ 56;

/** A legal move as chess.js generates it internally: squares in its 0x88 layout. */
interface RawMove {
  from: number;
  to: number;
  piece: string;
  captured?: string;
  promotion?: string;
  flags: number;
}

interface Internals {
  _moves(options: { legal: boolean }): RawMove[];
  _makeMove(move: RawMove): void;
}

/** The one place chess.js's private generator is reached (see the module comment). */
const internals = (board: Board): Internals => board as unknown as Internals;

/** chess.js numbers squares 0x88-style from a8 = 0; here they are rank * 8 + file. */
const fromOx88 = (square: number): number => (7 - (square >> 4)) * 8 + (square & 7);
const toOx88 = (square: number): number => (7 - (square >> 3)) * 16 + (square & 7);

const named = (move: RawMove): NamedMove => ({
  from: squareName(fromOx88(move.from)),
  to: squareName(fromOx88(move.to)),
  promotion: move.promotion ?? null,
});

export class ChessState {
  private legal?: readonly RawMove[];

  private constructor(
    readonly fen: string,
    /** Position keys since the last pawn move or capture, current position last. */
    readonly history: readonly string[],
    private readonly position: Board,
  ) {}

  /** The state for `fen`, continuing the repetition history `before`. */
  static create(fen: string, before: readonly string[] = []): ChessState {
    return ChessState.fromBoard(new Board(fen), before);
  }

  private static fromBoard(board: Board, before: readonly string[]): ChessState {
    // chess.js writes an en passant square into a FEN only when the capture is
    // actually legal, so the first four fields are exactly python-chess's notion
    // of "the same position" for repetition: pieces, side to move, castling
    // rights, and en passant when it can really happen.
    const fen = board.fen();
    const key = fen.split(" ").slice(0, 4).join(" ");
    return new ChessState(fen, [...before, key], board);
  }

  /** The chess.js board. Read it, never move on it: states are immutable. */
  board(): Board {
    return this.position;
  }

  /** Every legal move, generated once, as the page names them. */
  moves(): NamedMove[] {
    return this.raw().map(named);
  }

  /** @internal Legal moves in chess.js's own form, generated on first use. */
  raw(): readonly RawMove[] {
    return (this.legal ??= internals(this.position)._moves({ legal: true }));
  }

  /** The state after a legal move in chess.js's own form. */
  after(move: RawMove): ChessState {
    const board = new Board(this.fen);
    internals(board)._makeMove(move);
    // A pawn move or a capture can never be undone, so no earlier position can
    // come round again. (python-chess also resets on lost castling rights; the
    // positions either side of those differ anyway, so the counts agree.)
    const irreversible = move.piece === "p" || move.captured !== undefined;
    return ChessState.fromBoard(board, irreversible ? [] : this.history);
  }

  get whiteToMove(): boolean {
    return this.fen.split(" ")[1] === "w";
  }

  get halfmoveClock(): number {
    return Number(this.fen.split(" ")[4]);
  }

  /** Where a legal en passant capture would land, if there is one. */
  get enPassant(): string | null {
    const field = this.fen.split(" ")[3];
    return field === "-" ? null : field;
  }

  /** How many times the current position has occurred, counting now. */
  get repetitions(): number {
    const current = this.history[this.history.length - 1];
    return this.history.filter((key) => key === current).length;
  }
}

/** The action for a move between two squares, seen from the side making it. */
function indexOf(black: boolean, from: number, to: number, promotion?: string | null): number {
  const [start, end] = black ? [mirror(from), mirror(to)] : [from, to];
  const under = promotion === "n" || promotion === "b" || promotion === "r" ? promotion : null;
  const action = INDEX.get(frameKey(start, end, under));
  if (action === undefined) throw new Error(`no action for ${squareName(from)}${squareName(to)}`);
  return action;
}

/** The action index of a move, in the frame of the player making it. */
export function actionOf(state: ChessState, move: NamedMove): number {
  return indexOf(!state.whiteToMove, squareIndex(move.from), squareIndex(move.to), move.promotion);
}

/** The move an action index names in `state`. The inverse of {@link actionOf}. */
export function moveOf(state: ChessState, action: number): NamedMove {
  const entry = action >= 0 && action < ACTIONS ? MOVES[action] : null;
  if (!entry) throw new Error(`action ${action} names no move that stays on the board`);
  let { from, to } = entry;
  if (!state.whiteToMove) [from, to] = [mirror(from), mirror(to)];
  let promotion: string | null = entry.promotion;
  // A queen-like pawn move to the last rank is how a queen promotion is named.
  const piece = state.board().get(squareName(from) as Square);
  const lastRank = Math.floor(to / 8) === 0 || Math.floor(to / 8) === 7;
  if (promotion === null && piece && piece.type === "p" && lastRank) promotion = "q";
  return { from: squareName(from), to: squareName(to), promotion };
}

export class Chess implements Game<ChessState> {
  readonly name = "chess";
  readonly actionSize = ACTIONS;
  readonly boardShape = [8, 8] as const;
  readonly inputPlanes = PLANES;

  initialState(): ChessState {
    return ChessState.create(DEFAULT_POSITION);
  }

  /** A position from a FEN, as the board editor or a pasted game gives it. */
  positionFrom(fen: string): ChessState {
    return ChessState.create(fen);
  }

  toPlay(state: ChessState): number {
    return state.whiteToMove ? 0 : 1;
  }

  legalActions(state: ChessState): boolean[] {
    // Straight from chess.js's squares: this runs for every position the search
    // expands, so no square names are built on the way.
    const mask = new Array<boolean>(ACTIONS).fill(false);
    const black = !state.whiteToMove;
    for (const move of state.raw()) {
      mask[indexOf(black, fromOx88(move.from), fromOx88(move.to), move.promotion)] = true;
    }
    return mask;
  }

  apply(state: ChessState, action: number): ChessState {
    const { from, to, promotion } = moveOf(state, action);
    const [start, end] = [toOx88(squareIndex(from)), toOx88(squareIndex(to))];
    const legal = state.raw().find((move) =>
      move.from === start && move.to === end && (move.promotion ?? null) === promotion);
    if (!legal) throw new Error(`${from}${to} is not legal in ${state.fen}`);
    return state.after(legal);
  }

  terminalValue(state: ChessState): number | null {
    // Checkmate first: a mate delivered on the hundredth half-move still wins.
    if (state.raw().length === 0) return state.board().inCheck() ? -1 : 0;
    if (state.halfmoveClock >= 100 || state.repetitions >= 3
        || state.board().isInsufficientMaterial()) {
      return 0;
    }
    return null;
  }

  encode(state: ChessState): Float32Array {
    const planes = new Float32Array(PLANES * SQUARES);
    const board = state.board();
    const us = board.turn();
    const black = us === "b";
    // chess.js lists ranks from the eighth down; squares here count rank * 8 + file.
    board.board().forEach((row, r) => row.forEach((piece, file) => {
      if (!piece) return;
      const square = (7 - r) * 8 + file;
      const plane = (piece.color === us ? 0 : 6) + PIECES.indexOf(piece.type);
      planes[plane * SQUARES + (black ? mirror(square) : square)] = 1;
    }));
    const fill = (plane: number, value: number) =>
      planes.fill(value, plane * SQUARES, (plane + 1) * SQUARES);
    const own = board.getCastlingRights(us);
    const theirs = board.getCastlingRights(black ? "w" : "b");
    fill(12, Number(own.k));
    fill(13, Number(own.q));
    fill(14, Number(theirs.k));
    fill(15, Number(theirs.q));
    if (state.enPassant) {
      const square = squareIndex(state.enPassant);
      planes[16 * SQUARES + (black ? mirror(square) : square)] = 1;
    }
    fill(17, Math.min(state.halfmoveClock, 100) / 100);
    fill(18, state.repetitions > 1 ? 1 : 0);
    return planes;
  }
}
