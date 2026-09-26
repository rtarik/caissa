/**
 * The chess board.
 *
 * Squares are numbered as everywhere else in the port - rank * 8 + file, from
 * White's side - and only *drawn* in a different order when the human plays
 * Black, so the near side is always theirs. `data-square` always holds the real
 * square, so turning the board never reaches the rules or the action encoding.
 *
 * A move takes two clicks, a piece and then where it goes - the half-a-move
 * mechanism Isolation introduced. A promotion takes a third, the piece to
 * become, because four different actions share the same two squares.
 */
import { Chess as Board, type Square } from "chess.js";
import {
  ChessState,
  MOVES,
  actionOf,
  moveOf,
  squareIndex,
  squareName,
  type NamedMove,
} from "../games/chess";
import { heatmap } from "./heatmap";
import { PIECE_NAMES, pieceSvg } from "./pieces";
import type { View, ViewContext } from "./views";

/** Pending values from here up are a from/to pair waiting for the promotion piece. */
const PROMOTING = 64;

/** The real square drawn at each display position, with the human's side nearest. */
export function displayOrder(humanWhite: boolean): number[] {
  const order: number[] = [];
  for (let row = 0; row < 8; row++) {
    for (let col = 0; col < 8; col++) {
      order.push(humanWhite ? (7 - row) * 8 + col : row * 8 + (7 - col));
    }
  }
  return order;
}

/** A move's notation, from chess.js's public API: disambiguation, check and mate. */
/** A move in standard algebraic notation, from the position it is played in. */
export function san(state: ChessState, move: NamedMove): string {
  return new Board(state.fen).move({
    from: move.from,
    to: move.to,
    promotion: move.promotion ?? undefined,
  }).san;
}

/**
 * The last move shown, and the position it was played from.
 *
 * Taken from the page's own record of the game rather than replayed here. The
 * replay this replaced started from the usual first position, so once a game
 * could begin from a set-up position it decoded the engine's reply as a move
 * from the wrong board - a black king's step read as a white king's - and threw.
 */
function lastMove(ctx: ViewContext): { before: ChessState | null; last: NamedMove | null } {
  const before = (ctx.previous ?? null) as ChessState | null;
  const action = ctx.moves[ctx.moves.length - 1];
  return { before, last: before && action !== undefined ? moveOf(before, action) : null };
}

/** The square of the king in check, if the player to move is in check. */
function checkedKing(state: ChessState): number | null {
  const board = state.board();
  if (!board.inCheck()) return null;
  for (const row of board.board()) {
    for (const piece of row) {
      if (piece && piece.type === "k" && piece.color === board.turn()) {
        return squareIndex(piece.square);
      }
    }
  }
  return null;
}

export const chessView: View = {
  layout: "chess",

  prompt: (ctx) => ctx.pending !== null && ctx.pending >= PROMOTING
    ? "Choose a piece to promote to."
    : "Now choose where it goes.",

  board(ctx) {
    const state = ctx.state as ChessState;
    const board = state.board();
    const legal = state.moves();
    const promoting = ctx.pending !== null && ctx.pending >= PROMOTING
      ? ctx.pending - PROMOTING : null;
    const selected = promoting !== null ? Math.floor(promoting / 64) : ctx.pending;

    const targets = new Map<number, boolean>(); // square -> whether it is a capture
    if (selected !== null) {
      for (const move of legal) {
        if (squareIndex(move.from) !== selected) continue;
        targets.set(squareIndex(move.to), Boolean(board.get(move.to as Square)));
      }
    }
    const movable = new Set(ctx.locked ? [] : legal.map((move) => squareIndex(move.from)));
    const { last } = lastMove(ctx);
    const lastSquares = new Set(last ? [squareIndex(last.from), squareIndex(last.to)] : []);
    const check = checkedKing(state);

    const cells = displayOrder(ctx.humanFirst).map((square, display) => {
      const name = squareName(square);
      const piece = board.get(name as Square);
      const [file, rank] = [square % 8, Math.floor(square / 8)];
      const classes = ["sq", (file + rank) % 2 === 0 ? "dark" : "light"];
      if (lastSquares.has(square)) classes.push("last");
      if (square === selected) classes.push("selected");
      if (targets.has(square)) classes.push(targets.get(square) ? "capture" : "target");
      if (square === check) classes.push("check");
      if (movable.has(square) || targets.has(square)) classes.push("movable");
      // Coordinates on the near rank and the left-hand file, as on a real board.
      const coords = (display % 8 === 0 ? `<span class="coord rank">${rank + 1}</span>` : "")
        + (display >= 56 ? `<span class="coord file">${name[0]}</span>` : "");
      const label = piece
        ? `${name}, ${piece.color === "w" ? "white" : "black"} ${PIECE_NAMES[piece.type]}`
        : name;
      return `<button class="${classes.join(" ")}" data-square="${square}" aria-label="${label}"
        ${ctx.locked ? "disabled" : ""}>${coords}${piece ? pieceSvg(piece.type, piece.color) : ""}</button>`;
    });

    let overlay = "";
    if (promoting !== null) {
      const from = squareName(Math.floor(promoting / 64));
      const to = squareName(promoting % 64);
      const choices = ["q", "r", "b", "n"].map((piece) => {
        const action = actionOf(state, { from, to, promotion: piece });
        return `<button class="choice" data-action="${action}"
          aria-label="Promote to ${PIECE_NAMES[piece]}">${pieceSvg(piece, board.turn())}</button>`;
      }).join("");
      overlay = `<div class="promotion"><div class="choices">${choices}
        <button class="cancel" data-square="${Math.floor(promoting / 64)}"
          aria-label="Cancel the promotion">×</button></div></div>`;
    }
    return cells.join("") + overlay;
  },

  select(element, ctx) {
    if (element.dataset.action !== undefined) {
      return { kind: "action", action: Number(element.dataset.action) };
    }
    const value = element.dataset.square;
    if (value === undefined || ctx.locked) return null;
    // Anything but a promotion choice while one is being asked for cancels it.
    if (ctx.pending !== null && ctx.pending >= PROMOTING) return { kind: "pending", pending: null };

    const square = Number(value);
    const state = ctx.state as ChessState;
    const legal = state.moves();
    const movable = (at: number) => legal.some((move) => squareIndex(move.from) === at);
    const from = ctx.pending;

    if (from === null) return movable(square) ? { kind: "pending", pending: square } : null;
    if (square === from) return { kind: "pending", pending: null };

    const choices = legal.filter((move) =>
      squareIndex(move.from) === from && squareIndex(move.to) === square);
    if (choices.length > 1) return { kind: "pending", pending: PROMOTING + from * 64 + square };
    if (choices.length === 1) return { kind: "action", action: actionOf(state, choices[0]) };
    // Clicking another of your own pieces switches to it; anywhere else lets go.
    return { kind: "pending", pending: movable(square) ? square : null };
  },

  detail(ctx) {
    const state = ctx.state as ChessState;
    const outcome = ctx.game.terminalValue(state);
    let note = "";
    if (outcome === 0) {
      note = state.moves().length === 0 ? "Stalemate"
        : state.halfmoveClock >= 100 ? "Fifty-move rule"
        : state.repetitions >= 3 ? "Threefold repetition"
        : "Insufficient material";
    } else if (outcome !== null) {
      note = "Checkmate";
    } else if (state.board().inCheck()) {
      note = "Check";
    }
    return note ? `<span class="spacer score">${note}</span>` : "";
  },

  visits(counts, ctx) {
    // The engine searched the position before its move, so its actions are in
    // that position's frame - the engine's colour.
    const { before } = lastMove(ctx);
    const engineWhite = before ? before.whiteToMove : !ctx.humanFirst;
    const toSquare = new Array<number>(64).fill(0);
    counts.forEach((visits, action) => {
      const move = visits > 0 ? MOVES[action] : null;
      if (move) toSquare[engineWhite ? move.to : move.to ^ 56] += visits;
    });
    const map = heatmap(displayOrder(ctx.humanFirst).map((square) => toSquare[square]), 64, 8);

    if (!before) return map;
    const total = counts.reduce((a, b) => a + b, 0);
    const best = counts
      .map((visits, action) => [visits, action] as const)
      .filter(([visits]) => visits > 0)
      .sort((a, b) => b[0] - a[0])
      .slice(0, 3)
      .map(([visits, action]) =>
        `${san(before, moveOf(before, action))} ${Math.round((100 * visits) / total)}%`);
    return `${map}<div class="meta candidates"><span>${best.join(" · ")}</span></div>`;
  },

};
