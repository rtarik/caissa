/**
 * A chess game written out as PGN, for opening in any analysis tool.
 *
 * chess.js does the writing: it knows the format's rules - tag order, move
 * numbers, line lengths, the SetUp and FEN tags a game from a set-up position
 * needs - and a hand-rolled writer would only be a second place for them to be
 * wrong. What this module adds is what chess.js cannot know: who played, at
 * what level, how the game ended and why.
 */
import { Chess as Board, DEFAULT_POSITION } from "chess.js";
import { Chess, ChessState, moveOf } from "./games/chess";
import type { GameRecord, Result } from "./archive";

export interface Outcome {
  result: Result;
  /** Why it ended, in words, or undefined while it goes on. */
  termination?: string;
}

/**
 * The result of a chess position, and the reason, read the way the rules read it.
 *
 * The conditions are the ones `Chess.terminalValue` checks, in the same order,
 * so the PGN can never call a game drawn by repetition that the board thought
 * was still going. A resignation ends the game whatever the position says.
 */
export function chessOutcome(state: ChessState, humanWhite: boolean, resigned: boolean): Outcome {
  if (resigned) return { result: humanWhite ? "0-1" : "1-0", termination: "resignation" };

  const white = state.whiteToMove;
  if (state.raw().length === 0) {
    return state.board().inCheck()
      ? { result: white ? "0-1" : "1-0", termination: "checkmate" }
      : { result: "1/2-1/2", termination: "stalemate" };
  }
  if (state.halfmoveClock >= 100) return { result: "1/2-1/2", termination: "fifty-move rule" };
  if (state.repetitions >= 3) return { result: "1/2-1/2", termination: "threefold repetition" };
  if (state.board().isInsufficientMaterial()) {
    return { result: "1/2-1/2", termination: "insufficient material" };
  }
  return { result: "*" };
}

/** "2026.09.26", the PGN form of a date. */
function pgnDate(iso: string): string {
  const date = new Date(iso);
  if (Number.isNaN(date.getTime())) return "????.??.??";
  const pad = (n: number) => String(n).padStart(2, "0");
  return `${date.getFullYear()}.${pad(date.getMonth() + 1)}.${pad(date.getDate())}`;
}

/** How the engine appears in a game's record: its level, and its engine when not the usual one. */
export function engineName(record: GameRecord): string {
  return `Caissa (${record.level}, ${record.networkLabel})`;
}

/** The final comment: what an analysis tool shows under the last move. */
function closing(record: GameRecord): string | null {
  if (!record.termination) return null;
  if (record.termination === "resignation") {
    return `${record.humanWhite ? "White" : "Black"} resigns.`;
  }
  return record.termination.charAt(0).toUpperCase() + record.termination.slice(1) + ".";
}

/** Replay a record onto a chess.js board, which then knows every move's notation. */
function replay(record: GameRecord): Board {
  const rules = new Chess();
  const start = record.start ?? DEFAULT_POSITION;
  let state = ChessState.create(start);
  const board = new Board(start);
  for (const action of record.moves) {
    const move = moveOf(state, action);
    board.move({ from: move.from, to: move.to, promotion: move.promotion ?? undefined });
    state = rules.apply(state, action);
  }
  return board;
}

/** The moves as a score sheet reads them: "1. e4 c5 2. Nf3". */
export function movesText(record: GameRecord): string {
  // chess.js writes its tag block whatever is set, so the moves are what follows
  // the blank line after it. Its numbering is kept rather than redone: it is
  // right for a game that starts with Black to move, which counting is not.
  const text = replay(record).pgn();
  const movetext = text.slice(text.lastIndexOf("\n\n") + 2);
  return movetext.replace(/\s*(\*|1-0|0-1|1\/2-1\/2)\s*$/, "").trim();
}

export function recordPgn(record: GameRecord): string {
  const board = replay(record);

  const you = "You";
  const engine = engineName(record);
  board.setHeader("Event", "Game against Caissa");
  board.setHeader("Site", "Caissa");
  board.setHeader("Date", pgnDate(record.started));
  board.setHeader("Round", "-");
  board.setHeader("White", record.humanWhite ? you : engine);
  board.setHeader("Black", record.humanWhite ? engine : you);
  board.setHeader("Result", record.result);
  if (record.rating !== undefined) {
    // The engine's side only: a person's rating is not ours to guess at.
    board.setHeader(record.humanWhite ? "BlackElo" : "WhiteElo", String(record.rating));
  }
  board.setHeader("TimeControl", "-");
  board.setHeader("Termination", record.result === "*" ? "unterminated" : "normal");
  const comment = closing(record);
  if (comment) board.setComment(comment);
  return board.pgn({ maxWidth: 80 });
}

/** Several games as one file, the way PGN collections are written: a blank line between each. */
export function collectionPgn(records: GameRecord[]): string {
  return records.map(recordPgn).join("\n\n") + "\n";
}

/** A file name that sorts by date and says what the game was. */
export function pgnFileName(record: GameRecord): string {
  const day = pgnDate(record.started).replaceAll(".", "-");
  const side = record.humanWhite ? "white" : "black";
  return `caissa-${day}-${side}-vs-${record.level.toLowerCase()}.pgn`;
}
