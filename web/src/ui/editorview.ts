/**
 * The board editor, drawn: the board, a palette of pieces, who moves, castling,
 * the FEN, and what is wrong with the setup so far.
 *
 * Drawn in separate parts rather than one block of markup, because the FEN box
 * has to survive being typed into: redrawing it on every keystroke would throw
 * away the cursor and the half-typed text. The page redraws the other parts and
 * leaves the input alone.
 *
 * The board is the play board's own markup - same squares, same pieces, same
 * coordinates - so a position looks the same being set up as being played.
 */
import { BLACK_PIECES, WHITE_PIECES, possibleCastling, type Piece, type Setup } from "../editor";
import { squareName } from "../games/chess";
import { displayOrder } from "./chessview";
import { PIECE_NAMES, pieceSvg } from "./pieces";

/** The hand, which moves pieces; a piece the palette places; or null for the eraser. */
export type Tool = Piece | null | "hand";

const colourOf = (piece: Piece) => (piece === piece.toUpperCase() ? "w" : "b");
const nameOf = (piece: Piece) =>
  `${colourOf(piece) === "w" ? "white" : "black"} ${PIECE_NAMES[piece.toLowerCase()]}`;

/** `lifted` is the square whose piece the hand holds, between a click on it and a click on where it goes. */
export function editorBoardHtml(setup: Setup, whiteAtBottom: boolean, lifted: number | null = null): string {
  return displayOrder(whiteAtBottom).map((square, display) => {
    const name = squareName(square);
    const piece = setup.squares[square];
    const [file, rank] = [square % 8, Math.floor(square / 8)];
    const classes = ["sq", "movable", (file + rank) % 2 === 0 ? "dark" : "light"];
    if (square === lifted) classes.push("lifted");
    const coords = (display % 8 === 0 ? `<span class="coord rank">${rank + 1}</span>` : "")
      + (display >= 56 ? `<span class="coord file">${name[0]}</span>` : "");
    const label = piece ? `${name}, ${nameOf(piece)}` : `${name}, empty`;
    return `<button class="${classes.join(" ")}" data-square="${square}" aria-label="${label}">${coords}${
      piece ? pieceSvg(piece.toLowerCase(), colourOf(piece)) : ""}</button>`;
  }).join("");
}

export function paletteHtml(tool: Tool): string {
  const choice = (piece: Tool, label: string, body: string, wide = false) => `<label class="tool${
    wide ? " wide" : ""}" title="${label}">
    <input type="radio" name="tool" value="${piece ?? ""}"${tool === piece ? " checked" : ""} aria-label="${label}">
    <span>${body}</span>
  </label>`;
  const row = (pieces: Piece[]) => pieces
    .map((piece) => choice(piece, nameOf(piece), pieceSvg(piece.toLowerCase(), colourOf(piece))))
    .join("");
  const eraser = `<svg class="tool-icon" viewBox="0 0 24 24" aria-hidden="true">
    <path d="M7 21h10M5 15l7-7 5 5-7 7H8z" /><path d="M12 8l3-3 5 5-3 3" /></svg>`;
  const hand = `<svg class="tool-icon" viewBox="0 0 24 24" aria-hidden="true">
    <path d="M8 13V5.5a1.5 1.5 0 0 1 3 0V12m0-6.5v-1a1.5 1.5 0 0 1 3 0V12m0-6.5a1.5 1.5 0 0 1 3 0V12m0-4.5a1.5 1.5 0 0 1 3 0v6.5a7 7 0 0 1-7 7h-1.2a6 6 0 0 1-4.9-2.6L3.6 14.2a1.6 1.6 0 0 1 2.5-2L8 14" /></svg>`;
  return `<div class="palette" role="radiogroup" aria-label="What a click on the board does">
    ${row(WHITE_PIECES)}${row(BLACK_PIECES)}
    ${choice("hand", "Move pieces", `${hand}Move`, true)}${choice(null, "Remove a piece", `${eraser}Remove`, true)}
  </div>`;
}

/** Who moves, and castling: only the rights the pieces allow can be switched on. */
export function rulesHtml(setup: Setup): string {
  const possible = possibleCastling(setup.squares);
  const right = (key: "K" | "Q" | "k" | "q", label: string) => `<label class="check${possible[key] ? "" : " unavailable"}">
    <input type="checkbox" name="castle" value="${key}"${setup.castling[key] ? " checked" : ""}${possible[key] ? "" : " disabled"}>
    <span>${label}</span>
  </label>`;
  return `
    <fieldset class="choice-group">
      <legend>To move</legend>
      <div class="segmented">
        <label><input type="radio" name="tomove" value="w"${setup.whiteToMove ? " checked" : ""}><span>White</span></label>
        <label><input type="radio" name="tomove" value="b"${setup.whiteToMove ? "" : " checked"}><span>Black</span></label>
      </div>
    </fieldset>
    <fieldset class="choice-group">
      <legend>Castling still allowed</legend>
      <div class="castling">
        ${right("K", "White O-O")}${right("Q", "White O-O-O")}
        ${right("k", "Black O-O")}${right("q", "Black O-O-O")}
      </div>
    </fieldset>`;
}

export function problemsHtml(found: string[]): string {
  if (!found.length) return `<p class="setup-ok">Ready to play.</p>`;
  return `<ul class="setup-problems">${found.map((problem) => `<li>${problem}</li>`).join("")}</ul>`;
}
