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

/** The piece the palette places, or null for the eraser. */
export type Tool = Piece | null;

const colourOf = (piece: Piece) => (piece === piece.toUpperCase() ? "w" : "b");
const nameOf = (piece: Piece) =>
  `${colourOf(piece) === "w" ? "white" : "black"} ${PIECE_NAMES[piece.toLowerCase()]}`;

export function editorBoardHtml(setup: Setup, whiteAtBottom: boolean): string {
  return displayOrder(whiteAtBottom).map((square, display) => {
    const name = squareName(square);
    const piece = setup.squares[square];
    const [file, rank] = [square % 8, Math.floor(square / 8)];
    const classes = ["sq", "movable", (file + rank) % 2 === 0 ? "dark" : "light"];
    const coords = (display % 8 === 0 ? `<span class="coord rank">${rank + 1}</span>` : "")
      + (display >= 56 ? `<span class="coord file">${name[0]}</span>` : "");
    const label = piece ? `${name}, ${nameOf(piece)}` : `${name}, empty`;
    return `<button class="${classes.join(" ")}" data-square="${square}" aria-label="${label}">${coords}${
      piece ? pieceSvg(piece.toLowerCase(), colourOf(piece)) : ""}</button>`;
  }).join("");
}

export function paletteHtml(tool: Tool): string {
  const choice = (piece: Piece | null, label: string, body: string) => `<label class="tool" title="${label}">
    <input type="radio" name="tool" value="${piece ?? ""}"${tool === piece ? " checked" : ""} aria-label="${label}">
    <span>${body}</span>
  </label>`;
  const row = (pieces: Piece[]) => pieces
    .map((piece) => choice(piece, nameOf(piece), pieceSvg(piece.toLowerCase(), colourOf(piece))))
    .join("");
  const eraser = `<svg class="eraser" viewBox="0 0 24 24" aria-hidden="true">
    <path d="M7 21h10M5 15l7-7 5 5-7 7H8z" /><path d="M12 8l3-3 5 5-3 3" /></svg>`;
  return `<div class="palette" role="radiogroup" aria-label="Piece to place">
    ${row(WHITE_PIECES)}${row(BLACK_PIECES)}${choice(null, "Remove a piece", eraser)}
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

/** A small, still picture of a position, for the new-game sheet. */
export function miniBoardHtml(setup: Setup, whiteAtBottom: boolean): string {
  const cells = displayOrder(whiteAtBottom).map((square) => {
    const piece = setup.squares[square];
    const dark = (square % 8 + Math.floor(square / 8)) % 2 === 0;
    return `<span class="mini-sq ${dark ? "dark" : "light"}">${
      piece ? pieceSvg(piece.toLowerCase(), colourOf(piece)) : ""}</span>`;
  }).join("");
  return `<div class="mini-board" aria-hidden="true">${cells}</div>`;
}
