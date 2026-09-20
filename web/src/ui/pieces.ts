/**
 * The chess pieces, drawn for this site.
 *
 * Not a font's glyphs - the chess symbols in Unicode render differently on every
 * device, and some phones swap them for emoji - and not a borrowed set: every
 * visible part of the chess page is this project's own. Simple geometric
 * silhouettes on a 100-unit grid, filled and outlined from CSS, so one set of
 * shapes serves both colours and both themes. The bishop's cut and the knight's
 * eye are holes (even-odd fill), so the square shows through them.
 */

const BASE = `<rect x="25" y="79" width="50" height="10" rx="4"/>`;

/** Exported so the site's icons can borrow a piece rather than redraw one. */
export const SHAPES: Record<string, string> = {
  p: `<circle cx="50" cy="29" r="12"/>
      <path d="M41 42 H59 L57 48 C57 60 65 70 70 79 H30 C35 70 43 60 43 48 Z"/>${BASE}`,
  r: `<path d="M29 16 H39 V23 H46 V16 H54 V23 H61 V16 H71 V33 H29 Z"/>
      <path d="M34 33 H66 L63 69 H37 Z"/>
      <rect x="31" y="69" width="38" height="8" rx="2.5"/>${BASE}`,
  n: `<path fill-rule="evenodd" d="M64 79 C61 66 68 58 70 46 C72 32 64 20 52 16 L49 7 L43 16
        C33 20 25 31 23 43 L19 53 C18 59 24 63 30 60 L40 54 C44 57 45 62 39 70 L35 79 Z
        M40.5 31 a3.5 3.5 0 1 0 7 0 a3.5 3.5 0 1 0 -7 0 Z"/>
      <path class="detail" d="M57 22 C63 30 65 41 60 52"/>${BASE}`,
  b: `<circle cx="50" cy="14" r="5"/>
      <path fill-rule="evenodd" d="M50 20 C36 30 32 46 38 58 H62 C68 46 64 30 50 20 Z
        M53 29 L57 32 L48 45 L44 42 Z"/>
      <rect x="36" y="58" width="28" height="7" rx="3"/>
      <path d="M41 65 H59 C59 71 65 75 70 79 H30 C35 75 41 71 41 65 Z"/>${BASE}`,
  q: `<circle cx="24" cy="26" r="5"/><circle cx="37" cy="17" r="5"/><circle cx="50" cy="13" r="5"/>
      <circle cx="63" cy="17" r="5"/><circle cx="76" cy="26" r="5"/>
      <path d="M24 30 L33 58 H67 L76 30 L64 48 L63 22 L55 45 L50 18 L45 45 L37 22 L36 48 Z"/>
      <rect x="31" y="58" width="38" height="7" rx="3"/>
      <path d="M36 65 H64 C64 71 68 75 72 79 H28 C32 75 36 71 36 65 Z"/>${BASE}`,
  k: `<path d="M46 5 H54 V12 H61 V19 H54 V27 H46 V19 H39 V12 H46 Z"/>
      <path d="M50 31 C44 22 26 22 26 36 C26 46 34 52 36 58 H64 C66 52 74 46 74 36 C74 22 56 22 50 31 Z"/>
      <rect x="33" y="58" width="34" height="7" rx="3"/>
      <path d="M38 65 H62 C62 71 67 75 71 79 H29 C33 75 38 71 38 65 Z"/>${BASE}`,
};

export const PIECE_NAMES: Record<string, string> = {
  p: "pawn", n: "knight", b: "bishop", r: "rook", q: "queen", k: "king",
};

/** A piece as inline SVG; `type` and `color` as chess.js names them. */
export function pieceSvg(type: string, color: string): string {
  return `<svg class="chess-piece ${color === "w" ? "white" : "black"}" viewBox="0 0 100 100"
    aria-hidden="true">${SHAPES[type]}</svg>`;
}
