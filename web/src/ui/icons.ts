/**
 * A small drawing of each game, for the gallery cards and the play header.
 *
 * Six games listed as words alone read as a list of files. A glance at a shape
 * says which is which, and for people who do not know a game by name - Isolation
 * and Gomoku are not household words - the picture is the only clue they get
 * before clicking.
 *
 * Deliberately plain: one 24x24 grid, two tones taken from the theme
 * (`currentColor` and the game's accent), no detail that would disappear at the
 * 22px the tabs draw them at.
 */

import { SHAPES } from "./pieces";

/** Both tones come from CSS, so an icon suits whatever palette it sits in. */
const SVG = (body: string) =>
  `<svg class="icon" viewBox="0 0 24 24" aria-hidden="true" focusable="false">${body}</svg>`;

const disc = (cx: number, cy: number, r: number, cls: string) =>
  `<circle cx="${cx}" cy="${cy}" r="${r}" class="${cls}" />`;

export const ICONS: Record<string, string> = {
  // A rack with two discs dropped into it, one of each colour.
  connect4: SVG(`
    <rect x="2.5" y="4.5" width="19" height="15" rx="2.5" class="frame" />
    ${disc(7, 9, 2, "hole")}${disc(12, 9, 2, "hole")}${disc(17, 9, 2, "hole")}
    ${disc(7, 15, 2, "one")}${disc(12, 15, 2, "hole")}${disc(17, 15, 2, "two")}
  `),

  // One disc caught mid-flip: the whole game in a single shape.
  reversi: SVG(`
    <circle cx="12" cy="12" r="8.5" class="frame" />
    <path d="M12 3.5a8.5 8.5 0 0 0 0 17z" class="one" />
    <path d="M12 3.5a8.5 8.5 0 0 1 0 17z" class="two" />
  `),

  // Stones on the intersections of a grid, as they are actually played.
  gomoku: SVG(`
    <path d="M6 3.5v17M12 3.5v17M18 3.5v17M3.5 6h17M3.5 12h17M3.5 18h17" class="grid" />
    ${disc(12, 12, 3.1, "one")}${disc(6, 6, 3.1, "two")}${disc(18, 18, 3.1, "one")}
  `),

  // A piece, and the square next to it already gone.
  isola: SVG(`
    <rect x="2.5" y="2.5" width="8.5" height="8.5" rx="1.6" class="frame" />
    <rect x="13" y="13" width="8.5" height="8.5" rx="1.6" class="frame gone" />
    <rect x="13" y="2.5" width="8.5" height="8.5" rx="1.6" class="frame" />
    ${disc(6.75, 6.75, 2.6, "one")}
    <rect x="2.5" y="13" width="8.5" height="8.5" rx="1.6" class="frame" />
    ${disc(17.25, 6.75, 2.6, "two")}
  `),

  // Four dots, three sides drawn, one box already claimed.
  dotsandboxes: SVG(`
    <rect x="4" y="4" width="7.5" height="7.5" rx="1" class="one filled" />
    <path d="M12.5 4.5h7M12.5 12.5h7M20 5v7" class="grid" />
    ${disc(4, 4, 1.7, "dot")}${disc(12, 4, 1.7, "dot")}${disc(20, 4, 1.7, "dot")}
    ${disc(4, 12, 1.7, "dot")}${disc(12, 12, 1.7, "dot")}${disc(20, 12, 1.7, "dot")}
    ${disc(4, 20, 1.7, "dot")}${disc(12, 20, 1.7, "dot")}${disc(20, 20, 1.7, "dot")}
  `),

  // The knight from the board itself, on its own 100-unit grid. Drawing a
  // second one would be a second thing to keep in step with the first.
  chess: `<svg class="icon knight" viewBox="4 2 92 92" aria-hidden="true"
    focusable="false"><g class="one filled">${SHAPES.n}</g></svg>`,
};

/** A game's icon, or a neutral placeholder for one without a drawing yet. */
export function iconFor(key: string): string {
  return ICONS[key] ?? SVG(`<circle cx="12" cy="12" r="8.5" class="frame" />`);
}
