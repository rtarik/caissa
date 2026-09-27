/**
 * The choices before a game: Caissa's level, and your side.
 *
 * They used to live in a sheet over the board, which asked every question at
 * once and hid the board while it did. Now each sits on the card it describes -
 * the level on Caissa's, your side on yours - and only until the game begins:
 * your first move, or Start when Caissa is to open, folds them away, and they
 * come back with the next game. Nothing covers the board, and on a phone both
 * are right beside it rather than a scroll away.
 *
 * They are plain radio buttons underneath, so the keyboard, screen readers and
 * the browser's own form handling all work without a line of widget code; and
 * rendering is a pure function of the choices, so the tests can check what is
 * offered without a browser.
 */
import type { LadderEntry } from "../games/ladder";
import { LEVELS, type Level } from "../levels";

export type SeatChoice = "first" | "random" | "second";

export interface Settings {
  /** Index into LEVELS. */
  level: number;
  seat: SeatChoice;
}

/** A measured rating for one level, where there is one (chess, for now). */
export interface Rating {
  rating: number;
  low: number;
  high: number;
  /**
   * Set when the games only say the level is weaker than this: it lost (nearly)
   * everything to the weakest reference, so there is no rating to show, only a
   * ceiling. `above` is the same from the other side.
   */
  below?: number;
  above?: number;
}

const checked = (on: boolean) => (on ? " checked" : "");

/**
 * A rating as it should be shown: to the nearest fifty.
 *
 * The measurements carry a 95% range of roughly eighty points either way, so a
 * figure like 2288 claims three digits the data does not have.
 */
export function approximately(rating: number): number {
  return Math.round(rating / 50) * 50;
}

/**
 * A rating in words: "about 2300", or the bound when a bound is all there is.
 *
 * Bounds round outwards - a ceiling up, a floor down - so that rounding never
 * claims more than the games showed.
 */
export function ratingText(rating: Rating): string {
  if (rating.below !== undefined) return `under ${Math.ceil(rating.below / 50) * 50}`;
  if (rating.above !== undefined) return `over ${Math.floor(rating.above / 50) * 50}`;
  return `about ${approximately(rating.rating)}`;
}

/** The figure a rating field such as a PGN header may carry: a rating, never a bound. */
export function measuredElo(rating: Rating | undefined): number | undefined {
  if (!rating || rating.below !== undefined || rating.above !== undefined) return undefined;
  return rating.rating;
}

/** A row of radio buttons drawn as one switch, with a name for screen readers. */
function picker(name: string, legend: string, options: [string, string][], chosen: string): string {
  return `<fieldset class="picker">
    <legend class="visually-hidden">${legend}</legend>
    <div class="segmented">
      ${options.map(([value, label]) => `<label>
        <input type="radio" name="${name}" value="${value}"${checked(value === chosen)}>
        <span>${label}</span>
      </label>`).join("")}
    </div>
  </fieldset>`;
}

/** Caissa's four levels, as a switch on its card. */
export function levelPickerHtml(level: number): string {
  return picker("level", "Caissa's level", LEVELS.map((l, index) => [String(index), l.label]), String(level));
}

/** Your side, as a switch on your card, named the way the game names its sides. */
export function seatPickerHtml(entry: LadderEntry, seat: SeatChoice): string {
  const [first, second] = entry.seats ?? ["First", "Second"];
  return picker("seat", "Your side", [["first", first], ["random", "Random"], ["second", second]], seat);
}

/**
 * What Caissa's card says about its level: the rating where one was measured,
 * and - while you are still choosing - what the level is like to play against.
 */
export function levelDetail(level: Level, rating: Rating | undefined, choosing: boolean): string {
  const rated = rating ? `Rated ${ratingText(rating)}` : "";
  if (!choosing) return rated || level.note;
  return rated ? `${rated} · ${level.note}` : level.note;
}

/** One choice changed on a card: the new settings, or the old ones for anything unexpected. */
export function chooseSetting(before: Settings, name: string, value: string): Settings {
  if (name === "level") {
    // Digits only: Number("") is 0, and an empty value must not quietly mean Beginner.
    const level = /^\d+$/.test(value) ? Number(value) : -1;
    return level >= 0 && level < LEVELS.length ? { ...before, level } : before;
  }
  if (name === "seat" && (value === "first" || value === "random" || value === "second")) {
    return { ...before, seat: value };
  }
  return before;
}

/** Random is decided once, when the game starts, and then it is simply a side. */
export function resolveSeat(seat: SeatChoice, random: () => number = Math.random): boolean {
  if (seat === "random") return random() < 0.5;
  return seat === "first";
}

/**
 * Whether the game waits for Start rather than for your first move.
 *
 * Your move starts the game only when the first move is yours. Random has not
 * picked a side yet, and Caissa opening should not happen before you are ready -
 * both wait for the button.
 */
export function waitsForStart(seat: SeatChoice, youMoveFirst: boolean): boolean {
  return seat === "random" || !youMoveFirst;
}

/** What starts the game from here, in a sentence. */
export function startHint(seat: SeatChoice, youMoveFirst: boolean): string {
  if (seat === "random") return "Press Start, and a coin decides who moves first.";
  return youMoveFirst ? "Your first move starts the game." : "Press Start, and Caissa makes the first move.";
}

export const START_BUTTON = `<button type="button" class="button primary small" data-start>Start</button>`;

export interface NewGamePanel {
  entry: LadderEntry;
  settings: Settings;
  /** Whether the first move is yours, with the side you have picked. */
  youMoveFirst: boolean;
  /** For games that can start from a set-up position: the one chosen, or null for the usual. */
  start?: { fen: string | null };
  /** Whether the levels carry measured ratings, which then need a word on what they are. */
  rated: boolean;
  /** How many games the history holds, for games that keep one. */
  saved?: number;
}

/**
 * The side panel before a game: what winning means, where the choices are and
 * what starts it - the parts of the old sheet that belong to no card.
 */
export function newGamePanelHtml(panel: NewGamePanel): string {
  const games = panel.saved === undefined ? "" : `<a class="quiet-link" href="#/games/${panel.entry.key}">${
    panel.saved ? `Your games (${panel.saved})` : "Your games"}</a>`;
  return `<h2 class="new-game-title">New game</h2>
    <p class="new-game-rule">${panel.entry.howToWin}</p>
    <p class="new-game-hint">Choose Caissa's level on its card and your side on yours.
      ${startHint(panel.settings.seat, panel.youMoveFirst)}</p>
    ${panel.start ? startHtml(panel.start.fen) : ""}
    ${panel.rated ? `<p class="choice-note">Ratings are rough estimates from games against Stockfish.</p>` : ""}
    ${games}`;
}

/** Where a chess game will begin, and the way to change it; the board itself shows the position. */
function startHtml(fen: string | null): string {
  return `<div class="start-row">
    <span class="start-what">${fen ? "From your own position" : "From the usual starting position"}</span>
    <div class="start-buttons">
      <button type="button" class="button small" data-setup="edit">${fen ? "Edit position" : "Set up a position"}</button>
      ${fen ? `<button type="button" class="button small ghost" data-setup="standard">Use the usual start</button>` : ""}
    </div>
  </div>`;
}
