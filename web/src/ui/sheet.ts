/**
 * The new-game sheet: who you are playing, and as which side.
 *
 * It replaces a settings form that sat beside the board and asked the same
 * questions with drop-downs. Choices are shown rather than hidden in menus:
 * four level cards, and a three-way switch for your side. They are plain radio
 * buttons underneath, so the keyboard, screen readers and the browser's own form
 * handling all work without a line of widget code.
 *
 * Rendering is a pure function of the options, which is what lets the tests
 * check what the sheet offers without a browser.
 */
import type { LadderEntry } from "../games/ladder";
import { parseFen } from "../editor";
import { LEVELS } from "../levels";
import { miniBoardHtml } from "./editorview";
import { iconFor } from "./icons";

export type SeatChoice = "first" | "random" | "second";

export interface Settings {
  /** Index into LEVELS. */
  level: number;
  seat: SeatChoice;
  /** The network file to play, when a game has more than one. */
  network: string | null;
}

/** A measured rating for one level, where there is one (chess, for now). */
export interface Rating {
  rating: number;
  low: number;
  high: number;
}

export interface SheetOptions {
  entry: LadderEntry;
  settings: Settings;
  networks: { file: string; label: string }[];
  /**
   * Measured ratings, per network and then per level. A rating belongs to the
   * network it was measured on: shown against another it is simply wrong, which
   * is how the first version had the untrained network claiming 2300.
   */
  ratings?: Map<string, Map<string, Rating>>;
  /** Whether a game is under way, so that closing the sheet has somewhere to go back to. */
  cancellable: boolean;
  /**
   * For games that can start from a set-up position: the one chosen, or null for
   * the usual start. Absent for games without a board editor.
   */
  start?: { fen: string | null };
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

export function sheetHtml(options: SheetOptions): string {
  const { entry, settings, networks } = options;
  const ratings = options.ratings?.get(settings.network ?? "");
  const [first, second] = entry.seats ?? ["First", "Second"];
  const seats: [SeatChoice, string][] = [["first", first], ["random", "Random"], ["second", second]];

  const levels = LEVELS.map((level, index) => {
    const rating = ratings?.get(level.label);
    return `<label class="level-card">
      <input type="radio" name="level" value="${index}"${checked(index === settings.level)}>
      <span class="level-name">${level.label}</span>
      ${rating ? `<span class="level-rating">about ${approximately(rating.rating)}</span>` : ""}
      <span class="level-note">${level.note}</span>
    </label>`;
  }).join("");

  const network = networks.length < 2 ? "" : `
    <fieldset class="choice-group">
      <legend>Engine</legend>
      <div class="segmented">
        ${networks.map((n) => `<label>
          <input type="radio" name="network" value="${n.file}"${checked(n.file === settings.network)}>
          <span>${n.label}</span>
        </label>`).join("")}
      </div>
    </fieldset>`;

  return `<form class="sheet" id="sheet-form" role="dialog" aria-modal="true" aria-labelledby="sheet-title">
    <header class="sheet-head">
      <span class="sheet-icon" data-game="${entry.key}">${iconFor(entry.key)}</span>
      <div>
        <h2 id="sheet-title">${entry.title}</h2>
        <p class="sheet-rule">${entry.howToWin}</p>
      </div>
    </header>

    <fieldset class="choice-group">
      <legend>Opponent</legend>
      <div class="level-cards">${levels}</div>
      ${ratings?.size
        ? `<p class="choice-note">Ratings are rough estimates from games against Stockfish.</p>`
        : ""}
    </fieldset>

    <fieldset class="choice-group">
      <legend>You play</legend>
      <div class="segmented">
        ${seats.map(([value, label]) => `<label>
          <input type="radio" name="seat" value="${value}"${checked(settings.seat === value)}>
          <span>${label}</span>
        </label>`).join("")}
      </div>
    </fieldset>
    ${network}
    ${options.start ? startHtml(options.start.fen, settings.seat !== "second") : ""}

    <div class="sheet-actions">
      ${options.cancellable ? `<button type="button" class="button ghost" data-sheet="cancel">Cancel</button>` : ""}
      <button type="submit" class="button primary">Play</button>
    </div>
  </form>`;
}

function startHtml(fen: string | null, whiteAtBottom: boolean): string {
  const setup = fen ? parseFen(fen) : null;
  return `<fieldset class="choice-group">
    <legend>Start from</legend>
    <div class="start-row">
      ${setup ? miniBoardHtml(setup, whiteAtBottom) : ""}
      <div class="start-text">
        <span class="start-what">${setup ? "Your position" : "The usual starting position"}</span>
        <div class="start-buttons">
          <button type="button" class="button small" data-sheet="setup">${setup ? "Edit position" : "Set up a position"}</button>
          ${setup ? `<button type="button" class="button small ghost" data-sheet="standard">Use the usual start</button>` : ""}
        </div>
      </div>
    </div>
  </fieldset>`;
}

/** The choices in a submitted sheet, falling back to what was there before. */
export function readSettings(data: FormData, before: Settings): Settings {
  const level = Number(data.get("level"));
  const seat = data.get("seat");
  const network = data.get("network");
  return {
    level: Number.isInteger(level) && level >= 0 && level < LEVELS.length ? level : before.level,
    seat: seat === "first" || seat === "second" || seat === "random" ? seat : before.seat,
    network: typeof network === "string" ? network : before.network,
  };
}

/** Random is decided once, when the game starts, and then it is simply a side. */
export function resolveSeat(seat: SeatChoice, random: () => number = Math.random): boolean {
  if (seat === "random") return random() < 0.5;
  return seat === "first";
}
