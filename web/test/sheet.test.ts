/**
 * The new-game sheet and the player cards.
 *
 * Both are pure renders, which is the point: what the sheet offers and what a
 * card says can be checked without a browser. The failures worth catching are
 * the quiet ones - a sheet that forgets the level you chose last time, a Random
 * that is not random, a card that calls the wrong side's turn.
 */
import { describe, expect, it } from "vitest";
import { LADDER } from "../src/games/ladder";
import { DEFAULT_LEVEL, LEVELS } from "../src/levels";
import { playerCardHtml, tokenFor } from "../src/ui/players";
import { approximately, readSettings, resolveSeat, sheetHtml, type Settings } from "../src/ui/sheet";

const chess = LADDER.find((item) => item.key === "chess")!;
const gomoku = LADDER.find((item) => item.key === "gomoku")!;
const defaults: Settings = { level: DEFAULT_LEVEL, seat: "first", network: null };

/** The value of the checked radio in each named group. */
function checkedValues(html: string): Record<string, string> {
  const found: Record<string, string> = {};
  for (const match of html.matchAll(/<input type="radio" name="(\w+)" value="([^"]+)" checked>/g)) {
    found[match[1]] = match[2];
  }
  return found;
}

describe("the new-game sheet", () => {
  it("offers every level, with the strongest chosen to start with", () => {
    const html = sheetHtml({ entry: gomoku, settings: defaults, networks: [], cancellable: false });
    for (const level of LEVELS) expect(html).toContain(level.label);
    expect(checkedValues(html).level).toBe(String(LEVELS.length - 1));
  });

  it("reopens on the choices made last time", () => {
    const html = sheetHtml({
      entry: gomoku,
      settings: { level: 1, seat: "random", network: null },
      networks: [],
      cancellable: true,
    });
    expect(checkedValues(html)).toMatchObject({ level: "1", seat: "random" });
  });

  it("names the sides the way the game does", () => {
    const html = sheetHtml({ entry: chess, settings: defaults, networks: [], cancellable: false });
    expect(html).toContain(">White<");
    expect(html).toContain(">Black<");
    const plain = sheetHtml({ entry: gomoku, settings: defaults, networks: [], cancellable: false });
    expect(plain).toContain(">First<");
    expect(plain).toContain(">Second<");
  });

  it("offers a choice of engine only when there is one to make", () => {
    const one = sheetHtml({
      entry: chess, settings: defaults, networks: [{ file: "chess", label: "Only" }], cancellable: false,
    });
    expect(one).not.toContain('name="network"');
    const two = sheetHtml({
      entry: chess,
      settings: { ...defaults, network: "b" },
      networks: [{ file: "a", label: "A" }, { file: "b", label: "B" }],
      cancellable: false,
    });
    expect(checkedValues(two).network).toBe("b");
  });

  it("shows ratings when they have been measured, and says what they are", () => {
    const ratings = new Map([["imitation", new Map([["Master", { rating: 2100, low: 2000, high: 2200 }]])]]);
    const settings = { ...defaults, network: "imitation" };
    const html = sheetHtml({ entry: chess, settings, networks: [], ratings, cancellable: false });
    expect(html).toContain("about 2100");
    expect(html).toContain("Stockfish");
    const without = sheetHtml({ entry: chess, settings, networks: [], cancellable: false });
    expect(without).not.toContain("Stockfish");
  });

  it("shows a network's ratings only for that network", () => {
    // The first version rated every chess engine with the imitation network's
    // numbers, so the untrained one claimed to play at 2300.
    const ratings = new Map([["imitation", new Map([["Master", { rating: 2300, low: 2200, high: 2400 }]])]]);
    const networks = [{ file: "imitation", label: "Imitation 1" }, { file: "untrained", label: "Untrained" }];
    const measured = sheetHtml({
      entry: chess, settings: { ...defaults, network: "imitation" }, networks, ratings, cancellable: false,
    });
    const unmeasured = sheetHtml({
      entry: chess, settings: { ...defaults, network: "untrained" }, networks, ratings, cancellable: false,
    });
    expect(measured).toContain("about 2300");
    expect(unmeasured).not.toContain("about 2300");
    expect(unmeasured).not.toContain("Stockfish");
  });

  it("can be cancelled only when there is a game to go back to", () => {
    const first = sheetHtml({ entry: gomoku, settings: defaults, networks: [], cancellable: false });
    const later = sheetHtml({ entry: gomoku, settings: defaults, networks: [], cancellable: true });
    expect(first).not.toContain("Cancel");
    expect(later).toContain("Cancel");
  });
});

describe("reading the sheet back", () => {
  it("takes what was chosen", () => {
    const data = new FormData();
    data.set("level", "0");
    data.set("seat", "second");
    expect(readSettings(data, defaults)).toEqual({ level: 0, seat: "second", network: null });
  });

  it("keeps the previous choice for anything missing or malformed", () => {
    const data = new FormData();
    data.set("level", "99");
    data.set("seat", "sideways");
    expect(readSettings(data, { level: 2, seat: "random", network: "x" }))
      .toEqual({ level: 2, seat: "random", network: "x" });
  });
});

describe("choosing a side", () => {
  it("does what it says for first and second", () => {
    expect(resolveSeat("first")).toBe(true);
    expect(resolveSeat("second")).toBe(false);
  });

  it("really is random for Random, both ways", () => {
    expect(resolveSeat("random", () => 0.2)).toBe(true);
    expect(resolveSeat("random", () => 0.8)).toBe(false);
  });
});

describe("the player cards", () => {
  const base = { name: "You", detail: "White", status: "Your move", active: true, winner: false };

  it("lights the side to move and only that side", () => {
    expect(playerCardHtml({ ...base, side: "you" }, "")).toContain("player-card you active");
    expect(playerCardHtml({ ...base, side: "engine", active: false }, "")).not.toContain("active");
  });

  it("says nothing in the status slot when there is nothing to say", () => {
    expect(playerCardHtml({ ...base, side: "you", status: "" }, "")).not.toContain("player-status");
  });

  it("draws chess sides as kings of the right colour", () => {
    expect(tokenFor("chess", "you", true)).toContain("chess-piece white");
    expect(tokenFor("chess", "you", false)).toContain("chess-piece black");
  });

  it("draws other games' sides in their own colours", () => {
    expect(tokenFor("connect4", "engine", false)).toContain("disc engine");
    expect(tokenFor("isola", "you", true)).toContain("pawn you");
  });
});

describe("showing a rating", () => {
  it("rounds to the nearest fifty, which is all the measurement supports", () => {
    expect(approximately(2288)).toBe(2300);
    expect(approximately(2180)).toBe(2200);
    expect(approximately(1767)).toBe(1750);
    expect(approximately(1469)).toBe(1450);
  });

  it("shows the rounded figure on the level card, not the raw fit", () => {
    const ratings = new Map([["imitation", new Map([["Master", { rating: 2288, low: 2209, high: 2369 }]])]]);
    const html = sheetHtml({
      entry: chess, settings: { ...defaults, network: "imitation" }, networks: [], ratings, cancellable: false,
    });
    expect(html).toContain("about 2300");
    expect(html).not.toContain("2288");
  });
});
