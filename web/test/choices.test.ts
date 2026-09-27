/**
 * The choices before a game, and the player cards that carry them.
 *
 * All of it is a pure render, which is the point: what the cards offer and what
 * they say can be checked without a browser. The failures worth catching are
 * the quiet ones - a switch that forgets what you chose, a level out of range
 * slipping through, a Random that is not random, a game that starts before you
 * are ready or refuses to when you are.
 */
import { describe, expect, it } from "vitest";
import { LADDER } from "../src/games/ladder";
import { DEFAULT_LEVEL, LEVELS } from "../src/levels";
import { playerCardHtml, tokenFor } from "../src/ui/players";
import {
  approximately, chooseSetting, levelDetail, levelPickerHtml, measuredElo, newGamePanelHtml, ratingText,
  resolveSeat, seatPickerHtml, startHint, waitsForStart, type Settings,
} from "../src/ui/choices";

const chess = LADDER.find((item) => item.key === "chess")!;
const gomoku = LADDER.find((item) => item.key === "gomoku")!;
const defaults: Settings = { level: DEFAULT_LEVEL, seat: "first" };

/** The value of the checked radio in each named group. */
function checkedValues(html: string): Record<string, string> {
  const found: Record<string, string> = {};
  for (const match of html.matchAll(/<input type="radio" name="(\w+)" value="([^"]+)" checked>/g)) {
    found[match[1]] = match[2];
  }
  return found;
}

describe("the switches on the cards", () => {
  it("offer every level, with the one chosen checked", () => {
    const html = levelPickerHtml(1);
    for (const level of LEVELS) expect(html).toContain(`>${level.label}<`);
    expect(checkedValues(html)).toEqual({ level: "1" });
  });

  it("name the sides the way the game does, with Random between them", () => {
    const html = seatPickerHtml(chess, "first");
    expect(html).toContain(">White<");
    expect(html).toContain(">Random<");
    expect(html).toContain(">Black<");
    const plain = seatPickerHtml(gomoku, "random");
    expect(plain).toContain(">First<");
    expect(plain).toContain(">Second<");
    expect(checkedValues(plain)).toEqual({ seat: "random" });
  });
});

describe("changing a choice", () => {
  it("takes a level or a side", () => {
    expect(chooseSetting(defaults, "level", "0")).toEqual({ level: 0, seat: "first" });
    expect(chooseSetting(defaults, "seat", "second")).toEqual({ level: DEFAULT_LEVEL, seat: "second" });
  });

  it("keeps the previous choice for anything out of range or malformed", () => {
    for (const value of [String(LEVELS.length), "-1", "1.5", "strong", ""]) {
      expect(chooseSetting(defaults, "level", value)).toBe(defaults);
    }
    expect(chooseSetting(defaults, "seat", "sideways")).toBe(defaults);
    // A field it does not know changes nothing, even with a value a level would accept.
    expect(chooseSetting(defaults, "network", "0")).toBe(defaults);
  });
});

describe("what starts the game", () => {
  it("is your first move when that move is yours", () => {
    expect(waitsForStart("first", true)).toBe(false);
    expect(startHint("first", true)).toBe("Your first move starts the game.");
  });

  it("is Start whenever Caissa opens - including from a position with the other side to move", () => {
    expect(waitsForStart("second", false)).toBe(true);
    expect(waitsForStart("first", false)).toBe(true);
    expect(startHint("second", false)).toContain("Press Start");
  });

  it("is Start for Random, which has no side until then", () => {
    expect(waitsForStart("random", true)).toBe(true);
    expect(waitsForStart("random", false)).toBe(true);
    expect(startHint("random", true)).toContain("coin");
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

describe("the new-game panel", () => {
  const panel = { entry: gomoku, settings: defaults, youMoveFirst: true, rated: false };

  it("says what winning means and what starts the game", () => {
    const html = newGamePanelHtml(panel);
    expect(html).toContain(gomoku.howToWin);
    expect(html).toContain("Your first move starts the game.");
    expect(newGamePanelHtml({ ...panel, settings: { ...defaults, seat: "second" }, youMoveFirst: false }))
      .toContain("Press Start");
  });

  it("offers a set-up position only for the game that has one, and a way back from it", () => {
    expect(newGamePanelHtml(panel)).not.toContain("Set up a position");
    const usual = newGamePanelHtml({ ...panel, entry: chess, start: { fen: null } });
    expect(usual).toContain("Set up a position");
    expect(usual).not.toContain("Use the usual start");
    const own = newGamePanelHtml({ ...panel, entry: chess, start: { fen: "4k3/8/8/8/8/8/4P3/4K3 w - - 0 1" } });
    expect(own).toContain("Edit position");
    expect(own).toContain("Use the usual start");
  });

  it("explains the ratings only where there are some", () => {
    expect(newGamePanelHtml({ ...panel, rated: true })).toContain("Stockfish");
    expect(newGamePanelHtml(panel)).not.toContain("Stockfish");
  });

  it("links the history for games that keep one, with how many it holds", () => {
    expect(newGamePanelHtml({ ...panel, entry: chess, saved: 3 })).toContain("Your games (3)");
    expect(newGamePanelHtml({ ...panel, entry: chess, saved: 0 })).toContain(">Your games<");
    expect(newGamePanelHtml(panel)).not.toContain("Your games");
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

  it("carries a choice on a row of its own only before the game", () => {
    const choosing = playerCardHtml({ ...base, side: "you", controls: seatPickerHtml(chess, "first") }, "");
    expect(choosing).toContain("player-card you active choosing");
    expect(choosing).toContain('<div class="player-controls">');
    const playing = playerCardHtml({ ...base, side: "you" }, "");
    expect(playing).not.toContain("choosing");
    expect(playing).not.toContain("player-controls");
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

describe("the level on Caissa's card", () => {
  const master = LEVELS[LEVELS.length - 1];
  const measured = { rating: 2288, low: 2209, high: 2369 };

  it("shows the rounded rating, not the raw fit", () => {
    expect(levelDetail(master, measured, false)).toBe("Rated about 2300");
  });

  it("adds what the level is like to play while you are choosing", () => {
    expect(levelDetail(master, measured, true)).toBe(`Rated about 2300 · ${master.note}`);
  });

  it("describes an unrated level in words, choosing or not", () => {
    expect(levelDetail(master, undefined, false)).toBe(master.note);
    expect(levelDetail(master, undefined, true)).toBe(master.note);
  });

  it("shows a bound as a bound, never as a rating of zero", () => {
    // An untrained network lost every game to Stockfish's weakest setting, so
    // the fit ran into the bottom of its range and returned 0 - not a rating.
    const bound = { rating: 0, low: 0, high: 958, below: 958 };
    expect(levelDetail(master, bound, false)).toBe("Rated under 1000");
  });
});

describe("showing a rating", () => {
  it("rounds to the nearest fifty, which is all the measurement supports", () => {
    expect(approximately(2288)).toBe(2300);
    expect(approximately(2180)).toBe(2200);
    expect(approximately(1767)).toBe(1750);
    expect(approximately(1469)).toBe(1450);
  });

  it("rounds a bound outwards, so that it never claims more than was measured", () => {
    expect(ratingText({ rating: 720, low: 0, high: 1064, below: 1064 })).toBe("under 1100");
    expect(ratingText({ rating: 4000, low: 2640, high: 4000, above: 2640 })).toBe("over 2600");
    expect(ratingText({ rating: 2288, low: 2209, high: 2369 })).toBe("about 2300");
  });

  it("gives a PGN header a measured rating, never a bound", () => {
    expect(measuredElo({ rating: 2293, low: 2213, high: 2374 })).toBe(2293);
    expect(measuredElo({ rating: 720, low: 0, high: 1064, below: 1064 })).toBeUndefined();
    expect(measuredElo({ rating: 4000, low: 2640, high: 4000, above: 2640 })).toBeUndefined();
    expect(measuredElo(undefined)).toBeUndefined();
  });
});
