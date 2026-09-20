/**
 * The site's own rules: what a URL means, what the levels are, what a game card
 * needs before it can be shown.
 *
 * `main.ts` starts a worker and rewrites the document when it is imported, so
 * the parts worth pinning live in modules of their own. These are the three that
 * would fail quietly: a route that silently becomes the gallery, a difficulty
 * list whose order or default drifts, and a game added to the ladder without the
 * icon or the description the gallery draws.
 */
import { describe, expect, it } from "vitest";
import { LADDER } from "../src/games/ladder";
import { DEFAULT_LEVEL, LEVELS } from "../src/levels";
import { href, parseRoute } from "../src/routes";
import { ICONS, iconFor } from "../src/ui/icons";

describe("routes", () => {
  it("reads a game out of the hash", () => {
    expect(parseRoute("#/play/gomoku")).toEqual({ name: "play", key: "gomoku" });
    expect(parseRoute("#/how-it-works")).toEqual({ name: "about" });
    expect(parseRoute("#/")).toEqual({ name: "gallery" });
    expect(parseRoute("")).toEqual({ name: "gallery" });
  });

  it("sends anything it does not recognise to the gallery", () => {
    // A stale bookmark or a renamed game lands somewhere that works.
    expect(parseRoute("#/play/backgammon")).toEqual({ name: "gallery" });
    expect(parseRoute("#/play/")).toEqual({ name: "gallery" });
    expect(parseRoute("#/nonsense")).toEqual({ name: "gallery" });
  });

  it("round-trips every game in the ladder", () => {
    for (const item of LADDER) {
      expect(parseRoute(href({ name: "play", key: item.key }))).toEqual({
        name: "play",
        key: item.key,
      });
    }
    expect(parseRoute(href({ name: "about" })).name).toBe("about");
    expect(parseRoute(href({ name: "gallery" })).name).toBe("gallery");
  });
});

describe("difficulty levels", () => {
  it("runs weakest to strongest", () => {
    const budgets = LEVELS.map((level) => level.simulations);
    expect(budgets).toEqual([...budgets].sort((a, b) => a - b));
    expect(new Set(budgets).size).toBe(budgets.length);
  });

  it("starts everyone on the strongest one", () => {
    // Deliberate: an easier game is there for whoever goes looking, but the
    // default should be the best the engine can do.
    expect(DEFAULT_LEVEL).toBe(LEVELS.length - 1);
    expect(LEVELS[DEFAULT_LEVEL].simulations).toBe(
      Math.max(...LEVELS.map((level) => level.simulations)),
    );
  });

  it("describes each one without naming a search", () => {
    for (const level of LEVELS) {
      expect(level.label).toMatch(/^[A-Z]/);
      expect(level.note.length).toBeGreaterThan(8);
      expect(level.note.toLowerCase()).not.toMatch(/simulation|mcts|network|node/);
    }
  });
});

describe("the gallery's games", () => {
  it("draws an icon for every one of them", () => {
    for (const item of LADDER) {
      expect(ICONS[item.key], `no icon for ${item.key}`).toBeTruthy();
      expect(iconFor(item.key)).toContain("<svg");
    }
  });

  it("falls back rather than rendering nothing", () => {
    expect(iconFor("backgammon")).toContain("<svg");
  });

  it("says what each game is and how it is won", () => {
    for (const item of LADDER) {
      expect(item.blurb.length, item.key).toBeGreaterThan(10);
      expect(item.howToWin.length, item.key).toBeGreaterThan(20);
      expect(item.teaches.length, item.key).toBeGreaterThan(10);
      // The play screen is for playing: the framework talk belongs elsewhere.
      expect(item.howToWin.toLowerCase()).not.toMatch(/network|search|policy|encoding/);
    }
  });
});
