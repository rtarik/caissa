/**
 * The How it works page's table of level gaps.
 *
 * Its first version printed "off the scale" for a clean sweep, which left the
 * reader guessing, and showed only the self-play ladder, which chess proved
 * flatters. The failures worth catching are quiet ones: a sweep's placeholder
 * Elo leaking out as a number, a gap taken between the wrong two levels, or a
 * bound subtracted from a rating as if it were one.
 */
import { describe, expect, it } from "vitest";
import { gapText, itselfText, olderSteps, stockfishSteps, strengthRows } from "../src/strength";
import type { Rating } from "../src/ui/sheet";

const rated = (entries: [string, Rating][]) => new Map(entries);

describe("a step measured against itself", () => {
  it("says a clean sweep happened, rather than printing the fit's placeholder", () => {
    // The fit gives a sweep 3600: the edge of its range, not a measurement.
    const text = itselfText({ level: "Strong", elo: 3600, score: 1 });
    expect(text).toBe("won every game");
    expect(text).not.toContain("3600");
    expect(itselfText({ level: "Strong", elo: -3600, score: 0 })).toBe("lost every game");
  });

  it("signs the gap, whichever way it goes", () => {
    expect(itselfText({ level: "Master", elo: 373.8, score: 0.9 })).toBe("+374");
    expect(itselfText({ level: "Master", elo: -26.2, score: 0.46 })).toBe("−26");
  });
});

describe("steps from Stockfish ratings", () => {
  const imitation = rated([
    ["Beginner", { rating: 1423, low: 1338, high: 1505 }],
    ["Casual", { rating: 1858, low: 1783, high: 1933 }],
    ["Strong", { rating: 2047, low: 1972, high: 2123 }],
    ["Master", { rating: 2293, low: 2213, high: 2374 }],
  ]);

  it("takes each level over the one directly below it", () => {
    expect(stockfishSteps(imitation).map((step) => step?.gap)).toEqual([435, 189, 246]);
  });

  it("adds the two ratings' uncertainties as independent errors add", () => {
    // Half-widths of 30 and 40 make 50 in quadrature; end to end would say 70.
    const ratings = rated([
      ["Beginner", { rating: 1000, low: 970, high: 1030 }],
      ["Casual", { rating: 1400, low: 1360, high: 1440 }],
    ]);
    expect(stockfishSteps(ratings)[0]?.plusMinus).toBeCloseTo(50, 6);
  });

  it("gives no step where either end is only a bound", () => {
    // Every rating here is a ceiling or a rating next to one: no gap is a measurement.
    const untrained = rated([
      ["Beginner", { rating: 0, low: 0, high: 958, below: 958 }],
      ["Casual", { rating: 1200, low: 1100, high: 1300 }],
      ["Strong", { rating: 1400, low: 1300, high: 1500 }],
      ["Master", { rating: 4000, low: 2640, high: 4000, above: 2640 }],
    ]);
    expect(stockfishSteps(untrained).map((step) => step?.gap ?? null)).toEqual([null, 200, null]);
    expect(stockfishSteps(undefined)).toEqual([null, null, null]);
  });
});

describe("steps from an older network", () => {
  it("keeps measured steps, with half their range, and drops the ones a sweep decided", () => {
    const steps = [
      { level: "Casual", gap: 353, low: 262, high: 433, measured: true },
      { level: "Strong", gap: 320, low: 229, high: 424, measured: true },
      { level: "Master", gap: 195, low: 0, high: 323, measured: false },
    ];
    expect(olderSteps(steps)).toEqual([
      { gap: 353, plusMinus: 85.5 },
      { gap: 320, plusMinus: 97.5 },
      null,
    ]);
  });

  it("matches steps by level, not by position", () => {
    const shuffled = [
      { level: "Master", gap: 150, low: 100, high: 200, measured: true },
      { level: "Casual", gap: 300, low: 250, high: 350, measured: true },
    ];
    expect(olderSteps(shuffled).map((step) => step?.gap ?? null)).toEqual([300, null, 150]);
    expect(olderSteps(undefined)).toEqual([null, null, null]);
  });
});

describe("a gap in words", () => {
  it("gives the uncertainty to the nearest ten, never split from its figure by a line break", () => {
    expect(gapText({ gap: 181, plusMinus: 149 })).toBe("+181\u00a0±150");
    expect(gapText({ gap: 181, plusMinus: 144 })).toBe("+181\u00a0±140");
    expect(gapText({ gap: -26, plusMinus: 40 })).toBe("−26\u00a0±40");
  });
});

describe("the table", () => {
  const games = [
    { key: "connect4", title: "Four in a Row" },
    { key: "chess", title: "Chess" },
    { key: "nothing", title: "Unmeasured" },
  ];
  const own = {
    connect4: [
      { level: "Casual", elo: 176, score: 0.73 },
      { level: "Strong", elo: 199, score: 0.76 },
      { level: "Master", elo: 114, score: 0.66 },
    ],
    chess: [
      { level: "Casual", elo: 470, score: 0.94 },
      { level: "Strong", elo: 3600, score: 1 },
      { level: "Master", elo: 374, score: 0.9 },
    ],
  };

  it("has a row for each game with a ladder, in the order given", () => {
    const rows = strengthRows(games, own, {});
    expect(rows.map((row) => row.title)).toEqual(["Four in a Row", "Chess"]);
  });

  it("puts both measurements of a step side by side", () => {
    const outside = { chess: [435, 189, 246].map((gap) => ({ gap, plusMinus: 110 })) };
    const [connect4, chess] = strengthRows(games, own, outside);
    expect(chess.steps).toEqual([
      { itself: "+470", outside: "+435\u00a0±110" },
      { itself: "won every game", outside: "+189\u00a0±110" },
      { itself: "+374", outside: "+246\u00a0±110" },
    ]);
    // Nothing outside to measure against: the self-play figure stands alone.
    expect(connect4.steps.map((step) => step.outside)).toEqual([null, null, null]);
    expect(connect4.steps[0].itself).toBe("+176");
  });
});
