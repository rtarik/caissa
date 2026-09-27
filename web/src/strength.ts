/**
 * How far apart the four levels are, measured two ways, for the How it works page.
 *
 * Against itself: each level plays the one below it, the same network on both
 * sides (`scripts/levels.py`). Against outsiders: each level plays opponents from
 * outside its own family - Stockfish for chess (`scripts/stockfish.py`), and an
 * earlier generation of the same network for the games that kept one
 * (`scripts/crossfamily.py`).
 *
 * The first is cheap and flatters: a deeper search of the same network knows
 * exactly where its shallower twin will go wrong, and aims for it. The second is
 * what a step is worth against anybody else, which is what a player meets.
 */
import { LEVELS } from "./levels";
import { measuredElo, type Rating } from "./ui/choices";

/** One rung of the self-play ladder, as `levels.json` has it. */
export interface OwnRung {
  level: string;
  elo: number;
  score: number;
}

/** One step measured against an older network, as `crossfamily.json` has it. */
export interface OutsideStep {
  level: string;
  gap: number;
  /** The ends of its 95% range, from resampling the games. */
  low: number;
  high: number;
  /** False when a level on either side won or lost every game: then the fit's prior set the gap. */
  measured: boolean;
}

/** A step measured from outside, and how far either way it could be at 95%. */
export interface Gap {
  gap: number;
  plusMinus: number;
}

/** A game's row: for each step up from Beginner, the gap both ways, or null where it was not measured. */
export interface StrengthRow {
  title: string;
  steps: { itself: string | null; outside: string | null }[];
}

const STEPS = LEVELS.slice(1);

const signed = (value: number) => `${value < 0 ? "−" : "+"}${Math.abs(Math.round(value))}`;

/**
 * A step measured against the level below, in words.
 *
 * A clean sweep has no Elo: the fit only knows the gap is large, not how large,
 * so it says what happened rather than printing a number the games never gave.
 */
export function itselfText(rung: OwnRung | undefined): string | null {
  if (!rung) return null;
  if (rung.score >= 1) return "won every game";
  if (rung.score <= 0) return "lost every game";
  return signed(rung.elo);
}

/** A 95% range's half-width, as a standard error. */
const standardError = (rating: Rating) => (rating.high - rating.low) / (2 * 1.96);

/**
 * The steps between consecutive levels from their Stockfish ratings - null
 * where either end is a bound rather than a rating, since the gap between a
 * ceiling and a rating is not a measurement.
 *
 * The two ratings were fitted from separate games, so their errors add the way
 * independent errors do: in quadrature, not end to end.
 */
export function stockfishSteps(ratings: Map<string, Rating> | undefined): (Gap | null)[] {
  return STEPS.map((level, index) => {
    const upper = ratings?.get(level.label);
    const lower = ratings?.get(LEVELS[index].label);
    const top = measuredElo(upper);
    const bottom = measuredElo(lower);
    if (top === undefined || bottom === undefined) return null;
    return {
      gap: top - bottom,
      plusMinus: 1.96 * Math.hypot(standardError(upper!), standardError(lower!)),
    };
  });
}

/** The steps measured against an older network, null where the result was a sweep. */
export function olderSteps(steps: OutsideStep[] | undefined): (Gap | null)[] {
  return STEPS.map((level) => {
    const step = steps?.find((s) => s.level === level.label);
    return step?.measured ? { gap: step.gap, plusMinus: (step.high - step.low) / 2 } : null;
  });
}

/**
 * A gap and its uncertainty, the uncertainty to the nearest ten: "+181 ±150".
 * Joined by a non-breaking space, so a narrow table never splits a figure from
 * its margin.
 */
export function gapText({ gap, plusMinus }: Gap): string {
  return `${signed(gap)}\u00a0±${Math.round(plusMinus / 10) * 10}`;
}

/**
 * The table's rows, one per game that has a self-play ladder, in the order given.
 * `outside` holds each game's outside steps, from `stockfishSteps` or `olderSteps`.
 */
export function strengthRows(
  games: readonly { key: string; title: string }[],
  own: Record<string, OwnRung[]>,
  outside: Record<string, (Gap | null)[]>,
): StrengthRow[] {
  return games.filter((game) => own[game.key]).map((game) => ({
    title: game.title,
    steps: STEPS.map((level, index) => {
      const gap = outside[game.key]?.[index] ?? null;
      return {
        itself: itselfText(own[game.key].find((rung) => rung.level === level.label)),
        outside: gap === null ? null : gapText(gap),
      };
    }),
  }));
}
