/**
 * The TypeScript search must reproduce Python's exactly, for every game.
 *
 * With a uniform evaluator and no root noise, MCTS is completely deterministic,
 * so a port can be checked against Python's visit counts rather than merely
 * looking plausible. The algorithm has two separate sign flips, a specific
 * tie-breaking order and a root expanded before the loop - none of which
 * survives being reimplemented from memory.
 *
 * Running it over Reversi as well as Connect 4 is the browser-side version of
 * Phase 5's claim: the search does not know what game it is playing.
 */
import { describe, expect, it } from "vitest";
import { Chess } from "../src/games/chess";
import { Connect4 } from "../src/games/connect4";
import { DotsAndBoxes } from "../src/games/dotsandboxes";
import { Gomoku } from "../src/games/gomoku";
import { Isola } from "../src/games/isola";
import { Reversi } from "../src/games/reversi";
import type { Game } from "../src/games/types";
import { MCTS, UniformEvaluator, argmax } from "../src/engine/mcts";
import chessVectors from "./chess-vectors.json";
import connect4Vectors from "./connect4-vectors.json";
import dotsVectors from "./dotsandboxes-vectors.json";
import gomokuVectors from "./gomoku-vectors.json";
import isolaVectors from "./isola-vectors.json";
import reversiVectors from "./reversi-vectors.json";

interface SearchCase {
  moves: number[];
  simulations: number;
  visits?: number[];
  /** Chess lists visit counts by action rather than 4,672 wide. */
  visitsByAction?: number[][];
  rootValue: number;
}

const SUBJECTS: { game: Game<unknown>; searches: SearchCase[] }[] = [
  { game: new Connect4() as Game<unknown>, searches: connect4Vectors.searches },
  { game: new Reversi() as Game<unknown>, searches: reversiVectors.searches },
  { game: new Gomoku() as Game<unknown>, searches: gomokuVectors.searches },
  { game: new Isola() as Game<unknown>, searches: isolaVectors.searches },
  // Drawn from endgames: see SEARCH_PLIES in scripts/testvectors.py for why.
  { game: new DotsAndBoxes() as Game<unknown>, searches: dotsVectors.searches },
  { game: new Chess() as Game<unknown>, searches: chessVectors.searches },
];

/** The visit counts a case expects, stored densely or, for chess, by action. */
function visitsOf(testCase: SearchCase, actionSize: number): number[] {
  if (testCase.visits) return testCase.visits;
  const visits = new Array<number>(actionSize).fill(0);
  for (const [action, count] of testCase.visitsByAction ?? []) visits[action] = count;
  return visits;
}

function engine(game: Game<unknown>, simulations: number) {
  return new MCTS(game, new UniformEvaluator(), {
    simulations,
    cPuct: 1.5,
    dirichletAlpha: 1.0,
    dirichletEpsilon: 0.25,
  });
}

for (const { game, searches } of SUBJECTS) {
  // Three tests replay every case; states are immutable, so once is enough.
  const replayed = new Map<string, unknown>();
  const replay = (moves: number[]) => {
    const key = moves.join(",");
    if (!replayed.has(key)) {
      let state = game.initialState();
      for (const move of moves) state = game.apply(state, move);
      replayed.set(key, state);
    }
    return replayed.get(key);
  };

  describe(`${game.name} search agrees with Python`, () => {
    it("has varied search vectors to check against", () => {
      expect(searches.length).toBeGreaterThan(20);
      expect(new Set(searches.map((s) => s.simulations)).size).toBeGreaterThan(1);
    });

    it("reproduces Python's visit counts exactly", async () => {
      for (const testCase of searches) {
        const search = engine(game, testCase.simulations);
        const root = await search.search(replay(testCase.moves), false);
        expect(
          search.visitCounts(root),
          `moves ${testCase.moves} at ${testCase.simulations} sims`,
        ).toEqual(visitsOf(testCase, game.actionSize));
      }
    });

    it("reproduces Python's root values", async () => {
      for (const testCase of searches) {
        const search = engine(game, testCase.simulations);
        const root = await search.search(replay(testCase.moves), false);
        expect(root.value()).toBeCloseTo(testCase.rootValue, 10);
      }
    });

    it("spends exactly the simulation budget", async () => {
      for (const testCase of searches) {
        const search = engine(game, testCase.simulations);
        const root = await search.search(replay(testCase.moves), false);
        expect(root.visitCount).toBe(testCase.simulations);
        const total = search.visitCounts(root).reduce((a, b) => a + b, 0);
        expect(total).toBe(testCase.simulations - 1);
      }
    });
  });
}

describe("search behaviour", () => {
  const game = new Connect4() as Game<unknown>;
  const replay = (moves: number[]) => {
    let state = game.initialState();
    for (const move of moves) state = game.apply(state, move);
    return state;
  };

  it("finds a win in one with no knowledge at all", async () => {
    const { policy } = await engine(game, 100).run(replay([0, 1, 0, 1, 0, 1]), 0);
    expect(argmax(policy)).toBe(0);
  });

  it("finds the only blocking move", async () => {
    const { policy } = await engine(game, 600).run(replay([0, 1, 0, 1, 0]), 0);
    expect(argmax(policy)).toBe(0);
  });

  it("reports a winning position as positive for the mover", async () => {
    const { value } = await engine(game, 100).run(replay([0, 1, 0, 1, 0, 1]), 0);
    expect(value).toBeGreaterThan(0.5);
  });

  it("refuses to search a finished game", async () => {
    await expect(engine(game, 10).search(replay([0, 1, 0, 1, 0, 1, 0]))).rejects.toThrow();
  });

  it("never spends visits on an illegal move", async () => {
    let state = game.initialState();
    for (let i = 0; i < 6; i++) state = game.apply(state, 0);
    const { policy } = await engine(game, 100).run(state, 1);
    expect(policy[0]).toBe(0);
  });

  it("plays greedily at temperature zero and proportionally at one", async () => {
    const state = replay([3, 3, 4]);
    const greedy = await engine(game, 80).run(state, 0);
    expect(Array.from(greedy.policy).filter((p) => p > 0).length).toBe(1);

    const soft = await engine(game, 80).run(state, 1);
    expect(Array.from(soft.policy).filter((p) => p > 0).length).toBeGreaterThan(1);
    expect(Array.from(soft.policy).reduce((a, b) => a + b, 0)).toBeCloseTo(1, 6);
  });

  it("perturbs root priors when noise is on, and only at the root", async () => {
    const state = replay([3]);
    const seeded = new MCTS(
      game,
      new UniformEvaluator(),
      { simulations: 200, cPuct: 1.5, dirichletAlpha: 1.0, dirichletEpsilon: 0.25 },
      () => 0.42,
    );
    const clean = await engine(game, 200).search(state, false);
    const noisy = await seeded.search(state, true);

    const cleanPriors = [...clean.children.values()].map((c) => c.prior);
    const noisyPriors = [...noisy.children.values()].map((c) => c.prior);
    expect(noisyPriors).not.toEqual(cleanPriors);
    expect(noisyPriors.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 6);

    for (const child of noisy.children.values()) {
      if (child.children.size === 0) continue;
      const deeper = [...child.children.values()].map((c) => c.prior);
      for (const prior of deeper) expect(prior).toBeCloseTo(deeper[0], 10);
    }
  });
});
