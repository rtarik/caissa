/**
 * The TypeScript search must reproduce Python's exactly.
 *
 * With a uniform evaluator and no root noise, MCTS is completely deterministic:
 * the same position and simulation budget give the same visit counts every time.
 * That makes a port checkable rather than merely plausible - and the algorithm is
 * fiddly enough (two separate sign flips, a specific tie-breaking order, a root
 * expanded before the loop) that "looks right" is not good enough.
 */
import { describe, expect, it } from "vitest";
import { Connect4 } from "../src/games/connect4";
import { MCTS, UniformEvaluator, argmax } from "../src/engine/mcts";
import vectors from "./vectors.json";

interface SearchCase {
  moves: number[];
  simulations: number;
  visits: number[];
  rootValue: number;
}

const game = new Connect4();
const searches = vectors.searches as SearchCase[];

function replay(moves: number[]) {
  let state = game.initialState();
  for (const move of moves) state = game.apply(state, move);
  return state;
}

function search(simulations: number) {
  return new MCTS(game, new UniformEvaluator(), {
    simulations,
    cPuct: 1.5,
    dirichletAlpha: 1.0,
    dirichletEpsilon: 0.25,
  });
}

describe("agreement with the Python search", () => {
  it("has search vectors to check against", () => {
    expect(searches.length).toBeGreaterThan(20);
    expect(new Set(searches.map((s) => s.simulations)).size).toBeGreaterThan(1);
  });

  it("reproduces Python's visit counts exactly", async () => {
    for (const testCase of searches) {
      const engine = search(testCase.simulations);
      const root = await engine.search(replay(testCase.moves), false);
      expect(
        engine.visitCounts(root),
        `moves ${testCase.moves} at ${testCase.simulations} sims`,
      ).toEqual(testCase.visits);
    }
  });

  it("reproduces Python's root values", async () => {
    for (const testCase of searches) {
      const engine = search(testCase.simulations);
      const root = await engine.search(replay(testCase.moves), false);
      expect(root.value()).toBeCloseTo(testCase.rootValue, 10);
    }
  });

  it("spends exactly the simulation budget", async () => {
    for (const testCase of searches) {
      const engine = search(testCase.simulations);
      const root = await engine.search(replay(testCase.moves), false);
      expect(root.visitCount).toBe(testCase.simulations);
      // Every simulation after the root's own expansion descends into a child.
      const total = engine.visitCounts(root).reduce((a, b) => a + b, 0);
      expect(total).toBe(testCase.simulations - 1);
    }
  });
});

describe("search behaviour", () => {
  it("finds a win in one with no knowledge at all", async () => {
    // Three in column 0 for the player to move.
    const { policy } = await search(100).run(replay([0, 1, 0, 1, 0, 1]), 0);
    expect(argmax(policy)).toBe(0);
  });

  it("finds the only blocking move", async () => {
    const { policy } = await search(600).run(replay([0, 1, 0, 1, 0]), 0);
    expect(argmax(policy)).toBe(0);
  });

  it("reports a winning position as positive for the mover", async () => {
    const { value } = await search(100).run(replay([0, 1, 0, 1, 0, 1]), 0);
    expect(value).toBeGreaterThan(0.5);
  });

  it("refuses to search a finished game", async () => {
    await expect(search(10).search(replay([0, 1, 0, 1, 0, 1, 0]))).rejects.toThrow();
  });

  it("never spends visits on an illegal move", async () => {
    let state = game.initialState();
    for (let i = 0; i < 6; i++) state = game.apply(state, 0); // column 0 full
    const { policy } = await search(100).run(state, 1);
    expect(policy[0]).toBe(0);
  });

  it("plays greedily at temperature zero and proportionally at one", async () => {
    const state = replay([3, 3, 4]);
    const greedy = await search(80).run(state, 0);
    expect(Array.from(greedy.policy).filter((p) => p > 0).length).toBe(1);

    const soft = await search(80).run(state, 1);
    expect(Array.from(soft.policy).filter((p) => p > 0).length).toBeGreaterThan(1);
    expect(Array.from(soft.policy).reduce((a, b) => a + b, 0)).toBeCloseTo(1, 6);
  });

  it("perturbs root priors when noise is on, and only at the root", async () => {
    const state = replay([3]);
    const engine = search(200);
    const seeded = new MCTS(
      game,
      new UniformEvaluator(),
      { simulations: 200, cPuct: 1.5, dirichletAlpha: 1.0, dirichletEpsilon: 0.25 },
      () => 0.42,
    );

    const clean = await engine.search(state, false);
    const noisy = await seeded.search(state, true);

    const cleanPriors = [...clean.children.values()].map((c) => c.prior);
    const noisyPriors = [...noisy.children.values()].map((c) => c.prior);
    expect(noisyPriors).not.toEqual(cleanPriors);
    expect(noisyPriors.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 6);

    // Deeper nodes keep the evaluator's untouched, uniform priors.
    for (const child of noisy.children.values()) {
      if (child.children.size === 0) continue;
      const deeper = [...child.children.values()].map((c) => c.prior);
      for (const prior of deeper) expect(prior).toBeCloseTo(deeper[0], 10);
    }
  });
});
