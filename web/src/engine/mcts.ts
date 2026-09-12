/**
 * Monte Carlo tree search, ported from `src/caissa/mcts.py`.
 *
 * Deliberately a line-for-line port rather than a fresh implementation, and
 * checked against Python's visit counts by `test/mcts.test.ts`: with a uniform
 * evaluator and no root noise the search is fully deterministic, so the two can
 * be compared exactly rather than approximately.
 *
 * The two sign flips are the part to leave alone. A simulation's result is
 * pushed up the tree negated at every ply, and a child's mean value is negated
 * when its parent chooses between children - because a child's statistics are
 * recorded from the point of view of *its* mover, who is the opponent of whoever
 * is choosing. Removing either produces an agent that plays confidently toward
 * losing, with nothing about it looking wrong.
 */
import type { Game } from "../games/types";

export interface Evaluation {
  /** Probability per action, already masked to legal moves and summing to 1. */
  priors: Float32Array;
  /** Expected result in [-1, 1], for the player to move. */
  value: number;
}

export interface Evaluator<S> {
  evaluate(game: Game<S>, state: S): Promise<Evaluation>;
}

export interface SearchConfig {
  simulations: number;
  cPuct: number;
  dirichletAlpha: number;
  dirichletEpsilon: number;
}

export const DEFAULT_SEARCH: SearchConfig = {
  simulations: 200,
  cPuct: 1.5,
  dirichletAlpha: 1.0,
  dirichletEpsilon: 0.25,
};

class Node<S> {
  visitCount = 0;
  valueSum = 0;
  children = new Map<number, Node<S>>();

  constructor(
    public prior: number,
    public state: S,
  ) {}

  get expanded(): boolean {
    return this.children.size > 0;
  }

  /** Mean outcome over simulations through this node, for *its* mover. */
  value(): number {
    return this.visitCount === 0 ? 0 : this.valueSum / this.visitCount;
  }
}

/** A random source, so searches can be made reproducible in tests. */
export type Random = () => number;

export class MCTS<S> {
  constructor(
    private game: Game<S>,
    private evaluator: Evaluator<S>,
    private config: SearchConfig = DEFAULT_SEARCH,
    private random: Random = Math.random,
  ) {}

  async search(state: S, addNoise = false): Promise<Node<S>> {
    if (this.game.terminalValue(state) !== null) {
      throw new Error("cannot search from a finished game");
    }

    const root = new Node<S>(0, state);
    this.backup([root], await this.evaluate(root));
    if (addNoise) this.addDirichletNoise(root);

    for (let i = 0; i < this.config.simulations - 1; i++) {
      const path = this.select(root);
      this.backup(path, await this.evaluate(path[path.length - 1]));
    }
    return root;
  }

  /** Search `state` and return the move distribution plus the root's value. */
  async run(state: S, temperature = 0, addNoise = false) {
    const root = await this.search(state, addNoise);
    return { policy: this.policy(root, temperature), value: root.value() };
  }

  private select(root: Node<S>): Node<S>[] {
    const path = [root];
    let node = root;
    while (node.expanded) {
      node = this.bestChild(node);
      path.push(node);
    }
    return path;
  }

  private bestChild(node: Node<S>): Node<S> {
    const sqrtParentVisits = Math.sqrt(node.visitCount);
    let best: Node<S> | null = null;
    let bestScore = -Infinity;

    for (const child of node.children.values()) {
      // Negated: the child's statistics belong to the opponent.
      const exploit = child.visitCount > 0 ? -child.value() : 0;
      const explore =
        (this.config.cPuct * child.prior * sqrtParentVisits) / (1 + child.visitCount);
      const score = exploit + explore;
      if (score > bestScore) {
        bestScore = score;
        best = child;
      }
    }
    if (best === null) throw new Error("expanded node with no children");
    return best;
  }

  private async evaluate(node: Node<S>): Promise<number> {
    // A finished position needs no network: the rules give the exact answer.
    const terminal = this.game.terminalValue(node.state);
    if (terminal !== null) return terminal;

    const { priors, value } = await this.evaluator.evaluate(this.game, node.state);
    const legal = this.game.legalActions(node.state);
    for (let action = 0; action < legal.length; action++) {
      if (!legal[action]) continue;
      node.children.set(action, new Node(priors[action], this.game.apply(node.state, action)));
    }
    return value;
  }

  private backup(path: Node<S>[], value: number): void {
    // The leaf's value belongs to the leaf's mover; every ply up is the opponent.
    for (let i = path.length - 1; i >= 0; i--) {
      path[i].visitCount += 1;
      path[i].valueSum += value;
      value = -value;
    }
  }

  private addDirichletNoise(root: Node<S>): void {
    // Root only. The aim is to vary the games that get played, not to degrade
    // the search's judgement inside a line.
    const actions = [...root.children.keys()];
    const noise = dirichlet(actions.length, this.config.dirichletAlpha, this.random);
    const epsilon = this.config.dirichletEpsilon;
    actions.forEach((action, i) => {
      const child = root.children.get(action)!;
      child.prior = (1 - epsilon) * child.prior + epsilon * noise[i];
    });
  }

  /**
   * The search's move distribution - **visit counts**, not the network's priors
   * and not the mean values. A move earns visits only by repeatedly winning the
   * selection argument against its siblings, so visits aggregate every
   * simulation's evidence, while a mean value can rest on one lucky rollout.
   */
  policy(root: Node<S>, temperature = 0): Float32Array {
    const counts = new Float32Array(this.game.actionSize);
    for (const [action, child] of root.children) counts[action] = child.visitCount;

    if (temperature === 0) {
      const policy = new Float32Array(this.game.actionSize);
      policy[argmax(counts)] = 1;
      return policy;
    }

    // Scaled by the maximum first, so small temperatures do not overflow.
    const max = Math.max(...counts);
    const weights = counts.map((c) => Math.pow(c / max, 1 / temperature));
    const total = weights.reduce((a, b) => a + b, 0);
    return weights.map((w) => w / total) as Float32Array;
  }

  visitCounts(root: Node<S>): number[] {
    const counts = new Array(this.game.actionSize).fill(0);
    for (const [action, child] of root.children) counts[action] = child.visitCount;
    return counts;
  }
}

export function argmax(values: ArrayLike<number>): number {
  let best = 0;
  for (let i = 1; i < values.length; i++) if (values[i] > values[best]) best = i;
  return best;
}

/** Gamma(alpha, 1) samples normalised to sum to 1. */
function dirichlet(size: number, alpha: number, random: Random): number[] {
  const samples = Array.from({ length: size }, () => gamma(alpha, random));
  const total = samples.reduce((a, b) => a + b, 0);
  return samples.map((s) => s / total);
}

/** Marsaglia-Tsang, with the standard boost for alpha below 1. */
function gamma(alpha: number, random: Random): number {
  if (alpha < 1) return gamma(alpha + 1, random) * Math.pow(random(), 1 / alpha);
  const d = alpha - 1 / 3;
  const c = 1 / Math.sqrt(9 * d);
  for (;;) {
    let x: number, v: number;
    do {
      x = normal(random);
      v = 1 + c * x;
    } while (v <= 0);
    v = v * v * v;
    const u = random();
    if (u < 1 - 0.0331 * x * x * x * x) return d * v;
    if (Math.log(u) < 0.5 * x * x + d * (1 - v + Math.log(v))) return d * v;
  }
}

function normal(random: Random): number {
  // Box-Muller; the guard keeps log(0) out.
  const u = Math.max(random(), Number.MIN_VALUE);
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * random());
}

/** A uniform evaluator: no knowledge at all, so search stands on its own. */
export class UniformEvaluator<S> implements Evaluator<S> {
  async evaluate(game: Game<S>, state: S): Promise<Evaluation> {
    const legal = game.legalActions(state);
    const count = legal.filter(Boolean).length;
    const priors = new Float32Array(game.actionSize);
    legal.forEach((ok, i) => {
      if (ok) priors[i] = 1 / count;
    });
    return { priors, value: 0 };
  }
}
