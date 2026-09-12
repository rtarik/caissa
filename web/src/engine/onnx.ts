import * as ort from "onnxruntime-web/wasm";
import type { Game } from "../games/types";
import type { Evaluation, Evaluator } from "./mcts";

/**
 * Configure the WebAssembly runtime for static hosting.
 *
 * Multi-threaded WASM needs `SharedArrayBuffer`, which needs the page to be
 * served with `Cross-Origin-Opener-Policy: same-origin` and
 * `Cross-Origin-Embedder-Policy: require-corp`. GitHub Pages does not let you
 * set response headers, so those threads are simply not available.
 *
 * Asking for them anyway is worse than not asking: onnxruntime falls back to one
 * thread silently, and the only symptom is an engine that thinks more slowly
 * than expected. Requesting one thread explicitly makes the constraint visible
 * in the code rather than discovered in a profiler. SIMD needs no headers and is
 * widely supported, so that stays on.
 */
export function configureRuntime(): void {
  ort.env.wasm.numThreads = 1;
  ort.env.wasm.simd = true;
  // The .wasm location is deliberately left alone. Vite resolves onnxruntime's
  // own `new URL(..., import.meta.url)` import and emits the binary as a hashed
  // asset; overriding the path here would point at a second, unhashed copy and
  // ship 13 MB twice.
}

export interface Manifest {
  game: string;
  inputPlanes: number;
  boardShape: [number, number];
  actionSize: number;
  parameters: number;
  generation?: number;
  simulations?: number;
}

/**
 * Runs the exported network and hands search a masked distribution and a value.
 *
 * The masking happens here rather than in the search, exactly as in Python:
 * legality is a rule of the game, so search should be able to trust that what it
 * receives is already legal.
 */
export class NetworkEvaluator<S> implements Evaluator<S> {
  constructor(
    private session: ort.InferenceSession,
    private manifest: Manifest,
  ) {}

  static async load<S>(modelUrl: string, manifestUrl: string): Promise<NetworkEvaluator<S>> {
    configureRuntime();
    const [session, manifest] = await Promise.all([
      ort.InferenceSession.create(modelUrl, {
        executionProviders: ["wasm"],
        graphOptimizationLevel: "all",
      }),
      fetch(manifestUrl).then((r) => r.json() as Promise<Manifest>),
    ]);
    return new NetworkEvaluator<S>(session, manifest);
  }

  get info(): Manifest {
    return this.manifest;
  }

  async evaluate(game: Game<S>, state: S): Promise<Evaluation> {
    const [height, width] = game.boardShape;
    const input = new ort.Tensor("float32", game.encode(state), [
      1,
      game.inputPlanes,
      height,
      width,
    ]);
    const output = await this.session.run({ board: input });
    const logits = output.policy.data as Float32Array;
    const value = (output.value.data as Float32Array)[0];

    return { priors: maskedSoftmax(logits, game.legalActions(state)), value };
  }
}

/**
 * Softmax over legal moves only.
 *
 * Illegal moves get exactly zero rather than merely a small number, so search
 * cannot reach them at all. Subtracting the maximum before exponentiating is
 * what keeps a confident network's large logits from overflowing to Infinity.
 */
export function maskedSoftmax(logits: Float32Array, legal: boolean[]): Float32Array {
  let max = -Infinity;
  for (let i = 0; i < logits.length; i++) {
    if (legal[i] && logits[i] > max) max = logits[i];
  }

  const priors = new Float32Array(logits.length);
  let total = 0;
  for (let i = 0; i < logits.length; i++) {
    if (!legal[i]) continue;
    const weight = Math.exp(logits[i] - max);
    priors[i] = weight;
    total += weight;
  }
  for (let i = 0; i < priors.length; i++) priors[i] /= total;
  return priors;
}
