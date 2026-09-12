/**
 * The engine, off the main thread.
 *
 * Search is hundreds of network evaluations, and on a phone that is comfortably
 * long enough to freeze a page that ran it inline - taps ignored, animations
 * stopped, the tab reported as unresponsive. A worker keeps the board
 * interactive while the engine thinks.
 *
 * The worker is told the *moves so far* rather than a board, and replays them.
 * Sending a position would mean two copies of the game state that have to agree,
 * and the bug where they stop agreeing is silent.
 */
import { Connect4 } from "../games/connect4";
import { MCTS, argmax, DEFAULT_SEARCH } from "./mcts";
import { NetworkEvaluator } from "./onnx";
import type { FromEngine, ToEngine } from "./protocol";

const game = new Connect4();
let evaluator: NetworkEvaluator<unknown> | null = null;

function post(message: FromEngine): void {
  self.postMessage(message);
}

self.onmessage = async (event: MessageEvent<ToEngine>) => {
  const message = event.data;
  try {
    if (message.kind === "load") {
      evaluator = await NetworkEvaluator.load(message.model, message.manifest);
      post({
        kind: "ready",
        generation: evaluator.info.generation,
        parameters: evaluator.info.parameters,
      });
      return;
    }

    if (message.kind === "move") {
      if (!evaluator) throw new Error("engine asked to move before the model loaded");

      let state = game.initialState();
      for (const move of message.moves) state = game.apply(state, move);
      if (game.terminalValue(state) !== null) throw new Error("the game is already over");

      const search = new MCTS(game, evaluator as never, {
        ...DEFAULT_SEARCH,
        simulations: message.simulations,
      });

      const started = performance.now();
      const root = await search.search(state, false);
      // Temperature zero: the engine is trying to win, not generating training
      // data, so it takes the most-visited move rather than sampling.
      const policy = search.policy(root, 0);

      post({
        kind: "move",
        action: argmax(policy),
        value: root.value(),
        visits: search.visitCounts(root),
        ms: performance.now() - started,
      });
    }
  } catch (error) {
    post({ kind: "error", message: error instanceof Error ? error.message : String(error) });
  }
};
