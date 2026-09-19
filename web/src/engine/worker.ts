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
import { createGame } from "../games/registry";
import { MCTS, argmax, DEFAULT_SEARCH } from "./mcts";
import { NetworkEvaluator } from "./onnx";
import type { Game } from "../games/types";
import type { FromEngine, ToEngine } from "./protocol";

let game: Game<unknown> | null = null;
let evaluator: NetworkEvaluator<unknown> | null = null;

function post(message: FromEngine): void {
  self.postMessage(message);
}

self.onmessage = async (event: MessageEvent<ToEngine>) => {
  const message = event.data;
  try {
    if (message.kind === "load") {
      evaluator = await NetworkEvaluator.load(message.model, message.manifest);
      // The manifest names the game, so the page and the engine cannot disagree
      // about which rules are in force.
      game = createGame(evaluator.info.game);
      post({
        kind: "ready",
        game: evaluator.info.game,
        generation: evaluator.info.generation,
        parameters: evaluator.info.parameters,
      });
      return;
    }

    if (message.kind === "move") {
      if (!evaluator || !game) throw new Error("engine asked to move before the model loaded");

      let state = game.initialState();
      for (const move of message.moves) state = game.apply(state, move);
      if (game.terminalValue(state) !== null) throw new Error("the game is already over");

      const started = performance.now();
      if (message.simulations === 0) {
        // Instinct: the network's own first choice, with no search at all. It is
        // how Maia plays human-like chess, and the most direct look at what a
        // network has learned rather than what search recovers on its behalf.
        const { priors, value } = await evaluator.evaluate(game, state);
        post({
          kind: "move",
          action: argmax(priors),
          value,
          visits: Array.from(priors),
          simulations: 0,
          ms: performance.now() - started,
        });
        return;
      }

      const search = new MCTS(game, evaluator as never, {
        ...DEFAULT_SEARCH,
        simulations: message.simulations,
      });
      const root = await search.search(state, false);
      // Temperature zero: the engine is trying to win, not generating training
      // data, so it takes the most-visited move rather than sampling.
      const policy = search.policy(root, 0);

      post({
        kind: "move",
        action: argmax(policy),
        value: root.value(),
        visits: search.visitCounts(root),
        simulations: message.simulations,
        ms: performance.now() - started,
      });
    }
  } catch (error) {
    post({ kind: "error", message: error instanceof Error ? error.message : String(error) });
  }
};
