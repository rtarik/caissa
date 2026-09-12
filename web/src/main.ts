import "./style.css";
import { createGame, TITLES } from "./games/registry";
import type { Game } from "./games/types";
import { VIEWS, type View } from "./ui/views";
import type { FromEngine, ToEngine } from "./engine/protocol";

const asset = (path: string) => new URL(path, document.baseURI).href;

const LEVELS = [
  { label: "Casual", simulations: 40 },
  { label: "Steady", simulations: 200 },
  { label: "Careful", simulations: 600 },
];

let name = "connect4";
let game: Game<unknown> = createGame(name);
let view: View = VIEWS[name];
let moves: number[] = [];
let humanFirst = true;
let simulations = LEVELS[1].simulations;
let thinking = false;
let ready = false;
let report: Extract<FromEngine, { kind: "move" }> | null = null;
let error: string | null = null;

const worker = new Worker(new URL("./engine/worker.ts", import.meta.url), {
  type: "module",
});

worker.onmessage = (event: MessageEvent<FromEngine>) => {
  const message = event.data;
  if (message.kind === "ready") {
    ready = true;
    error = null;
    describe(message.game, message.generation, message.parameters);
    render();
    advance();
  } else if (message.kind === "move") {
    thinking = false;
    report = message;
    moves = [...moves, message.action];
    render();
    advance();
  } else {
    thinking = false;
    error = message.message;
    render();
  }
};

function loadEngine(): void {
  ready = false;
  error = null;
  render();
  worker.postMessage({
    kind: "load",
    model: asset(`models/${name}.onnx`),
    manifest: asset(`models/${name}.json`),
  } satisfies ToEngine);
}

// ---------------------------------------------------------------- game state

const state = () => {
  let current = game.initialState();
  for (const move of moves) current = game.apply(current, move);
  return current;
};

const humanToMove = () => (moves.length % 2 === 0) === humanFirst;
const finished = () => game.terminalValue(state()) !== null;

/** The only legal action, when there is exactly one. Used to auto-pass. */
function forcedAction(): number | null {
  const legal = game.legalActions(state());
  const actions = legal.flatMap((ok, i) => (ok ? [i] : []));
  return actions.length === 1 ? actions[0] : null;
}

function play(action: number): void {
  if (thinking || !ready || finished() || !humanToMove()) return;
  if (!game.legalActions(state())[action]) return;
  report = null;
  moves = [...moves, action];
  render();
  advance();
}

/**
 * Move the game forward: let the engine think, or pass for a stuck human.
 *
 * Reversi can leave a player with nothing to do. Making them click a "pass"
 * button would be asking for a decision they do not have, so the pass is taken
 * for them - after a beat, so the board does not appear to skip their turn.
 */
function advance(): void {
  if (!ready || thinking || finished()) return;

  if (humanToMove()) {
    const forced = forcedAction();
    if (forced !== null && isPassOnly()) {
      render();
      setTimeout(() => {
        moves = [...moves, forced];
        render();
        advance();
      }, 700);
    }
    return;
  }

  thinking = true;
  render();
  worker.postMessage({ kind: "move", moves, simulations } satisfies ToEngine);
}

/** True when the mover's single legal action is a pass rather than a real move. */
function isPassOnly(): boolean {
  if (name !== "reversi") return false;
  const legal = game.legalActions(state());
  return !legal.slice(0, game.actionSize - 1).some(Boolean);
}

function reset(): void {
  moves = [];
  report = null;
  thinking = false;
  render();
  advance();
}

// ------------------------------------------------------------------ rendering

const app = document.getElementById("app")!;
app.innerHTML = `
  <header>
    <h1 id="title">Caissa</h1>
    <p class="subtitle" id="engine-info">Loading the engine…</p>
  </header>
  <div class="board" id="board"></div>
  <div class="status" id="status"></div>
  <div class="panel">
    <div class="row">
      <label for="game">Game</label>
      <select id="game">
        ${Object.entries(TITLES)
          .map(([key, label]) => `<option value="${key}">${label}</option>`)
          .join("")}
      </select>
    </div>
    <div class="row">
      <label for="level">Difficulty</label>
      <select id="level">
        ${LEVELS.map(
          (l, i) =>
            `<option value="${l.simulations}"${i === 1 ? " selected" : ""}>${l.label} · ${l.simulations} sims</option>`,
        ).join("")}
      </select>
    </div>
    <div class="row">
      <label for="first">You play</label>
      <select id="first"><option value="1" selected>First</option><option value="0">Second</option></select>
    </div>
    <div class="row"><button class="action" id="new">New game</button></div>
  </div>
  <div class="panel" id="analysis"></div>
  <footer>Trained by self-play. Runs entirely in your browser — nothing is sent anywhere.</footer>
`;

const boardEl = document.getElementById("board")!;
const statusEl = document.getElementById("status")!;
const analysisEl = document.getElementById("analysis")!;

boardEl.addEventListener("click", (event) => {
  const target = (event.target as HTMLElement).closest<HTMLElement>("[data-action]");
  if (!target) return;
  const action = view.actionFor(target);
  if (action !== null) play(action);
});
document.getElementById("new")!.addEventListener("click", reset);
document.getElementById("level")!.addEventListener("change", (e) => {
  simulations = Number((e.target as HTMLSelectElement).value);
});
document.getElementById("first")!.addEventListener("change", (e) => {
  humanFirst = (e.target as HTMLSelectElement).value === "1";
  reset();
});
document.getElementById("game")!.addEventListener("change", (e) => {
  name = (e.target as HTMLSelectElement).value;
  game = createGame(name);
  view = VIEWS[name];
  moves = [];
  report = null;
  thinking = false;
  loadEngine();
});

function describe(loaded: string, generation: number | undefined, parameters: number): void {
  const heading = `Caissa \u2014 ${TITLES[loaded] ?? loaded}`;
  document.getElementById("title")!.textContent = heading;
  document.title = heading;  // the tab should follow the game too
  const params = `${(parameters / 1000).toFixed(0)}k parameters`;
  document.getElementById("engine-info")!.textContent = generation
    ? `Self-play network, generation ${generation} · ${params}`
    : `Self-play network · ${params}`;
}

function render(): void {
  const locked = thinking || finished() || !humanToMove() || !ready;
  const context = { game, state: state(), moves, humanFirst, locked };

  boardEl.className = `board ${view.layout}${locked ? " locked" : ""}`;
  boardEl.innerHTML = view.board(context);
  statusEl.innerHTML = statusText();
  analysisEl.innerHTML = analysisText(view.detail(context));
}

function statusText(): string {
  if (error) return `<span class="thinking">Could not load the engine: ${error}</span>`;
  if (!ready) return `<span class="thinking">Loading the engine…</span>`;

  const outcome = game.terminalValue(state());
  if (outcome !== null) {
    if (outcome === 0) return "Drawn.";
    // terminalValue is for the player to move: +1 means they are ahead.
    const moverWon = outcome > 0;
    const humanWon = moverWon === humanToMove();
    return humanWon
      ? `<span class="dot you"></span> You win.`
      : `<span class="dot engine"></span> The engine wins.`;
  }
  if (thinking) return `<span class="thinking">Thinking…</span>`;
  if (humanToMove() && isPassOnly()) {
    return `<span class="thinking">No legal move — passing.</span>`;
  }
  return humanToMove()
    ? `<span class="dot you"></span> Your move.`
    : `<span class="dot engine"></span> Engine to move.`;
}

function analysisText(detail: string): string {
  const score = detail ? `<div class="meta score">${detail}</div>` : "";
  if (!report) {
    return score + `<div class="meta"><span>Engine analysis appears after its first move.</span></div>`;
  }
  // The engine reports its root value for itself; flipped so the bar always
  // reads from the human's point of view.
  const forYou = -report.value;
  const percent = ((forYou + 1) / 2) * 100;
  const best = Math.max(...report.visits);

  return `
    ${score}
    <div class="evalbar" style="--split:${Math.round(percent)}%"></div>
    <div class="meta">
      <span>You ${percent.toFixed(0)}%</span>
      <span>${report.visits.reduce((a, b) => a + b, 0)} simulations in ${Math.round(report.ms)} ms</span>
    </div>
    <div class="visits ${view.layout}">
      ${report.visits
        .slice(0, game.actionSize)
        .map((v) => `<div class="visit ${v === best && v > 0 ? "best" : ""}"
               style="height:${best > 0 ? Math.max(2, (v / best) * 100) : 2}%"
               title="${v} visits"></div>`)
        .join("")}
    </div>
    <div class="meta"><span>Where the search spent its time.</span></div>
  `;
}

loadEngine();
render();
