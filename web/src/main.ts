import "./style.css";
import { createGame } from "./games/registry";
import { LADDER, type LadderEntry } from "./games/ladder";
import type { Game } from "./games/types";
import { VIEWS, type View } from "./ui/views";
import type { FromEngine, ToEngine } from "./engine/protocol";

const asset = (path: string) => new URL(path, document.baseURI).href;

const LEVELS = [
  { label: "Casual", simulations: 40 },
  { label: "Steady", simulations: 200 },
  { label: "Careful", simulations: 600 },
];

interface ModelIndex {
  models: { game: string; generation?: number; parameters?: number }[];
}

let trained = new Set<string>();
let entry: LadderEntry = LADDER[0];
let game: Game<unknown> = createGame(entry.key);
let view: View = VIEWS[entry.key];
let moves: number[] = [];
let humanFirst = true;
let simulations = LEVELS[1].simulations;
let thinking = false;
let ready = false;
let engineInfo = "";
let report: Extract<FromEngine, { kind: "move" }> | null = null;
/** Half a turn, for games that ask for more than one decision. */
let pending: number | null = null;
let error: string | null = null;

const worker = new Worker(new URL("./engine/worker.ts", import.meta.url), {
  type: "module",
});

worker.onmessage = (event: MessageEvent<FromEngine>) => {
  const message = event.data;
  if (message.kind === "ready") {
    ready = true;
    error = null;
    const params = message.parameters
      ? `${(message.parameters / 1000).toFixed(0)}k parameters`
      : "";
    engineInfo = message.generation
      ? `Generation ${message.generation} · ${params}`
      : params;
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
  engineInfo = "";
  render();
  worker.postMessage({
    kind: "load",
    model: asset(`models/${entry.key}.onnx`),
    manifest: asset(`models/${entry.key}.json`),
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

/** True when the mover's only legal action is to pass. */
function mustPass(): boolean {
  const legal = game.legalActions(state());
  const actions = legal.flatMap((ok, i) => (ok ? [i] : []));
  return actions.length === 1 && actions[0] === game.actionSize - 1
    && !legal.slice(0, game.actionSize - 1).some(Boolean);
}

function play(action: number): void {
  if (thinking || !ready || finished() || !humanToMove()) return;
  if (!game.legalActions(state())[action]) return;
  report = null;
  pending = null;
  moves = [...moves, action];
  render();
  advance();
}

/**
 * Move the game on: let the engine think, or pass for a stuck player.
 *
 * Reversi can leave someone with nothing to do. Asking them to click a "pass"
 * button would be demanding a decision they do not have, so it is taken for them
 * after a beat - long enough that the turn does not appear to have been skipped.
 */
function advance(): void {
  if (!ready || thinking || finished()) return;

  if (humanToMove()) {
    if (mustPass()) {
      render();
      setTimeout(() => {
        moves = [...moves, game.actionSize - 1];
        render();
        advance();
      }, 750);
    }
    return;
  }

  thinking = true;
  render();
  worker.postMessage({ kind: "move", moves, simulations } satisfies ToEngine);
}

function reset(): void {
  moves = [];
  report = null;
  pending = null;
  thinking = false;
  render();
  advance();
}

function selectGame(next: LadderEntry): void {
  if (next.key === entry.key || !trained.has(next.key)) return;
  entry = next;
  game = createGame(entry.key);
  view = VIEWS[entry.key];
  moves = [];
  report = null;
  pending = null;
  thinking = false;
  document.title = `Caissa — ${entry.title}`;
  loadEngine();
}

// ------------------------------------------------------------------ skeleton

const app = document.getElementById("app")!;
app.innerHTML = `
  <header>
    <h1>Caissa</h1>
    <p class="tagline">Board games learned from scratch by self-play, playing in your browser.</p>
  </header>
  <nav class="games" id="games" role="tablist" aria-label="Game"></nav>
  <div class="layout">
    <div class="stage">
      <div class="board" id="board"></div>
      <div class="status" id="status"></div>
    </div>
    <aside class="side">
      <section class="panel">
        <h2>Opponent</h2>
        <div class="field">
          <label for="level">Difficulty</label>
          <select id="level">
            ${LEVELS.map(
              (l, i) =>
                `<option value="${l.simulations}"${i === 1 ? " selected" : ""}>${l.label}</option>`,
            ).join("")}
          </select>
        </div>
        <div class="field">
          <label for="first">You play</label>
          <select id="first"><option value="1" selected>First</option><option value="0">Second</option></select>
        </div>
        <button class="action" id="new">New game</button>
      </section>
      <section class="panel" id="analysis"></section>
      <section class="panel" id="about"></section>
    </aside>
  </div>
  <footer>
    Every move is computed on your device. Nothing is uploaded, and there is no server.
  </footer>
`;

const gamesEl = document.getElementById("games")!;
const boardEl = document.getElementById("board")!;
const statusEl = document.getElementById("status")!;
const analysisEl = document.getElementById("analysis")!;
const aboutEl = document.getElementById("about")!;

boardEl.addEventListener("click", (event) => {
  const target = (event.target as HTMLElement).closest<HTMLElement>(
    "[data-action], [data-square]",
  );
  if (!target) return;

  const selection = view.select(target, context());
  if (!selection) return;
  if (selection.kind === "action") {
    pending = null;
    play(selection.action);
  } else {
    pending = selection.pending;
    render();
  }
});
gamesEl.addEventListener("click", (event) => {
  const target = (event.target as HTMLElement).closest<HTMLElement>("[data-game]");
  const found = LADDER.find((g) => g.key === target?.dataset.game);
  if (found) selectGame(found);
});
document.getElementById("new")!.addEventListener("click", reset);
document.getElementById("level")!.addEventListener("change", (e) => {
  simulations = Number((e.target as HTMLSelectElement).value);
});
document.getElementById("first")!.addEventListener("change", (e) => {
  humanFirst = (e.target as HTMLSelectElement).value === "1";
  reset();
});

// ----------------------------------------------------------------- rendering

function renderTabs(): void {
  gamesEl.innerHTML = LADDER.map((item) => {
    const available = trained.has(item.key);
    const selected = item.key === entry.key;
    return `<button class="game-tab" role="tab" data-game="${item.key}"
      aria-selected="${selected}" ${available ? "" : "disabled"}
      title="${item.blurb}">${item.title}${
        available ? "" : `<span class="soon">soon</span>`
      }</button>`;
  }).join("");
}

function context() {
  const locked = thinking || finished() || !humanToMove() || !ready;
  return { game, state: state(), moves, humanFirst, locked, pending };
}

function render(): void {
  const ctx = context();

  renderTabs();
  boardEl.className = `board ${view.layout}${ctx.locked ? " locked" : ""}`;
  boardEl.innerHTML = view.board(ctx);
  statusEl.innerHTML = statusText() + view.detail(ctx);
  analysisEl.innerHTML = analysisPanel();
  aboutEl.innerHTML = `
    <h2>${entry.title}</h2>
    <p class="teaches"><strong>${entry.blurb}</strong>${entry.teaches}</p>
  `;
}

function statusText(): string {
  if (error) return `<span class="muted">Could not load the engine: ${error}</span>`;
  if (!ready) return `<span class="muted">Loading ${entry.title}…</span>`;

  const outcome = game.terminalValue(state());
  if (outcome !== null) {
    if (outcome === 0) return "Drawn.";
    // terminalValue is for the player to move: +1 means they are ahead.
    const humanWon = outcome > 0 === humanToMove();
    return humanWon
      ? `<span class="dot you"></span> You win.`
      : `<span class="dot engine"></span> The engine wins.`;
  }
  if (thinking) return `<span class="muted">Thinking…</span>`;
  if (humanToMove() && mustPass()) return `<span class="muted">No legal move — passing.</span>`;
  if (humanToMove() && pending !== null) {
    return `<span class="dot you"></span> Now choose a square to destroy.`;
  }
  return humanToMove()
    ? `<span class="dot you"></span> Your move.`
    : `<span class="dot engine"></span> Engine to move.`;
}

function analysisPanel(): string {
  if (!report) {
    return `<h2>Analysis</h2><p class="teaches">${
      ready ? "Appears after the engine's first move." : engineInfo || "Loading…"
    }</p>`;
  }
  // The engine reports its root value for itself; flipped so the bar always
  // reads from the human's point of view.
  const percent = ((-report.value + 1) / 2) * 100;

  return `
    <h2>Analysis</h2>
    <div class="evalbar"><span style="width:${percent.toFixed(1)}%"></span></div>
    <div class="meta"><span>You ${percent.toFixed(0)}%</span><span>Engine ${(100 - percent).toFixed(0)}%</span></div>
    ${view.visits(report.visits)}
    <div class="meta">
      <span>${report.visits.reduce((a, b) => a + b, 0)} simulations</span>
      <span>${Math.round(report.ms)} ms</span>
    </div>
    <div class="meta"><span>${engineInfo}</span></div>
  `;
}

async function start(): Promise<void> {
  try {
    const index: ModelIndex = await fetch(asset("models/index.json")).then((r) => r.json());
    trained = new Set(index.models.map((m) => m.game));
  } catch {
    // No index: fall back to offering the games that ship a view, so a missing
    // file degrades to "try it and see" rather than an empty menu.
    trained = new Set(Object.keys(VIEWS));
  }
  const first = LADDER.find((item) => trained.has(item.key));
  if (first) entry = first;
  game = createGame(entry.key);
  view = VIEWS[entry.key];
  document.title = `Caissa — ${entry.title}`;
  render();
  loadEngine();
}

render();
void start();
