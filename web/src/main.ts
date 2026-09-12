import "./style.css";
import { COLS, Connect4, ROWS } from "./games/connect4";
import { gridFromMoves, lastMoveIndex, winningLine } from "./ui/board";
import type { FromEngine, ToEngine } from "./engine/protocol";

const game = new Connect4();
const asset = (path: string) => new URL(path, document.baseURI).href;

const LEVELS = [
  { label: "Casual", simulations: 40 },
  { label: "Steady", simulations: 200 },
  { label: "Careful", simulations: 600 },
];

let moves: number[] = [];
let humanFirst = true;
let simulations = LEVELS[1].simulations;
let thinking = false;
let ready = false;
let lastReport: Extract<FromEngine, { kind: "move" }> | null = null;
let error: string | null = null;

const worker = new Worker(new URL("./engine/worker.ts", import.meta.url), {
  type: "module",
});

worker.onmessage = (event: MessageEvent<FromEngine>) => {
  const message = event.data;
  if (message.kind === "ready") {
    ready = true;
    describeEngine(message.generation, message.parameters);
    render();
    maybeEngineMove();
  } else if (message.kind === "move") {
    thinking = false;
    lastReport = message;
    moves = [...moves, message.action];
    render();
    maybeEngineMove();
  } else {
    thinking = false;
    error = message.message;
    render();
  }
};

worker.postMessage({
  kind: "load",
  model: asset("models/connect4.onnx"),
  manifest: asset("models/connect4.json"),
} satisfies ToEngine);

// ---------------------------------------------------------------- game state

const state = () => {
  let current = game.initialState();
  for (const move of moves) current = game.apply(current, move);
  return current;
};

const humanToMove = () => (moves.length % 2 === 0) === humanFirst;
const finished = () => game.terminalValue(state()) !== null;

function play(column: number): void {
  if (thinking || !ready || finished() || !humanToMove()) return;
  if (!game.legalActions(state())[column]) return;
  lastReport = null;
  moves = [...moves, column];
  render();
  maybeEngineMove();
}

function maybeEngineMove(): void {
  if (!ready || thinking || finished() || humanToMove()) return;
  thinking = true;
  render();
  worker.postMessage({ kind: "move", moves, simulations } satisfies ToEngine);
}

function reset(): void {
  moves = [];
  lastReport = null;
  error = null;
  thinking = false;
  render();
  maybeEngineMove();
}

// ------------------------------------------------------------------ rendering

const app = document.getElementById("app")!;
app.innerHTML = `
  <header>
    <h1>Caissa — Connect 4</h1>
    <p class="subtitle" id="engine-info">Loading the engine…</p>
  </header>
  <div class="board" id="board"></div>
  <div class="status" id="status"></div>
  <div class="panel">
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
      <select id="first">
        <option value="1" selected>First</option>
        <option value="0">Second</option>
      </select>
    </div>
    <div class="row"><button class="action" id="new">New game</button></div>
  </div>
  <div class="panel" id="analysis"></div>
  <footer>
    Trained by self-play. Runs entirely in your browser — nothing is sent anywhere.
  </footer>
`;

const boardEl = document.getElementById("board")!;
const statusEl = document.getElementById("status")!;
const analysisEl = document.getElementById("analysis")!;

boardEl.addEventListener("click", (event) => {
  const target = (event.target as HTMLElement).closest<HTMLElement>("[data-col]");
  if (target) play(Number(target.dataset.col));
});
document.getElementById("new")!.addEventListener("click", reset);
document.getElementById("level")!.addEventListener("change", (e) => {
  simulations = Number((e.target as HTMLSelectElement).value);
});
document.getElementById("first")!.addEventListener("change", (e) => {
  humanFirst = (e.target as HTMLSelectElement).value === "1";
  reset();
});

function describeEngine(generation: number | undefined, parameters: number): void {
  const info = document.getElementById("engine-info")!;
  const params = `${(parameters / 1000).toFixed(0)}k parameters`;
  info.textContent = generation
    ? `Self-play network, generation ${generation} · ${params}`
    : `Self-play network · ${params}`;
}

function render(): void {
  const grid = gridFromMoves(moves);
  const last = lastMoveIndex(moves);
  const line = new Set(winningLine(grid, last) ?? []);
  const legal = game.legalActions(state());
  const locked = thinking || finished() || !humanToMove() || !ready;

  boardEl.classList.toggle("locked", locked);
  boardEl.innerHTML = Array.from({ length: ROWS * COLS }, (_, i) => {
    const col = i % COLS;
    const owner = grid[i];
    const who = owner === 0 ? "" : (owner === 1) === humanFirst ? "you" : "engine";
    const classes = ["cell", who, line.has(i) ? "win" : "", i === last ? "last" : "",
                     !locked && legal[col] ? "playable" : ""];
    return `<button class="${classes.filter(Boolean).join(" ")}" data-col="${col}"
      ${locked || !legal[col] ? "disabled" : ""} aria-label="Column ${col + 1}"></button>`;
  }).join("");

  statusEl.innerHTML = statusText();
  analysisEl.innerHTML = analysisText();
}

function statusText(): string {
  if (error) return `<span class="thinking">Error: ${error}</span>`;
  if (!ready) return `<span class="thinking">Loading the engine…</span>`;

  const outcome = game.terminalValue(state());
  if (outcome !== null) {
    if (outcome === 0) return "Drawn — the board is full.";
    // terminalValue is -1 for the player to move, so the other side won.
    return humanToMove()
      ? `<span class="dot engine"></span> The engine wins.`
      : `<span class="dot you"></span> You win.`;
  }
  if (thinking) return `<span class="thinking">Thinking…</span>`;
  return humanToMove()
    ? `<span class="dot you"></span> Your move.`
    : `<span class="dot engine"></span> Engine to move.`;
}

function analysisText(): string {
  if (!lastReport) {
    return `<div class="meta"><span>Engine analysis appears after its first move.</span></div>`;
  }
  // The engine reports its root value for the side it was about to move - itself.
  // Flipped here so the bar always reads from the human's point of view.
  const forYou = -lastReport.value;
  const split = `${Math.round(((forYou + 1) / 2) * 100)}%`;
  const best = Math.max(...lastReport.visits);

  return `
    <div class="evalbar" style="--split:${split}"></div>
    <div class="meta">
      <span>You ${(((forYou + 1) / 2) * 100).toFixed(0)}%</span>
      <span>${lastReport.visits.reduce((a, b) => a + b, 0)} simulations in ${Math.round(lastReport.ms)} ms</span>
    </div>
    <div class="visits">
      ${lastReport.visits
        .map(
          (v) =>
            `<div class="visit ${v === best && v > 0 ? "best" : ""}" style="height:${
              best > 0 ? Math.max(2, (v / best) * 100) : 2
            }%" title="${v} visits"></div>`,
        )
        .join("")}
    </div>
    <div class="meta"><span>Where the search spent its time, by column.</span></div>
  `;
}

render();
