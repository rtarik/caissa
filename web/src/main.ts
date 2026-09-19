import "./style.css";
import { createGame } from "./games/registry";
import { LADDER, type LadderEntry } from "./games/ladder";
import type { Game } from "./games/types";
import { VIEWS, type View } from "./ui/views";
import type { FromEngine, ToEngine } from "./engine/protocol";
import { networksByGame, type ModelEntry, type Network } from "./models";

const asset = (path: string) => new URL(path, document.baseURI).href;

const LEVELS = [
  // No search: the network's first instinct. Human-like for a network trained
  // on human games, and the plainest view of what any network has learned.
  { label: "Instinct", simulations: 0 },
  { label: "Casual", simulations: 40 },
  { label: "Steady", simulations: 200 },
  { label: "Careful", simulations: 600 },
];
const DEFAULT_LEVEL = 2;

interface ModelIndex {
  models: ModelEntry[];
}

let trained = new Set<string>();
/** Each game's networks, newest first, and the one being played. */
let networks = new Map<string, Network[]>();
let network: Network | null = null;
let entry: LadderEntry = LADDER[0];
let game: Game<unknown> = createGame(entry.key);
let view: View = VIEWS[entry.key];
let moves: number[] = [];
let humanFirst = true;
let simulations = LEVELS[DEFAULT_LEVEL].simulations;
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
    engineInfo = message.generation === 0
      ? `Untrained network · ${params}`
      : message.generation
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
  const file = network?.file ?? entry.key;
  worker.postMessage({
    kind: "load",
    model: asset(`models/${file}.onnx`),
    manifest: asset(`models/${file}.json`),
  } satisfies ToEngine);
}

// ---------------------------------------------------------------- game state

/**
 * The position, replayed from the moves - once per move list rather than on
 * every call, since a render asks several times and a chess position costs far
 * more to rebuild than a Four in a Row one. `moves` is replaced, never mutated,
 * so the array itself says whether the replay is current.
 */
let replayed: { moves: number[]; state: unknown } | null = null;
const state = () => {
  if (replayed?.moves !== moves) {
    let current = game.initialState();
    for (const move of moves) current = game.apply(current, move);
    replayed = { moves, state: current };
  }
  return replayed.state;
};

/** The seat the human plays: 0 moves first. */
const humanSeat = () => (humanFirst ? 0 : 1);
// Asked of the game rather than counted from the move list: after a Dots & Boxes
// bonus move the same player is to move again, and parity would say otherwise.
const humanToMove = () => game.toPlay(state()) === humanSeat();
const finished = () => game.terminalValue(state()) !== null;

/**
 * True when the mover's only legal action is the game's pass.
 *
 * Asked of the game rather than guessed from where the action sits. The first
 * version assumed the pass was the last action index, which is true for Reversi
 * and false everywhere else: in Four in a Row, with only the seventh column
 * open, it dropped the disc for you and announced that you had passed.
 */
function mustPass(): boolean {
  const pass = game.passAction;
  if (pass === undefined) return false;
  const legal = game.legalActions(state());
  return legal[pass] && legal.filter(Boolean).length === 1;
}

/** Whether the last move kept the turn - a closed box, in Dots & Boxes. */
function lastMoveKeptTurn(): boolean {
  if (moves.length === 0) return false;
  let before = game.initialState();
  for (const move of moves.slice(0, -1)) before = game.apply(before, move);
  return game.toPlay(before) === game.toPlay(state());
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
        moves = [...moves, game.passAction!];
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
  network = networks.get(entry.key)?.[0] ?? null;
  renderNetworks();
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
                `<option value="${l.simulations}"${i === DEFAULT_LEVEL ? " selected" : ""}>${l.label}</option>`,
            ).join("")}
          </select>
        </div>
        <div class="field">
          <label for="first">You play</label>
          <select id="first"><option value="1" selected>First</option><option value="0">Second</option></select>
        </div>
        <div class="field" id="network-field" hidden>
          <label for="network">Network</label>
          <select id="network"></select>
        </div>
        <button class="action" id="new">New game</button>
      </section>
      <section class="panel" id="analysis"></section>
      <section class="panel" id="notes" hidden></section>
      <section class="panel" id="about"></section>
    </aside>
  </div>
  <footer>
    Every move is computed on your device. Nothing is uploaded, and there is no server.
    <a href="${asset("THIRD_PARTY_NOTICES.txt")}">Licences</a>
  </footer>
`;

const gamesEl = document.getElementById("games")!;
const boardEl = document.getElementById("board")!;
const statusEl = document.getElementById("status")!;
const analysisEl = document.getElementById("analysis")!;
const notesEl = document.getElementById("notes")!;
const seatEl = document.getElementById("first") as HTMLSelectElement;
const networkField = document.getElementById("network-field")!;
const networkEl = document.getElementById("network") as HTMLSelectElement;
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
networkEl.addEventListener("change", () => {
  const chosen = networks.get(entry.key)?.find((n) => n.file === networkEl.value);
  if (!chosen || chosen.file === network?.file) return;
  // A different opponent is a different game: start again against it.
  network = chosen;
  moves = [];
  report = null;
  pending = null;
  thinking = false;
  loadEngine();
});

/** The choice of network, offered only when the game has more than one. */
function renderNetworks(): void {
  const choices = networks.get(entry.key) ?? [];
  networkField.hidden = choices.length < 2;
  networkEl.innerHTML = choices
    .map((n) => `<option value="${n.file}"${n.file === network?.file ? " selected" : ""}>${n.label}</option>`)
    .join("");
}

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
  const [first, second] = entry.seats ?? ["First", "Second"];
  seatEl.options[0].text = first;
  seatEl.options[1].text = second;
  boardEl.className = `board ${view.layout}${ctx.locked ? " locked" : ""}`;
  boardEl.innerHTML = view.board(ctx);
  statusEl.innerHTML = statusText() + view.detail(ctx);
  analysisEl.innerHTML = analysisPanel();
  const notes = view.notes?.(ctx);
  notesEl.hidden = !notes;
  notesEl.innerHTML = notes ?? "";
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
  if (thinking) {
    return lastMoveKeptTurn()
      ? `<span class="muted">The engine closed a box and moves again…</span>`
      : `<span class="muted">Thinking…</span>`;
  }
  if (humanToMove() && mustPass()) return `<span class="muted">No legal move — passing.</span>`;
  if (humanToMove() && pending !== null) {
    return `<span class="dot you"></span> ${view.prompt?.(context()) ?? "Now finish the move."}`;
  }
  if (humanToMove() && lastMoveKeptTurn()) {
    return `<span class="dot you"></span> Box closed — your move again.`;
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
    ${view.visits(report.visits, context())}
    <div class="meta">
      <span>${report.simulations > 0 ? `${report.simulations} simulations` : "Network only, no search"}</span>
      <span>${Math.round(report.ms)} ms</span>
    </div>
    <div class="meta"><span>${engineInfo}</span></div>
  `;
}

async function start(): Promise<void> {
  try {
    const index: ModelIndex = await fetch(asset("models/index.json")).then((r) => r.json());
    networks = networksByGame(index.models);
    trained = new Set(networks.keys());
  } catch {
    // No index: fall back to offering the games that ship a view, so a missing
    // file degrades to "try it and see" rather than an empty menu.
    trained = new Set(Object.keys(VIEWS));
  }
  const first = LADDER.find((item) => trained.has(item.key));
  if (first) entry = first;
  network = networks.get(entry.key)?.[0] ?? null;
  renderNetworks();
  game = createGame(entry.key);
  view = VIEWS[entry.key];
  document.title = `Caissa — ${entry.title}`;
  render();
  loadEngine();
}

render();
void start();
