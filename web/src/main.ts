/**
 * The site: a gallery of games, and a screen for playing one.
 *
 * Two audiences, one page. Someone who wants a game of Reversi should meet a
 * board and nothing else; someone who wants to know what the engine is thinking
 * should be able to ask. So the play screen shows the board, whose turn it is
 * and the result, and everything about *how* it plays - search budgets, visit
 * heatmaps, what each game taught the framework - sits behind a quiet toggle and
 * a page of its own.
 *
 * Routing is the URL hash, which costs nothing and buys the back button, a
 * bookmarkable game, and a reload that stays where you were.
 */
import "./style.css";
import { createGame } from "./games/registry";
import { LADDER, type LadderEntry } from "./games/ladder";
import type { Game } from "./games/types";
import { VIEWS, type View } from "./ui/views";
import { iconFor } from "./ui/icons";
import type { FromEngine, ToEngine } from "./engine/protocol";
import { networksByGame, type ModelEntry, type Network } from "./models";
import { LEVELS, DEFAULT_LEVEL } from "./levels";
import { parseRoute, type Route } from "./routes";

const asset = (path: string) => new URL(path, document.baseURI).href;

const DETAILS_KEY = "caissa:details";

interface ModelIndex {
  models: ModelEntry[];
}

let route: Route = { name: "gallery" };
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
let showDetails = false;
try {
  showDetails = localStorage.getItem(DETAILS_KEY) === "1";
} catch {
  // Private browsing, or storage blocked. The toggle still works for this visit.
}

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
  if (!ready || thinking || finished() || route.name !== "play") return;

  if (humanToMove()) {
    if (mustPass()) {
      render();
      setTimeout(() => {
        moves = [...moves, game.passAction!];
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

function reset(): void {
  moves = [];
  report = null;
  pending = null;
  thinking = false;
  render();
  advance();
}

// -------------------------------------------------------------------- routing

function applyRoute(): void {
  const next = parseRoute(location.hash);
  const changedGame = next.name === "play" && (route.name !== "play" || route.key !== next.key);
  route = next;

  if (route.name === "play") {
    const found = LADDER.find((item) => item.key === (route as { key: string }).key)!;
    document.title = `Caissa — ${found.title}`;
    if (changedGame) {
      entry = found;
      network = networks.get(entry.key)?.[0] ?? null;
      game = createGame(entry.key);
      view = VIEWS[entry.key];
      moves = [];
      report = null;
      pending = null;
      thinking = false;
      renderNetworks();
      render();
      loadEngine();
      return;
    }
  } else {
    document.title = route.name === "about" ? "Caissa — How it works" : "Caissa";
  }
  render();
}

// ------------------------------------------------------------------ skeleton

const app = document.getElementById("app")!;
app.innerHTML = `
  <header class="masthead">
    <a class="brand" href="#/" aria-label="All games">
      <span class="mark" aria-hidden="true"></span>
      <span class="wordmark">Caissa</span>
    </a>
    <nav class="masthead-links">
      <a href="#/how-it-works" class="quiet-link" id="about-link">How it works</a>
    </nav>
  </header>

  <main id="screens">
    <section class="screen" id="gallery">
      <h1 class="section-title">Choose a game</h1>
      <div class="cards" id="cards"></div>
    </section>

    <section class="screen" id="play" hidden>
      <div class="play-head">
        <a class="back" href="#/">
          <svg viewBox="0 0 24 24" aria-hidden="true" class="chevron"><path d="M15 5l-7 7 7 7"/></svg>
          All games
        </a>
        <h1 class="play-title" id="play-title"></h1>
        <p class="play-rule" id="play-rule"></p>
      </div>
      <div class="layout">
        <div class="stage">
          <div class="board" id="board"></div>
          <div class="status" id="status"></div>
        </div>
        <aside class="side">
          <section class="panel">
            <div class="field">
              <label for="level">Opponent</label>
              <select id="level">
                ${LEVELS.map(
                  (l, i) =>
                    `<option value="${l.simulations}"${i === DEFAULT_LEVEL ? " selected" : ""}>${l.label}</option>`,
                ).join("")}
              </select>
            </div>
            <p class="field-note" id="level-note">${LEVELS[DEFAULT_LEVEL].note}</p>
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
          <div class="details">
            <button class="quiet-link toggle" id="details-toggle" aria-expanded="false">
              Engine details
            </button>
            <section class="panel" id="analysis" hidden></section>
            <section class="panel" id="notes" hidden></section>
          </div>
        </aside>
      </div>
    </section>

    <section class="screen" id="about" hidden></section>
  </main>

  <footer>
    <span>Caissa</span>
    <a class="quiet-link" href="#/how-it-works">How it works</a>
    <a class="quiet-link" href="${asset("THIRD_PARTY_NOTICES.txt")}">Licences</a>
  </footer>
`;

const cardsEl = document.getElementById("cards")!;
const boardEl = document.getElementById("board")!;
const statusEl = document.getElementById("status")!;
const analysisEl = document.getElementById("analysis")!;
const notesEl = document.getElementById("notes")!;
const detailsEl = document.querySelector<HTMLElement>(".details")!;
const detailsToggle = document.getElementById("details-toggle") as HTMLButtonElement;
const seatEl = document.getElementById("first") as HTMLSelectElement;
const levelEl = document.getElementById("level") as HTMLSelectElement;
const levelNoteEl = document.getElementById("level-note")!;
const networkField = document.getElementById("network-field")!;
const networkEl = document.getElementById("network") as HTMLSelectElement;
const playTitleEl = document.getElementById("play-title")!;
const playRuleEl = document.getElementById("play-rule")!;
const aboutEl = document.getElementById("about")!;
const screens: Record<string, HTMLElement> = {
  gallery: document.getElementById("gallery")!,
  play: document.getElementById("play")!,
  about: aboutEl,
};

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
document.getElementById("new")!.addEventListener("click", reset);
levelEl.addEventListener("change", () => {
  simulations = Number(levelEl.value);
  levelNoteEl.textContent =
    LEVELS.find((l) => l.simulations === simulations)?.note ?? "";
});
seatEl.addEventListener("change", (e) => {
  humanFirst = (e.target as HTMLSelectElement).value === "1";
  reset();
});
detailsToggle.addEventListener("click", () => {
  showDetails = !showDetails;
  try {
    localStorage.setItem(DETAILS_KEY, showDetails ? "1" : "0");
  } catch {
    // Not being able to remember the choice is not a reason to refuse it.
  }
  render();
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
window.addEventListener("hashchange", applyRoute);

/** The choice of network, offered only when the game has more than one. */
function renderNetworks(): void {
  const choices = networks.get(entry.key) ?? [];
  networkField.hidden = choices.length < 2;
  networkEl.innerHTML = choices
    .map((n) => `<option value="${n.file}"${n.file === network?.file ? " selected" : ""}>${n.label}</option>`)
    .join("");
}

function context() {
  const locked = thinking || finished() || !humanToMove() || !ready;
  return { game, state: state(), moves, humanFirst, locked, pending };
}

// ------------------------------------------------------------------- screens

function renderCards(): void {
  cardsEl.innerHTML = LADDER.map((item) => {
    const available = trained.has(item.key);
    const tag = available ? "a" : "div";
    const href = available ? ` href="#/play/${item.key}"` : "";
    return `<${tag} class="card${available ? "" : " unavailable"}"${href} data-game="${item.key}">
      <span class="card-icon">${iconFor(item.key)}</span>
      <span class="card-text">
        <span class="card-title">${item.title}${
          available ? "" : `<span class="soon">soon</span>`
        }</span>
        <span class="card-blurb">${item.blurb}</span>
      </span>
    </${tag}>`;
  }).join("");
}

function renderAbout(): void {
  aboutEl.innerHTML = `
    <div class="prose">
      <h1>How it works</h1>
      <p class="lede">
        Six board games, and one opponent that taught itself to play all of them.
      </p>
      <p>
        Each game here is played by a neural network that was given the rules and
        nothing else — no openings, no strategy, no games by people. It learned by
        playing itself, millions of times, keeping what worked.
      </p>
      <h2>Two halves</h2>
      <p>
        The <strong>network</strong> looks at a position and answers two questions
        at once: which moves look worth considering, and who is winning. It is
        quick and it is often wrong.
      </p>
      <p>
        The <strong>search</strong> takes those hunches and checks them, playing
        out the most promising lines a few hundred times before choosing. The
        difficulty setting is simply how many of those lines it is allowed —
        from none at all to six hundred.
      </p>
      <h2>Learning with no teacher</h2>
      <p>
        Training is a loop. The agent plays itself; the search finds moves better
        than the network's first instinct; the network is trained to expect what
        the search found, and to predict how the game ended. A slightly better
        network makes a slightly better search, which produces slightly better
        training data, and round it goes. This is AlphaZero's recipe, run on one
        laptop rather than a data centre.
      </p>
      <h2>Why these six games</h2>
      <p>Each one was added because it breaks something the previous ones let slide.</p>
      <ul class="ladder">
        ${LADDER.map((item) => `<li>
          <span class="ladder-icon" data-game="${item.key}">${iconFor(item.key)}</span>
          <span><strong>${item.title}</strong> — ${item.teaches}</span>
        </li>`).join("")}
      </ul>
      <h2>How strong are the four levels?</h2>
      <p>
        Every level plays the one below it, sixty games each, same network on
        both sides. The numbers are Elo — the gap a rating system would put
        between them — and they only compare levels <em>within</em> one game.
      </p>
      <div id="levels-table"><p class="teaches">Measuring…</p></div>

      <h2>On your device</h2>
      <p>
        The networks are a few hundred kilobytes each and run in your browser.
        Nothing you play is sent anywhere, and there is no server to send it to.
      </p>
      <p class="prose-links">
        <a class="quiet-link" href="https://github.com/tarikrahmatallah/caissa">Source and write-up</a>
      </p>
    </div>
  `;
}

/** The measured ladder, fetched once and only when the page asks for it. */
let ladderTable: string | null = null;

async function fillLevels(): Promise<void> {
  const host = document.getElementById("levels-table");
  if (!host) return;
  if (ladderTable !== null) {
    host.innerHTML = ladderTable;
    return;
  }
  try {
    const data: {
      games: Record<string, { level: string; elo: number; score: number }[]>;
    } = await fetch(asset("levels.json")).then((r) => r.json());

    const steps = LEVELS.slice(1).map((l) => l.label);
    const rows = LADDER.filter((item) => data.games[item.key]).map((item) => {
      const cells = data.games[item.key]
        .map((rung) => {
          // A clean sweep has no upper end to report, only a floor.
          const gain = rung.score >= 1 ? "off the scale" : `+${Math.round(rung.elo)}`;
          return `<td class="gain">${gain}</td>`;
        })
        .join("");
      return `<tr><td>${item.title}</td>${cells}</tr>`;
    });
    ladderTable = `<table class="levels">
      <thead><tr><th>Game</th>${steps
        .map((name, index) => `<th>${name}<br><span class="muted">over ${LEVELS[index].label}</span></th>`)
        .join("")}</tr></thead>
      <tbody>${rows.join("")}</tbody>
    </table>`;
    host.innerHTML = ladderTable;
  } catch {
    // The file is optional; without it the page simply says less.
    host.innerHTML = "";
  }
}

function render(): void {
  for (const [name, element] of Object.entries(screens)) {
    element.hidden = route.name !== name;
  }
  document.body.dataset.screen = route.name;

  if (route.name === "gallery") {
    document.body.removeAttribute("data-game");
    renderCards();
    return;
  }
  if (route.name === "about") {
    document.body.removeAttribute("data-game");
    renderAbout();
    void fillLevels();
    return;
  }

  document.body.dataset.game = entry.key;
  const ctx = context();
  const [first, second] = entry.seats ?? ["First", "Second"];
  seatEl.options[0].text = first;
  seatEl.options[1].text = second;
  playTitleEl.innerHTML = `<span class="play-icon">${iconFor(entry.key)}</span>${entry.title}`;
  playRuleEl.textContent = entry.howToWin;

  boardEl.className = `board ${view.layout}${ctx.locked ? " locked" : ""}`;
  boardEl.innerHTML = view.board(ctx);
  statusEl.innerHTML = statusText() + view.detail(ctx);

  detailsToggle.setAttribute("aria-expanded", String(showDetails));
  detailsToggle.textContent = showDetails ? "Hide engine details" : "Engine details";
  detailsEl.classList.toggle("open", showDetails);
  analysisEl.hidden = !showDetails;
  const notes = showDetails ? view.notes?.(ctx) : undefined;
  notesEl.hidden = !notes;
  if (showDetails) {
    analysisEl.innerHTML = analysisPanel();
    notesEl.innerHTML = notes ?? "";
  }
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
  const requested = parseRoute(location.hash);
  const key = requested.name === "play" ? requested.key : null;
  entry = LADDER.find((item) => item.key === key && trained.has(item.key))
    ?? LADDER.find((item) => trained.has(item.key))
    ?? LADDER[0];
  network = networks.get(entry.key)?.[0] ?? null;
  game = createGame(entry.key);
  view = VIEWS[entry.key];
  renderNetworks();
  render();
  if (requested.name === "play") {
    route = { name: "play", key: entry.key };
    render();
    loadEngine();
  } else {
    applyRoute();
  }
}

render();
void start();
