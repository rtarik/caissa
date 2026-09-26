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
import { playerCardHtml, tokenFor, type PlayerCard } from "./ui/players";
import { approximately, readSettings, resolveSeat, sheetHtml, type Rating, type Settings } from "./ui/sheet";
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
/** What the new-game sheet last chose, per visit; Random stays Random. */
let settings: Settings = { level: DEFAULT_LEVEL, seat: "first", network: null };
let sheetOpen = false;
let sheetCancellable = false;
/** Measured ratings per level, for the games that have them. */
let ratings = new Map<string, Map<string, Rating>>();
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
  if (!ready || thinking || finished() || route.name !== "play" || sheetOpen) return;

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
      settings = { ...settings, network: network?.file ?? null };
      render();
      loadEngine();
      openSheet(false);
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
        <a class="back" href="#/" aria-label="All games" title="All games">
          <svg viewBox="0 0 24 24" aria-hidden="true" class="chevron"><path d="M15 5l-7 7 7 7"/></svg>
        </a>
        <h1 class="play-title" id="play-title"></h1>
      </div>
      <div class="layout">
        <div class="stage">
          <div id="opponent"></div>
          <div class="board" id="board"></div>
          <div id="you"></div>
          <div class="status" id="status"></div>
        </div>
        <aside class="side">
          <section class="panel game-panel">
            <button class="button primary wide" id="new">New game</button>
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
      <div class="sheet-backdrop" id="sheet" hidden></div>
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
const opponentEl = document.getElementById("opponent")!;
const youEl = document.getElementById("you")!;
const sheetEl = document.getElementById("sheet")!;
const playTitleEl = document.getElementById("play-title")!;
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
document.getElementById("new")!.addEventListener("click", () => openSheet(true));
sheetEl.addEventListener("submit", (event) => {
  event.preventDefault();
  const form = event.target as HTMLFormElement;
  const chosen = readSettings(new FormData(form), settings);
  closeSheet();
  startGame(chosen);
});
sheetEl.addEventListener("click", (event) => {
  const target = event.target as HTMLElement;
  // The backdrop itself, or the Cancel button - not a click inside the sheet.
  if (target === sheetEl || target.closest("[data-sheet='cancel']")) closeSheet();
});
document.addEventListener("keydown", (event) => {
  if (event.key === "Escape" && sheetOpen) closeSheet();
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
window.addEventListener("hashchange", applyRoute);

/**
 * Open the new-game sheet over the board.
 *
 * Rendered here and not in render(): render runs on every engine message, and
 * re-rendering the form would throw away a choice the player is halfway through.
 */
function openSheet(cancellable: boolean): void {
  sheetCancellable = cancellable;
  drawSheet(settings);
  sheetOpen = true;
  sheetEl.hidden = false;
  // Everything behind the sheet stops taking focus and clicks until it closes.
  for (const element of document.querySelectorAll<HTMLElement>(".play-head, .layout")) {
    element.inert = true;
  }
  sheetEl.querySelector<HTMLButtonElement>("button[type=submit]")?.focus();
}

function drawSheet(chosen: Settings): void {
  sheetEl.innerHTML = sheetHtml({
    entry,
    settings: chosen,
    networks: (networks.get(entry.key) ?? []).map((n) => ({ file: n.file, label: n.label })),
    ratings: ratings.get(entry.key),
    cancellable: sheetCancellable,
  });
}

/**
 * Redraw an open sheet - when ratings arrive after it opened, say - keeping
 * whatever the player has already picked in it rather than the saved settings.
 */
function refreshSheet(): void {
  const form = sheetEl.querySelector<HTMLFormElement>("form");
  if (!sheetOpen || !form) return;
  const focused = document.activeElement === sheetEl.querySelector("button[type=submit]");
  drawSheet(readSettings(new FormData(form), settings));
  if (focused) sheetEl.querySelector<HTMLButtonElement>("button[type=submit]")?.focus();
}

function closeSheet(): void {
  sheetOpen = false;
  sheetEl.hidden = true;
  for (const element of document.querySelectorAll<HTMLElement>(".play-head, .layout")) {
    element.inert = false;
  }
  // Closing without choosing still starts the game that was waiting.
  render();
  advance();
}

function startGame(chosen: Settings): void {
  settings = chosen;
  simulations = LEVELS[settings.level].simulations;
  humanFirst = resolveSeat(settings.seat);
  const next = networks.get(entry.key)?.find((n) => n.file === settings.network);
  if (next && next.file !== network?.file) {
    // A different network is a different opponent: load it, then play.
    network = next;
    moves = [];
    report = null;
    pending = null;
    thinking = false;
    loadEngine();
    return;
  }
  reset();
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

const REPO = "https://github.com/rtarik/caissa";

/** A link to a file in the repository, so a claim can be checked. */
function code(path: string, label = path): string {
  return `<a class="code-link" href="${REPO}/blob/main/${path}">${label}</a>`;
}

function renderAbout(): void {
  const generations = new Map(
    [...networks.entries()].map(([key, list]) => [key, list[0]?.generation ?? null]),
  );

  aboutEl.innerHTML = `
    <div class="prose">
      <h1>How it works</h1>
      <p class="lede">
        Six board games, and one opponent that taught itself to play all of them.
      </p>
      <p>
        Nothing here was told how to play. Each game's engine was given the rules,
        a way to see the board, and no advice at all: no openings, no tactics, no
        games by people. It got good by playing itself and keeping what worked.
        This page is a walk through how that is done, and where each game made it
        harder.
      </p>

      <h2>1. What you are playing against</h2>
      <p>
        Two pieces, and it helps to keep them apart, because they fail in
        different ways.
      </p>
      <p>
        The <strong>network</strong> is a small neural net (a few hundred thousand
        numbers, ${code("src/caissa/network.py", "network.py")}). Shown a position,
        it answers two questions at once: which moves look worth considering, and
        who is winning. It answers instantly, and it is often wrong. Playing at
        <em>Beginner</em> is playing the network alone, with no thinking on top.
      </p>
      <p>
        The <strong>search</strong> is what turns those hunches into a move
        (${code("src/caissa/mcts.py", "mcts.py")}). It plays out promising lines
        one at a time, a few hundred times, keeping a count of how often each move
        was explored and how those lines turned out. Moves the network likes get
        looked at first; moves that keep working get looked at more. When the
        budget runs out, the move that was explored most is played.
      </p>
      <p>
        The difficulty setting is that budget, and nothing else. It is worth a lot:
        in Reversi, going from 50 lines to 200 is worth about as much as a large
        jump in playing strength, while the network itself never changes.
      </p>

      <h2>2. How it learned</h2>
      <p>A loop, repeated until the network stops improving.</p>
      <ol class="steps">
        <li>
          <strong>Play itself.</strong> A few hundred games, both sides the same
          network, with the search running on every move
          (${code("src/caissa/selfplay.py", "selfplay.py")}). Early moves are
          picked with deliberate randomness, or every game would be the same game.
        </li>
        <li>
          <strong>Write down two answers per position.</strong> What the search
          explored, and how the game actually ended for whoever was to move there.
        </li>
        <li>
          <strong>Train on them.</strong> The network is adjusted to expect what
          the search found, and to predict how games end
          (${code("src/caissa/train.py", "train.py")}).
        </li>
      </ol>
      <p>
        The whole thing rests on one idea: <strong>searching is better than
        guessing</strong>. The search, using the network, finds better moves than
        the network alone would play, so training the network to imitate the search
        makes it better, which makes the next search better, and round it goes.
        AlphaZero's recipe, on one laptop instead of a data centre.
      </p>
      <p>
        It can also fail, and it did here. If the network's sense of who is winning
        becomes confident faster than it becomes accurate, the search starts
        believing it over its own findings, and the loop teaches the network to
        imitate a worse teacher every round. Every number inside training keeps
        improving while the agent gets weaker, which is why strength is measured
        from outside, by playing old versions against new ones
        (${code("src/caissa/arena.py", "arena.py")}).
      </p>

      <h2>3. The six games, and what each one changed</h2>
      <p>
        The engine is the same for all of them: it never mentions a game by name,
        and everything it needs sits behind one interface
        (${code("src/caissa/games/base.py", "games/base.py")}). Each game was added
        because it breaks an assumption the previous ones let stand.
      </p>
      <div class="game-notes">
        ${LADDER.map((item) => `<section class="game-note">
          <h3>
            <span class="ladder-icon" data-game="${item.key}">${iconFor(item.key)}</span>
            ${item.title}
          </h3>
          <p>${item.detail}</p>
          <dl class="facts">
            <div><dt>Board</dt><dd>${item.board}</dd></div>
            <div><dt>Moves to choose from</dt><dd>${item.moves}</dd></div>
            <div><dt>How it learned</dt><dd>${item.learned}</dd></div>
            ${generations.get(item.key)
              ? `<div><dt>Version playing here</dt><dd>generation ${generations.get(item.key)}</dd></div>`
              : ""}
          </dl>
          <p class="rules-link">Rules: ${code(`src/caissa/games/${item.key}.py`, `${item.key}.py`)}</p>
        </section>`).join("")}
      </div>

      <h2>4. Chess is the exception</h2>
      <p>
        Every other game here started from nothing. Chess did not, and the reason
        is arithmetic: AlphaZero learned chess from scratch using roughly a million
        times the computing power available here. Starting from zero was not a
        method choice, it was a budget.
      </p>
      <p>
        So chess began by copying people. Lichess publishes every game played on
        the site under a public licence, about 14 GB for a single month. January
        2020 was downloaded and filtered down to the games where
        <strong>both players were rated 2200 or above</strong>, which is a strong
        club player: 530,000 games, 39 million positions
        (${code("scripts/lichess.py", "lichess.py")}). The network was then trained
        to predict the move the human played and how the game ended
        (${code("scripts/imitate.py", "imitate.py")}), and it reaches about half of
        their moves exactly.
      </p>
      <p>
        That gives a chess engine with a real opening repertoire and no experience
        of its own. Self-play from there is unfinished work: the first attempt made
        it measurably weaker, for reasons that took a day to pin down and are
        written up in ${code("PLAN.md")} under <em>Chess, self-play stage 1</em>.
        The chess opponent you can play is the imitation network, so it plays a bit
        like the people it learned from.
      </p>
      <h3 class="prose-sub">How strong is it?</h3>
      <p>
        Each level played Stockfish, the strongest open-source engine, told to play
        at a chosen strength: 1320 up to 2500, both colours, 24 games at every step
        (${code("scripts/stockfish.py", "stockfish.py")}). One rating per level is then
        fitted to all of its results at once, the rating that makes those scores most
        likely (${code("src/caissa/rating.py", "rating.py")}).
      </p>
      ${chessRatingsTable()}
      <p>
        Two honest caveats. Stockfish's strength setting is calibrated against other
        engines rather than people, so these sit on roughly the FIDE scale rather
        than on it. And the step from Strong to Master is worth about 100 points
        here, where the same two levels played against each other showed nearly
        400. The same network at two depths shares its blind spots, so the deeper
        search knows exactly where its twin will go wrong; an outside opponent does
        not make those tailored mistakes. Ratings measured inside one family of
        players stretch the gaps between them, which is why these came from
        outside it.
      </p>
      <p>
        One smaller difference worth naming: <strong>Gomoku</strong> was trained
        with a restriction, where stones could only be played next to existing
        ones. Without it, a search that knows nothing never stumbles into a
        finished game and there is nothing to learn from. That restriction is a
        training aid, not a rule, so it is switched off for the game you play, and
        the network turned out not to need it.
      </p>

      <h2>5. How strong are the four levels?</h2>
      <p>
        Every level plays the one below it, sixty games each, same network on both
        sides (${code("scripts/levels.py", "levels.py")}). The numbers are Elo, the
        gap a rating system would put between them, and they only compare levels
        <em>within</em> one game.
      </p>
      <div id="levels-table"><p class="teaches">Measuring…</p></div>
      <p>
        The spread is the interesting part. Search is worth far more in some games
        than others: a Reversi flip three moves ahead is invisible to the network
        and obvious to a search, while in Gomoku, with eighty-one places to put a
        stone, tripling the budget barely deepens anything.
      </p>

      <h2>6. On your device</h2>
      <p>
        Every move is computed on your device. Nothing you play is uploaded, and
        there is no server to upload it to. Each network is a few hundred kilobytes
        and runs in a background thread in your browser
        (${code("web/src/engine/worker.ts", "worker.ts")}); the rules exist twice,
        once in Python for training and once in TypeScript for playing, and both
        are checked against the same recorded positions so they cannot drift apart
        (${code("scripts/testvectors.py", "testvectors.py")}).
      </p>

      <h2>The code</h2>
      <p>
        The whole project is on GitHub, including a long write-up of what was
        tried, what was measured and what went wrong.
      </p>
      <p class="prose-links">
        <a class="quiet-link" href="${REPO}">github.com/rtarik/caissa</a>
        <a class="quiet-link" href="${REPO}/blob/main/PLAN.md">The plan and decision log</a>
      </p>
    </div>
  `;
}

/** The chess levels' measured ratings, as a table, once they have loaded. */
function chessRatingsTable(): string {
  const measured = ratings.get("chess");
  if (!measured?.size) return "";
  const rows = LEVELS.map((level) => {
    const rating = measured.get(level.label);
    return rating
      ? `<tr><td>${level.label}</td><td class="gain">about ${approximately(rating.rating)}</td>
         <td>${rating.low}–${rating.high}</td></tr>`
      : "";
  }).join("");
  return `<table class="levels">
    <thead><tr><th>Level</th><th>Rating</th><th>95% range</th></tr></thead>
    <tbody>${rows}</tbody>
  </table>`;
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
  playTitleEl.innerHTML = `<span class="play-icon">${iconFor(entry.key)}</span>${entry.title}`;

  const [engine, you] = playerCards();
  opponentEl.innerHTML = playerCardHtml(engine, tokenFor(entry.key, "engine", !humanFirst));
  youEl.innerHTML = playerCardHtml(you, tokenFor(entry.key, "you", humanFirst));

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

/**
 * The two players, as the cards above and below the board show them.
 *
 * Whose turn it is lives here now, on the card of the side to move, rather than
 * in a sentence under the board; the status line keeps what needs a sentence.
 */
function playerCards(): [PlayerCard, PlayerCard] {
  const level = LEVELS[settings.level];
  const rating = ratings.get(entry.key)?.get(level.label);
  const outcome = ready ? game.terminalValue(state()) : null;
  const over = outcome !== null;
  const humanWon = over && outcome !== 0 && (outcome > 0) === humanToMove();
  const engineWon = over && outcome !== 0 && !humanWon;
  const yourTurn = ready && !over && humanToMove();

  const [first, second] = entry.seats ?? ["", ""];
  const yourSide = entry.seats
    ? (humanFirst ? first : second)
    : (humanFirst ? "Moving first" : "Moving second");

  return [
    {
      side: "engine",
      name: `Caissa <span class="player-level">${level.label}</span>`,
      detail: rating ? `Rated about ${approximately(rating.rating)}` : level.note,
      status: thinking ? "Thinking…" : "",
      active: ready && !over && !humanToMove(),
      winner: engineWon,
    },
    {
      side: "you",
      name: "You",
      detail: yourSide,
      status: yourTurn && pending === null ? "Your move" : "",
      active: yourTurn,
      winner: humanWon,
    },
  ];
}

/** What needs a sentence: the result, a prompt, a pass, a bonus move, a problem. */
function statusText(): string {
  if (error) return `<span class="muted">The engine didn't load: ${error}</span>`;
  if (!ready) return `<span class="muted">Loading ${entry.title}…</span>`;

  const outcome = game.terminalValue(state());
  if (outcome !== null) {
    if (outcome === 0) return `<span class="result">Drawn.</span>`;
    // terminalValue is for the player to move: +1 means they are ahead.
    const humanWon = outcome > 0 === humanToMove();
    return humanWon
      ? `<span class="result">You win.</span>`
      : `<span class="result">Caissa wins.</span>`;
  }
  if (thinking && lastMoveKeptTurn()) {
    return `<span class="muted">Caissa closed a box and moves again.</span>`;
  }
  if (humanToMove() && mustPass()) return `<span class="muted">No legal move, so you pass.</span>`;
  if (humanToMove() && pending !== null) {
    return `<span>${view.prompt?.(context()) ?? "Now finish the move."}</span>`;
  }
  if (humanToMove() && lastMoveKeptTurn()) {
    return `<span>Box closed, so it's your move again.</span>`;
  }
  return "";
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
  settings = { ...settings, network: network?.file ?? null };
  void loadRatings();
  render();
  if (requested.name === "play") {
    route = { name: "play", key: entry.key };
    render();
    loadEngine();
    openSheet(false);
  } else {
    applyRoute();
  }
}

/**
 * Measured ratings, where they exist. Optional: without the file the level
 * cards simply say less, and nothing waits for it.
 */
async function loadRatings(): Promise<void> {
  try {
    const data: { levels: { label: string; rating: number; low: number; high: number }[] } =
      await fetch(asset("ratings.json")).then((r) => r.json());
    ratings = new Map([
      ["chess", new Map(data.levels.map((l) => [l.label, { rating: l.rating, low: l.low, high: l.high }]))],
    ]);
    // Both the play screen and the guide show them; the gallery does not.
    if (route.name !== "gallery") render();
    refreshSheet();
  } catch {
    ratings = new Map();
  }
}

render();
void start();
